from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any, Iterable

from .card_db import attack_data, card_data
from .deck_db import archetype_gameplan_profiles, expand_deck, get_deck, load_deck_db, rank_decks_by_observed_cards
from .opponent_memory import EMPTY_OPPONENT_INTENT, OpponentIntentSnapshot
from .probability import ProbabilityModel, ProbabilitySnapshot
from .state import AgentState


@dataclass(frozen=True)
class PrizeInfo:
    known_prized: Counter[int]
    possibly_prized: Counter[int]
    available_key_cards: dict[int, int]
    unknown_prize_count: int
    unknown_prize_pool: Counter[int]
    known_deck_counts: Counter[int] = field(default_factory=Counter)


@dataclass(frozen=True)
class OpponentDeckHypothesis:
    deck_id: str
    archetype: str
    weight: float
    match_score: float
    original_card_counts: Counter[int]
    remaining_card_counts: Counter[int]


@dataclass(frozen=True)
class OpponentBelief:
    archetype_probs: dict[str, float]
    deck_hypotheses: tuple[OpponentDeckHypothesis, ...]
    top_deck_id: str | None
    top_archetype: str | None
    observed_cards: tuple[int, ...]
    observed_card_counts: Counter[int]
    known_hand_counts: Counter[int]
    remaining_card_counts: Counter[int]
    deck_confidence: float
    deck_entropy: float
    prob_in_hand_gust: float
    prob_in_hand_switch: float
    prob_in_hand_energy: float
    prob_in_hand_tool: float
    prob_has_gust: float
    prob_has_switch: float
    prob_has_energy: float
    prob_has_tool: float
    remaining_gust_count: float
    remaining_switch_count: float
    remaining_energy_count: float
    remaining_tool_count: float
    exhausted_gust: bool
    exhausted_switch: bool
    exhausted_tool: bool
    prob_can_attack_next_turn: float
    expected_damage_next_turn: float
    expected_first_attack_turn: float
    expected_game_end_turn: float
    plan_type: str | None
    main_attacker_ids: tuple[int, ...]
    intent: OpponentIntentSnapshot = EMPTY_OPPONENT_INTENT


@dataclass(frozen=True)
class HiddenSample:
    your_deck: tuple[int, ...]
    your_prize: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    opponent_prize: tuple[int, ...]
    opponent_hand: tuple[int, ...]
    opponent_active: tuple[int, ...] = ()


@dataclass(frozen=True)
class BeliefState:
    prize_info: PrizeInfo
    opponent: OpponentBelief
    probabilities: ProbabilitySnapshot = field(default_factory=ProbabilitySnapshot.empty)
    hidden_samples: tuple[HiddenSample, ...] = field(default_factory=tuple)

    @property
    def primary_hidden_sample(self) -> HiddenSample | None:
        return self.hidden_samples[0] if self.hidden_samples else None


def _card_id(card: Any | None) -> int | None:
    cid = getattr(card, "id", None)
    return int(cid) if cid is not None else None


def _cards(cards: Iterable[Any | None] | None) -> list[int]:
    ids: list[int] = []
    for card in cards or []:
        cid = _card_id(card)
        if cid is not None:
            ids.append(cid)
    return ids


def _pokemon_ids(pokemon: Any | None) -> list[int]:
    if pokemon is None:
        return []
    ids = [int(pokemon.id)]
    ids.extend(_cards(getattr(pokemon, "energyCards", None)))
    ids.extend(_cards(getattr(pokemon, "tools", None)))
    ids.extend(_cards(getattr(pokemon, "preEvolution", None)))
    return ids


def _player_visible_cards(player: Any, *, include_hand: bool) -> list[int]:
    ids: list[int] = []
    if include_hand:
        ids.extend(_cards(getattr(player, "hand", None)))
    ids.extend(_cards(getattr(player, "discard", None)))
    for pokemon in getattr(player, "active", []) or []:
        ids.extend(_pokemon_ids(pokemon))
    for pokemon in getattr(player, "bench", []) or []:
        ids.extend(_pokemon_ids(pokemon))
    ids.extend(_cards(card for card in getattr(player, "prize", []) or [] if card is not None))
    return ids


def _visible_prize_cards(player: Any) -> list[int]:
    return _cards(card for card in getattr(player, "prize", []) or [] if card is not None)


def _visible_deck_cards(obs: Any, player_index: int) -> list[int]:
    select = getattr(obs, "select", None)
    deck = getattr(select, "deck", None)
    if not deck:
        return []
    return [cid for card, cid in zip(deck, _cards(deck)) if getattr(card, "playerIndex", player_index) == player_index]


def _temporary_visible_cards(obs: Any, player_index: int) -> list[int]:
    ids: list[int] = []
    select = getattr(obs, "select", None)
    for attr in ("effect", "contextCard"):
        card = getattr(select, attr, None) if select is not None else None
        if card is not None and getattr(card, "playerIndex", player_index) == player_index:
            cid = _card_id(card)
            if cid is not None:
                ids.append(cid)
    current = getattr(obs, "current", None)
    for card in getattr(current, "looking", None) or []:
        if card is not None and getattr(card, "playerIndex", player_index) == player_index:
            cid = _card_id(card)
            if cid is not None:
                ids.append(cid)
    return ids


def _count_deck(decklist: Iterable[int]) -> Counter[int]:
    return Counter(int(card_id) for card_id in decklist)


def _positive_counts(counts: Counter[int]) -> Counter[int]:
    return Counter({card_id: count for card_id, count in counts.items() if count > 0})


def _subtract_counts(base: Counter[int], *others: Counter[int]) -> Counter[int]:
    result = Counter(base)
    for other in others:
        for card_id, count in other.items():
            result[card_id] -= count
    return _positive_counts(result)


def _take_from_counts(counts: Counter[int], count: int, order: Iterable[int]) -> list[int]:
    taken: list[int] = []
    for card_id in order:
        while counts.get(card_id, 0) > 0 and len(taken) < count:
            taken.append(int(card_id))
            counts[card_id] -= 1
        if len(taken) >= count:
            break
    return taken


def _pad_zone(zone: list[int], count: int, fallback_order: list[int]) -> list[int]:
    if len(zone) >= count:
        return zone[:count]
    for card_id in fallback_order:
        zone.append(int(card_id))
        if len(zone) >= count:
            break
    return zone[:count]


def _rotated(values: list[int], offset: int) -> list[int]:
    if not values:
        return values
    offset = offset % len(values)
    return values[offset:] + values[:offset]


def _key_ids(config: dict[str, Any]) -> set[int]:
    keys = (
        "main_attackers",
        "main_line",
        "backup_attackers",
        "setup_basics",
        "key_evolutions",
        "search_cards",
        "draw_cards",
        "gust_cards",
        "energy_cards",
    )
    return {int(card_id) for key in keys for card_id in config.get(key, [])}


class PrizeTracker:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def update(self, obs: Any, state: AgentState, decklist: list[int]) -> PrizeInfo:
        player = obs.current.players[state.player_index]
        deck_counts = _count_deck(decklist)
        visible = Counter(_player_visible_cards(player, include_hand=True))
        visible_prizes = Counter(_visible_prize_cards(player))
        visible_deck = Counter(_visible_deck_cards(obs, state.player_index))
        temporary_visible = Counter(_temporary_visible_cards(obs, state.player_index))
        hidden = _subtract_counts(deck_counts, visible, temporary_visible)

        known = Counter(visible_prizes)
        inferred_from_deck_reveal = Counter()
        if visible_deck and sum(visible_deck.values()) >= state.my_deck_count:
            inferred_from_deck_reveal = _subtract_counts(deck_counts, visible, visible_deck, temporary_visible)
            known.update(inferred_from_deck_reveal)

        known_deck_counts = Counter(visible_deck) if sum(visible_deck.values()) == state.my_deck_count else Counter()
        unknown_prize_count = max(0, state.my_prizes_remaining - sum(known.values()))
        unknown_prize_pool = _subtract_counts(hidden, visible_deck, inferred_from_deck_reveal)
        possible = Counter(_take_from_counts(Counter(unknown_prize_pool), unknown_prize_count, decklist))

        available: dict[int, int] = {}
        for card_id in _key_ids(self.config):
            available[card_id] = max(0, deck_counts.get(card_id, 0) - known.get(card_id, 0))

        return PrizeInfo(
            known_prized=known,
            possibly_prized=possible,
            available_key_cards=available,
            unknown_prize_count=unknown_prize_count,
            unknown_prize_pool=unknown_prize_pool,
            known_deck_counts=known_deck_counts,
        )


def _observed_opponent_cards(obs: Any, opponent_index: int) -> list[int]:
    if obs.current is None:
        return []
    player = obs.current.players[opponent_index]
    observed = _player_visible_cards(player, include_hand=False)
    for log in getattr(obs, "logs", []) or []:
        if getattr(log, "playerIndex", None) == opponent_index and getattr(log, "cardId", None) is not None:
            observed.append(int(log.cardId))
        if getattr(log, "playerIndex", None) == opponent_index and getattr(log, "cardIdAfter", None) is not None:
            observed.append(int(log.cardIdAfter))
    return observed


def _observed_opponent_card_counts(obs: Any, opponent_index: int, memory: Any | None = None) -> Counter[int]:
    observed = Counter(_observed_opponent_cards(obs, opponent_index))
    if memory is None:
        return observed
    for card_id, count in getattr(memory, "observed_card_ids", {}).items():
        observed[int(card_id)] = max(int(observed.get(int(card_id), 0)), int(count))
    return observed


def _contains_name(card_id: int, parts: tuple[str, ...]) -> bool:
    data = card_data(card_id)
    name = (getattr(data, "name", "") or "").lower() if data else ""
    return any(part in name for part in parts)


def _is_energy_card(card_id: int) -> bool:
    data = card_data(card_id)
    return bool(data and data.cardType in (5, 6))


def _is_tool_card(card_id: int) -> bool:
    data = card_data(card_id)
    name = (getattr(data, "name", "") or "").lower() if data else ""
    return bool(data and (data.cardType == 4 or any(part in name for part in ("tool", "cape", "belt", "band"))))


def _remaining_category_count(counts: Counter[int], predicate: Any) -> float:
    return float(sum(count for cid, count in counts.items() if predicate(int(cid))))


def _prob_from_count(count: float, hand_count: int, hidden_count: int) -> float:
    if count <= 0 or hand_count <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - ((max(0, hidden_count - count) / max(1, hidden_count)) ** hand_count)))


def _access_after_draw(hand_probability: float, remaining_count: float, hidden_count: int) -> float:
    draw_probability = max(0.0, min(1.0, float(remaining_count) / max(1.0, float(hidden_count))))
    return max(0.0, min(1.0, 1.0 - ((1.0 - hand_probability) * (1.0 - draw_probability))))


def _normalized_entropy(probs: dict[str, float]) -> float:
    values = [float(value) for value in probs.values() if value > 0.0]
    if len(values) <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in values)
    return max(0.0, min(1.0, entropy / math.log(len(values))))


def _hypothesis_weights(ranked: list[dict[str, Any]]) -> list[float]:
    raw = [max(0.01, float(row["score"]) + 0.01) for row in ranked]
    total = sum(raw) or 1.0
    return [value / total for value in raw]


def _weighted_remaining_counts(hypotheses: Iterable[OpponentDeckHypothesis]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for hypothesis in hypotheses:
        for card_id, count in hypothesis.remaining_card_counts.items():
            counts[int(card_id)] += float(hypothesis.weight) * float(count)
    return _positive_counts(counts)


def _robustly_exhausted(
    hypotheses: tuple[OpponentDeckHypothesis, ...],
    category: Any,
    *,
    high_confidence: float = 0.85,
) -> bool:
    if not hypotheses:
        return False
    meaningful = tuple(hypothesis for hypothesis in hypotheses if hypothesis is hypotheses[0] or hypothesis.weight >= 0.10)
    original = [
        _remaining_category_count(hypothesis.original_card_counts, category)
        for hypothesis in meaningful
    ]
    remaining = [
        _remaining_category_count(hypothesis.remaining_card_counts, category)
        for hypothesis in meaningful
    ]
    if any(count > 0 for count in original) and all(count <= 0 for count in remaining):
        return True
    top = hypotheses[0]
    return bool(
        top.weight >= high_confidence
        and _remaining_category_count(top.original_card_counts, category) > 0
        and _remaining_category_count(top.remaining_card_counts, category) <= 0
    )


class OpponentTracker:
    def __init__(self, db: dict[str, Any] | None = None):
        self.db = db or load_deck_db()
        self.profiles = archetype_gameplan_profiles(self.db)

    def update(self, obs: Any, state: AgentState, memory: Any | None = None) -> OpponentBelief:
        observed_counts = _observed_opponent_card_counts(obs, state.opponent_index, memory)
        observed = list(observed_counts.elements())
        ranked = rank_decks_by_observed_cards(observed, self.db, top_n=3)
        weights = _hypothesis_weights(ranked)
        hypotheses: list[OpponentDeckHypothesis] = []
        probs: dict[str, float] = {}
        for row, weight in zip(ranked, weights):
            deck = get_deck(row["deck_id"], self.db)
            deck_counts = _count_deck(expand_deck(deck)) if deck else Counter()
            hypothesis = OpponentDeckHypothesis(
                deck_id=str(row["deck_id"]),
                archetype=str(row["archetype"]),
                weight=float(weight),
                match_score=float(row["score"]),
                original_card_counts=deck_counts,
                remaining_card_counts=_subtract_counts(deck_counts, observed_counts),
            )
            hypotheses.append(hypothesis)
            probs[hypothesis.archetype] = probs.get(hypothesis.archetype, 0.0) + hypothesis.weight
        hypothesis_tuple = tuple(hypotheses)
        top = ranked[0] if ranked else None
        top_profile = self.profiles.get(top["archetype"]) if top else None
        remaining_counts = _weighted_remaining_counts(hypothesis_tuple)
        hidden_count = max(1, state.opp_deck_count + state.opp_hand_count + state.opp_prizes_remaining)
        remaining_gust = _remaining_category_count(remaining_counts, lambda cid: _contains_name(cid, ("boss", "catcher", "gust")))
        remaining_switch = _remaining_category_count(remaining_counts, lambda cid: _contains_name(cid, ("switch", "cart")))
        remaining_energy = _remaining_category_count(remaining_counts, _is_energy_card)
        remaining_tool = _remaining_category_count(remaining_counts, _is_tool_card)
        known_hand_counts = memory.known_hand_counts() if memory is not None else Counter()
        base_hand = {
            "gust": _prob_from_count(remaining_gust, state.opp_hand_count, hidden_count),
            "switch": _prob_from_count(remaining_switch, state.opp_hand_count, hidden_count),
            "energy": _prob_from_count(remaining_energy, state.opp_hand_count, hidden_count),
            "tool": _prob_from_count(remaining_tool, state.opp_hand_count, hidden_count),
        }
        remaining_by_category = {
            "gust": remaining_gust,
            "switch": remaining_switch,
            "energy": remaining_energy,
            "tool": remaining_tool,
        }
        hand_probability = dict(base_hand)
        if memory is not None:
            for category, probability in base_hand.items():
                draw_probability = max(0.0, min(1.0, remaining_by_category[category] / max(1.0, float(hidden_count))))
                hand_probability[category] = memory.adjusted_hand_probability(
                    category,
                    probability,
                    draw_probability=draw_probability,
                    current_turn=state.turn,
                )
        if _remaining_category_count(known_hand_counts, lambda cid: _contains_name(cid, ("boss", "catcher", "gust"))) > 0:
            hand_probability["gust"] = 1.0
        if _remaining_category_count(known_hand_counts, lambda cid: _contains_name(cid, ("switch", "cart"))) > 0:
            hand_probability["switch"] = 1.0
        if _remaining_category_count(known_hand_counts, _is_energy_card) > 0:
            hand_probability["energy"] = 1.0
        if _remaining_category_count(known_hand_counts, _is_tool_card) > 0:
            hand_probability["tool"] = 1.0
        damage_ceiling = 0.0
        for pokemon in ([state.opp_active] if state.opp_active else []) + list(state.opp_bench):
            data = card_data(pokemon.id)
            for attack_id in getattr(data, "attacks", []) or []:
                attack = attack_data(attack_id)
                if attack is not None:
                    damage_ceiling = max(damage_ceiling, float(getattr(attack, "damage", 0) or 0))
        main_attacker_ids = tuple(int(card_id) for card_id in top_profile.get("main_attacker_ids", [])) if top_profile else ()
        intent = memory.snapshot(state, main_attacker_ids) if memory is not None else EMPTY_OPPONENT_INTENT
        return OpponentBelief(
            archetype_probs=probs,
            deck_hypotheses=hypothesis_tuple,
            top_deck_id=top["deck_id"] if top else None,
            top_archetype=top["archetype"] if top else None,
            observed_cards=tuple(observed),
            observed_card_counts=observed_counts,
            known_hand_counts=known_hand_counts,
            remaining_card_counts=remaining_counts,
            deck_confidence=hypothesis_tuple[0].weight if hypothesis_tuple else 0.0,
            deck_entropy=_normalized_entropy({hypothesis.deck_id: hypothesis.weight for hypothesis in hypothesis_tuple}),
            prob_in_hand_gust=hand_probability["gust"],
            prob_in_hand_switch=hand_probability["switch"],
            prob_in_hand_energy=hand_probability["energy"],
            prob_in_hand_tool=hand_probability["tool"],
            prob_has_gust=_access_after_draw(hand_probability["gust"], remaining_gust, hidden_count),
            prob_has_switch=_access_after_draw(hand_probability["switch"], remaining_switch, hidden_count),
            prob_has_energy=_access_after_draw(hand_probability["energy"], remaining_energy, hidden_count),
            prob_has_tool=_access_after_draw(hand_probability["tool"], remaining_tool, hidden_count),
            remaining_gust_count=remaining_gust,
            remaining_switch_count=remaining_switch,
            remaining_energy_count=remaining_energy,
            remaining_tool_count=remaining_tool,
            exhausted_gust=_robustly_exhausted(hypothesis_tuple, lambda cid: _contains_name(cid, ("boss", "catcher", "gust"))),
            exhausted_switch=_robustly_exhausted(hypothesis_tuple, lambda cid: _contains_name(cid, ("switch", "cart"))),
            exhausted_tool=_robustly_exhausted(hypothesis_tuple, _is_tool_card),
            prob_can_attack_next_turn=1.0 if state.opp_active and state.opp_active.energy_count > 0 else 0.35,
            expected_damage_next_turn=damage_ceiling,
            expected_first_attack_turn=float(top_profile.get("average_first_attack_turn", 3.0)) if top_profile else 3.0,
            expected_game_end_turn=float(top_profile.get("average_game_end_turn", 9.0)) if top_profile else 9.0,
            plan_type=str(top_profile.get("plan_type")) if top_profile else None,
            main_attacker_ids=main_attacker_ids,
            intent=intent,
        )


class BeliefBuilder:
    def __init__(self, config: dict[str, Any], decklist: list[int], db: dict[str, Any] | None = None, num_hidden_samples: int = 1):
        self.config = config
        self.decklist = [int(card_id) for card_id in decklist]
        self.num_hidden_samples = max(1, int(num_hidden_samples))
        self.prizes = PrizeTracker(config)
        self.opponents = OpponentTracker(db)
        self.probabilities = ProbabilityModel(config)
        self.db = db or load_deck_db()

    def build(self, obs: Any, state: AgentState, memory: Any | None = None) -> BeliefState:
        prize_info = self.prizes.update(obs, state, self.decklist)
        opponent = self.opponents.update(obs, state, memory=memory)
        probabilities = self.probabilities.evaluate(
            state=state,
            deck_counts=_count_deck(self.decklist),
            known_prized=prize_info.known_prized,
            unknown_prize_pool=prize_info.unknown_prize_pool,
            unknown_prize_count=prize_info.unknown_prize_count,
            known_deck_counts=prize_info.known_deck_counts,
        )
        samples = tuple(self._hidden_sample(obs, state, opponent, variant=index) for index in range(self.num_hidden_samples))
        return BeliefState(prize_info=prize_info, opponent=opponent, probabilities=probabilities, hidden_samples=samples)

    def _hidden_sample(self, obs: Any, state: AgentState, opponent: OpponentBelief, variant: int = 0) -> HiddenSample:
        your_player = obs.current.players[state.player_index]
        opp_player = obs.current.players[state.opponent_index]
        own_visible = Counter(_player_visible_cards(your_player, include_hand=True))
        own_visible_deck = Counter(_visible_deck_cards(obs, state.player_index))
        own_temporary_visible = Counter(_temporary_visible_cards(obs, state.player_index))
        own_hidden = _subtract_counts(_count_deck(self.decklist), own_visible, own_visible_deck, own_temporary_visible)
        own_counts = Counter(own_hidden)
        own_order = _rotated(self.decklist, variant)
        your_deck = [] if own_visible_deck else _take_from_counts(own_counts, state.my_deck_count, own_order)
        your_prize = _take_from_counts(own_counts, state.my_prizes_remaining, own_order)
        your_deck = _pad_zone(your_deck, 0 if own_visible_deck else state.my_deck_count, own_order)
        your_prize = _pad_zone(your_prize, state.my_prizes_remaining, own_order)

        opponent_decklist = self.decklist
        sampled_hypothesis = opponent.deck_hypotheses[0] if opponent.deck_hypotheses else None
        if variant > 0 and opponent.deck_hypotheses:
            sampled_hypothesis = opponent.deck_hypotheses[variant % len(opponent.deck_hypotheses)]
        if sampled_hypothesis is not None:
            try:
                opponent_decklist = expand_deck(get_deck(sampled_hypothesis.deck_id, self.db))
            except KeyError:
                opponent_decklist = self.decklist
        opp_visible = Counter(opponent.observed_card_counts)
        opp_counts = _subtract_counts(_count_deck(opponent_decklist), opp_visible)
        opp_order = _rotated(opponent_decklist, variant)
        known_hand = list(opponent.known_hand_counts.elements())[: state.opp_hand_count]
        opponent_hand = known_hand + _take_from_counts(opp_counts, max(0, state.opp_hand_count - len(known_hand)), opp_order)
        opponent_prize = _take_from_counts(opp_counts, state.opp_prizes_remaining, opp_order)
        opponent_deck = _take_from_counts(opp_counts, state.opp_deck_count, opp_order)
        opponent_hand = _pad_zone(opponent_hand, state.opp_hand_count, opp_order)
        opponent_prize = _pad_zone(opponent_prize, state.opp_prizes_remaining, opp_order)
        opponent_deck = _pad_zone(opponent_deck, state.opp_deck_count, opp_order)
        opponent_active: list[int] = []
        active = getattr(opp_player, "active", []) or []
        if active and active[0] is None:
            for card_id in opponent_decklist:
                data = card_data(card_id)
                if data and data.cardType == 0 and getattr(data, "basic", False):
                    opponent_active = [int(card_id)]
                    break
        return HiddenSample(
            your_deck=tuple(your_deck),
            your_prize=tuple(your_prize),
            opponent_deck=tuple(opponent_deck),
            opponent_prize=tuple(opponent_prize),
            opponent_hand=tuple(opponent_hand),
            opponent_active=tuple(opponent_active),
        )
