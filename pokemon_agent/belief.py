from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .card_db import attack_data, card_data
from .deck_db import archetype_gameplan_profiles, expand_deck, get_deck, load_deck_db, rank_decks_by_observed_cards
from .state import AgentState


@dataclass(frozen=True)
class PrizeInfo:
    known_prized: Counter[int]
    possibly_prized: Counter[int]
    available_key_cards: dict[int, int]
    unknown_prize_count: int


@dataclass(frozen=True)
class OpponentBelief:
    archetype_probs: dict[str, float]
    top_deck_id: str | None
    top_archetype: str | None
    observed_cards: tuple[int, ...]
    prob_has_gust: float
    prob_has_switch: float
    prob_has_energy: float
    prob_can_attack_next_turn: float
    expected_damage_next_turn: float
    expected_first_attack_turn: float
    expected_game_end_turn: float
    plan_type: str | None
    main_attacker_ids: tuple[int, ...]


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
        visible_deck = Counter(_visible_deck_cards(obs, state.player_index))
        temporary_visible = Counter(_temporary_visible_cards(obs, state.player_index))
        hidden = _subtract_counts(deck_counts, visible, temporary_visible)
        possible = Counter(_take_from_counts(Counter(hidden), state.my_prizes_remaining, decklist))

        known = Counter()
        if visible_deck and sum(visible_deck.values()) >= state.my_deck_count:
            known = _subtract_counts(deck_counts, visible, visible_deck, temporary_visible)

        available: dict[int, int] = {}
        for card_id in _key_ids(self.config):
            available[card_id] = max(0, deck_counts.get(card_id, 0) - known.get(card_id, 0))

        return PrizeInfo(
            known_prized=known,
            possibly_prized=possible,
            available_key_cards=available,
            unknown_prize_count=max(0, state.my_prizes_remaining - sum(known.values())),
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


def _contains_name(card_id: int, parts: tuple[str, ...]) -> bool:
    data = card_data(card_id)
    name = (getattr(data, "name", "") or "").lower() if data else ""
    return any(part in name for part in parts)


def _prob_from_count(count: int, hand_count: int, hidden_count: int) -> float:
    if count <= 0 or hand_count <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - ((max(0, hidden_count - count) / max(1, hidden_count)) ** hand_count)))


class OpponentTracker:
    def __init__(self, db: dict[str, Any] | None = None):
        self.db = db or load_deck_db()
        self.profiles = archetype_gameplan_profiles(self.db)

    def update(self, obs: Any, state: AgentState) -> OpponentBelief:
        observed = _observed_opponent_cards(obs, state.opponent_index)
        ranked = rank_decks_by_observed_cards(observed, self.db, top_n=3)
        total = sum(max(0.01, row["score"] + 0.01) for row in ranked) or 1.0
        probs = {row["archetype"]: max(0.01, row["score"] + 0.01) / total for row in ranked}
        top = ranked[0] if ranked else None
        top_deck = get_deck(top["deck_id"], self.db) if top else None
        top_profile = self.profiles.get(top["archetype"]) if top else None
        counts = _count_deck(expand_deck(top_deck)) if top_deck else Counter()
        hidden_count = max(1, state.opp_deck_count + state.opp_hand_count + state.opp_prizes_remaining)
        gust_count = sum(count for cid, count in counts.items() if _contains_name(cid, ("boss", "catcher", "gust")))
        switch_count = sum(count for cid, count in counts.items() if _contains_name(cid, ("switch", "cart")))
        energy_count = sum(count for cid, count in counts.items() if card_data(cid) and card_data(cid).cardType in (5, 6))
        damage_ceiling = 0.0
        for pokemon in ([state.opp_active] if state.opp_active else []) + list(state.opp_bench):
            data = card_data(pokemon.id)
            for attack_id in getattr(data, "attacks", []) or []:
                attack = attack_data(attack_id)
                if attack is not None:
                    damage_ceiling = max(damage_ceiling, float(getattr(attack, "damage", 0) or 0))
        return OpponentBelief(
            archetype_probs=probs,
            top_deck_id=top["deck_id"] if top else None,
            top_archetype=top["archetype"] if top else None,
            observed_cards=tuple(observed),
            prob_has_gust=_prob_from_count(gust_count, state.opp_hand_count, hidden_count),
            prob_has_switch=_prob_from_count(switch_count, state.opp_hand_count, hidden_count),
            prob_has_energy=_prob_from_count(energy_count, state.opp_hand_count, hidden_count),
            prob_can_attack_next_turn=1.0 if state.opp_active and state.opp_active.energy_count > 0 else 0.35,
            expected_damage_next_turn=damage_ceiling,
            expected_first_attack_turn=float(top_profile.get("average_first_attack_turn", 3.0)) if top_profile else 3.0,
            expected_game_end_turn=float(top_profile.get("average_game_end_turn", 9.0)) if top_profile else 9.0,
            plan_type=str(top_profile.get("plan_type")) if top_profile else None,
            main_attacker_ids=tuple(int(card_id) for card_id in top_profile.get("main_attacker_ids", [])) if top_profile else (),
        )


class BeliefBuilder:
    def __init__(self, config: dict[str, Any], decklist: list[int], db: dict[str, Any] | None = None):
        self.config = config
        self.decklist = [int(card_id) for card_id in decklist]
        self.prizes = PrizeTracker(config)
        self.opponents = OpponentTracker(db)
        self.db = db or load_deck_db()

    def build(self, obs: Any, state: AgentState) -> BeliefState:
        prize_info = self.prizes.update(obs, state, self.decklist)
        opponent = self.opponents.update(obs, state)
        sample = self._hidden_sample(obs, state, opponent)
        return BeliefState(prize_info=prize_info, opponent=opponent, hidden_samples=(sample,))

    def _hidden_sample(self, obs: Any, state: AgentState, opponent: OpponentBelief) -> HiddenSample:
        your_player = obs.current.players[state.player_index]
        opp_player = obs.current.players[state.opponent_index]
        own_visible = Counter(_player_visible_cards(your_player, include_hand=True))
        own_visible_deck = Counter(_visible_deck_cards(obs, state.player_index))
        own_temporary_visible = Counter(_temporary_visible_cards(obs, state.player_index))
        own_hidden = _subtract_counts(_count_deck(self.decklist), own_visible, own_visible_deck, own_temporary_visible)
        own_counts = Counter(own_hidden)
        your_deck = [] if own_visible_deck else _take_from_counts(own_counts, state.my_deck_count, self.decklist)
        your_prize = _take_from_counts(own_counts, state.my_prizes_remaining, self.decklist)
        your_deck = _pad_zone(your_deck, 0 if own_visible_deck else state.my_deck_count, self.decklist)
        your_prize = _pad_zone(your_prize, state.my_prizes_remaining, self.decklist)

        opponent_decklist = self.decklist
        if opponent.top_deck_id:
            try:
                opponent_decklist = expand_deck(get_deck(opponent.top_deck_id, self.db))
            except KeyError:
                opponent_decklist = self.decklist
        opp_visible = Counter(_player_visible_cards(opp_player, include_hand=False))
        opp_counts = _subtract_counts(_count_deck(opponent_decklist), opp_visible)
        opp_order = opponent_decklist
        opponent_hand = _take_from_counts(opp_counts, state.opp_hand_count, opp_order)
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
