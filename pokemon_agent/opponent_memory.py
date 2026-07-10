from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Iterable

from .card_db import card_data, card_name, cards_by_id, is_pokemon, is_trainer
from .state import AgentState, CardView, PokemonView


def _norm_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _log_type_name(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    try:
        from cg.api import LogType

        return LogType(value).name
    except Exception:
        return str(value)


def _card_instances_from_log(log: Any) -> tuple[tuple[int, int | None], ...]:
    pairs = (
        ("cardId", "serial"),
        ("cardIdBefore", "serialBefore"),
        ("cardIdAfter", "serialAfter"),
        ("cardIdActive", "serialActive"),
        ("cardIdBench", "serialBench"),
        ("cardIdTarget", "serialTarget"),
    )
    instances: list[tuple[int, int | None]] = []
    for card_attr, serial_attr in pairs:
        card_id = getattr(log, card_attr, None)
        if card_id is None:
            continue
        serial = getattr(log, serial_attr, None)
        instances.append((int(card_id), int(serial) if serial is not None else None))
    return tuple(instances)


def _area_type_name(value: Any) -> str:
    if value is None:
        return "NONE"
    try:
        from cg.api import AreaType

        return AreaType(value).name
    except Exception:
        return str(value)


def _pokemon_views(state: AgentState) -> tuple[PokemonView, ...]:
    return tuple(([state.opp_active] if state.opp_active else []) + list(state.opp_bench))


def _visible_discard_key(card: CardView) -> tuple[int, int | None]:
    return (int(card.id), None if card.serial is None else int(card.serial))


def _decayed_score(turn: int, event_turn: int, *, horizon: int = 4) -> float:
    age = max(0, int(turn) - int(event_turn))
    if age > horizon:
        return 0.0
    return max(0.0, 1.0 - (float(age) / float(horizon + 1)))


@lru_cache(maxsize=1)
def _name_to_card_ids() -> dict[str, tuple[int, ...]]:
    by_name: dict[str, list[int]] = {}
    for card_id, data in cards_by_id().items():
        by_name.setdefault(_norm_name(getattr(data, "name", "")), []).append(int(card_id))
    return {name: tuple(ids) for name, ids in by_name.items()}


@lru_cache(maxsize=None)
def _ancestor_names(card_id: int) -> tuple[str, ...]:
    data = card_data(card_id)
    parent = _norm_name(getattr(data, "evolvesFrom", None)) if data is not None else ""
    if not parent:
        return ()
    names: set[str] = set()

    def visit(name: str) -> None:
        if not name or name in names:
            return
        names.add(name)
        for parent_id in _name_to_card_ids().get(name, ()):
            parent_data = card_data(parent_id)
            visit(_norm_name(getattr(parent_data, "evolvesFrom", None)) if parent_data is not None else "")

    visit(parent)
    return tuple(sorted(names))


def _possible_evolution_ids(base_id: int, candidate_evolution_ids: Iterable[int]) -> tuple[int, ...]:
    base_name = _norm_name(card_name(base_id))
    if not base_name:
        return ()
    matches = [int(card_id) for card_id in candidate_evolution_ids if base_name in _ancestor_names(int(card_id))]
    return tuple(sorted(set(matches)))


def _is_evolved_pokemon(card_id: int | None) -> bool:
    data = card_data(card_id)
    return bool(data and data.cardType == 0 and not getattr(data, "basic", False))


def _classify_card(card_id: int | None) -> tuple[str, ...]:
    data = card_data(card_id)
    if data is None:
        return ()
    name = _norm_name(getattr(data, "name", ""))
    tags: set[str] = set()
    if is_trainer(card_id):
        if any(part in name for part in ("ball", "poffin", "incense", "aroma", "communication", "signal", "search")):
            tags.add("pokemon_search")
        if any(part in name for part in ("mega signal", "evolution", "rare candy", "salvatore")):
            tags.add("evolution_search")
        if any(part in name for part in ("research", "draw", "carmine", "lillie", "professor", "iono", "judge")):
            tags.add("draw")
        if any(part in name for part in ("boss", "catcher", "gust")):
            tags.add("gust")
        if any(part in name for part in ("switch", "cart", "rope")):
            tags.add("switch")
        if data.cardType == 4 or any(part in name for part in ("tool", "cape", "belt", "band")):
            tags.add("tool")
    if data.cardType in (5, 6):
        tags.add("energy")
    if is_pokemon(card_id):
        tags.add("pokemon")
        if _is_evolved_pokemon(card_id):
            tags.add("evolved_pokemon")
            tags.add("evolution")
    return tuple(sorted(tags))


@dataclass(frozen=True)
class ObservedOpponentAction:
    turn: int
    action_count: int
    action_type: str
    card_ids: tuple[int, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResponseOpportunity:
    category: str
    prior_probability: float
    use_probability: float


@dataclass(frozen=True)
class HandProbabilityEvidence:
    probability: float
    turn: int


@dataclass(frozen=True)
class OpponentIntentSnapshot:
    recent_pokemon_search: float = 0.0
    recent_evolution_search: float = 0.0
    recent_draw: float = 0.0
    recent_gust_seen: float = 0.0
    recent_switch_seen: float = 0.0
    likely_setting_up_attacker: float = 0.0
    evolution_window_open: float = 0.0
    setup_target_ids: tuple[int, ...] = ()
    must_answer_target_ids: tuple[int, ...] = ()
    played_evolution_ids: tuple[int, ...] = ()

    @property
    def has_setup_targets(self) -> bool:
        return bool(self.setup_target_ids)

    @property
    def has_must_answer_targets(self) -> bool:
        return bool(self.must_answer_target_ids)


class OpponentMemory:
    """Per-game memory of real opponent actions and inferred short-term intent."""

    def __init__(self, max_events: int = 96):
        self.max_events = int(max_events)
        self.events: deque[ObservedOpponentAction] = deque(maxlen=self.max_events)
        self.observed_card_ids: Counter[int] = Counter()
        self._observed_card_instances: set[tuple[int, int]] = set()
        self.known_hand_cards: dict[int, int] = {}
        self.played_evolution_ids: set[int] = set()
        self._last_turn: int | None = None
        self._last_result: int | None = None
        self._last_opponent_index: int | None = None
        self._seen_log_keys: set[tuple[Any, ...]] = set()
        self._last_discard: Counter[tuple[int, int | None]] = Counter()
        self._last_board: Counter[int] = Counter()
        self.pending_response_opportunities: tuple[ResponseOpportunity, ...] = ()
        self.hand_probability_evidence: dict[str, HandProbabilityEvidence] = {}
        self._response_opportunity_turn: int | None = None
        self._last_hand_reset_turn: int | None = None

    def reset(self) -> None:
        self.events.clear()
        self.observed_card_ids.clear()
        self._observed_card_instances.clear()
        self.known_hand_cards.clear()
        self.played_evolution_ids.clear()
        self._last_turn = None
        self._last_result = None
        self._last_opponent_index = None
        self._seen_log_keys.clear()
        self._last_discard.clear()
        self._last_board.clear()
        self.pending_response_opportunities = ()
        self.hand_probability_evidence.clear()
        self._response_opportunity_turn = None
        self._last_hand_reset_turn = None

    def update(self, obs: Any, state: AgentState) -> None:
        if self._should_reset(state):
            self.reset()
        opponent_index = int(state.opponent_index)
        self._ingest_logs(obs, state, opponent_index)
        self._ingest_visible_deltas(state)
        self._resolve_response_opportunities(state)
        self._last_turn = int(state.turn)
        self._last_result = int(state.result)
        self._last_opponent_index = opponent_index

    def arm_response_opportunities(
        self,
        turn: int,
        opportunities: Iterable[ResponseOpportunity],
    ) -> None:
        self.pending_response_opportunities = tuple(
            ResponseOpportunity(
                category=str(opportunity.category),
                prior_probability=max(0.0, min(1.0, float(opportunity.prior_probability))),
                use_probability=max(0.0, min(0.98, float(opportunity.use_probability))),
            )
            for opportunity in opportunities
            if opportunity.prior_probability > 0.0 and opportunity.use_probability > 0.0
        )
        self._response_opportunity_turn = int(turn) if self.pending_response_opportunities else None

    def adjusted_hand_probability(
        self,
        category: str,
        base_probability: float,
        *,
        draw_probability: float,
        current_turn: int,
    ) -> float:
        evidence = self.hand_probability_evidence.get(str(category))
        if evidence is None:
            return max(0.0, min(1.0, float(base_probability)))
        elapsed_opponent_turns = max(0, (int(current_turn) - int(evidence.turn)) // 2)
        recovered = 1.0 - ((1.0 - evidence.probability) * ((1.0 - max(0.0, min(1.0, draw_probability))) ** elapsed_opponent_turns))
        return max(0.0, min(1.0, recovered))

    def known_hand_counts(self) -> Counter[int]:
        return Counter(self.known_hand_cards.values())

    def snapshot(self, state: AgentState, main_attacker_ids: Iterable[int] = ()) -> OpponentIntentSnapshot:
        main_attackers = {int(card_id) for card_id in main_attacker_ids}
        current_pokemon = _pokemon_views(state)
        current_ids = {pokemon.id for pokemon in current_pokemon}
        played_evolutions = set(self.played_evolution_ids)
        played_evolutions.update(card_id for card_id in current_ids if _is_evolved_pokemon(card_id))
        played_main_attackers = played_evolutions & main_attackers if main_attackers else set()

        recent_pokemon_search = self._recent_tag_score("pokemon_search", state.turn)
        recent_evolution_search = self._recent_tag_score("evolution_search", state.turn)
        recent_draw = self._recent_tag_score("draw", state.turn)
        recent_gust = self._recent_tag_score("gust", state.turn)
        recent_switch = self._recent_tag_score("switch", state.turn)

        setup_targets: set[int] = set()
        must_answer: set[int] = set()
        for pokemon in current_pokemon:
            if pokemon.id in main_attackers:
                must_answer.add(pokemon.id)
                continue
            if pokemon.prize_value >= 2 or (pokemon.energy_count >= 2 and _is_evolved_pokemon(pokemon.id)):
                must_answer.add(pokemon.id)
            possible_evolutions = set(_possible_evolution_ids(pokemon.id, main_attackers)) if main_attackers else set()
            unresolved_evolutions = possible_evolutions - played_main_attackers
            if unresolved_evolutions and (recent_pokemon_search > 0.0 or recent_evolution_search > 0.0 or pokemon.energy_count > 0):
                setup_targets.add(pokemon.id)

        setup_signal = max(recent_pokemon_search * 0.8, recent_evolution_search, 0.35 if setup_targets else 0.0)
        evolution_window = max(recent_pokemon_search, recent_evolution_search) if setup_targets else 0.0
        if setup_targets and any(pokemon.energy_count > 0 and pokemon.id in setup_targets for pokemon in current_pokemon):
            setup_signal = max(setup_signal, 0.55)
            evolution_window = max(evolution_window, 0.35)

        return OpponentIntentSnapshot(
            recent_pokemon_search=recent_pokemon_search,
            recent_evolution_search=recent_evolution_search,
            recent_draw=recent_draw,
            recent_gust_seen=recent_gust,
            recent_switch_seen=recent_switch,
            likely_setting_up_attacker=max(0.0, min(1.0, setup_signal)),
            evolution_window_open=max(0.0, min(1.0, evolution_window)),
            setup_target_ids=tuple(sorted(setup_targets)),
            must_answer_target_ids=tuple(sorted(must_answer)),
            played_evolution_ids=tuple(sorted(played_evolutions)),
        )

    def _should_reset(self, state: AgentState) -> bool:
        if self._last_turn is None:
            return False
        if self._last_result is not None and self._last_result != -1 and state.result == -1:
            return True
        if int(state.turn) < self._last_turn:
            return True
        if self._last_opponent_index is not None and int(state.opponent_index) != self._last_opponent_index and int(state.turn) <= 1:
            return True
        return False

    def _resolve_response_opportunities(self, state: AgentState) -> None:
        armed_turn = getattr(self, "_response_opportunity_turn", None)
        if armed_turn is None or int(state.turn) <= int(armed_turn):
            return
        observed_tags = {
            tag
            for event in self.events
            if int(event.turn) > int(armed_turn)
            for tag in event.tags
        }
        hand_reset = self._last_hand_reset_turn is not None and int(self._last_hand_reset_turn) > int(armed_turn)
        for opportunity in self.pending_response_opportunities:
            category = str(opportunity.category)
            if category in observed_tags:
                self.hand_probability_evidence.pop(category, None)
                continue
            prior = max(0.0, min(1.0, opportunity.prior_probability))
            use = max(0.0, min(0.98, opportunity.use_probability))
            if hand_reset:
                use *= 0.35
            denominator = max(1e-9, 1.0 - (prior * use))
            posterior = (prior * (1.0 - use)) / denominator
            self.hand_probability_evidence[category] = HandProbabilityEvidence(
                probability=max(0.0, min(1.0, posterior)),
                turn=int(state.turn),
            )
        self.pending_response_opportunities = ()
        self._response_opportunity_turn = None

    def _ingest_logs(self, obs: Any, state: AgentState, opponent_index: int) -> None:
        for log in getattr(obs, "logs", []) or []:
            if getattr(log, "playerIndex", None) != opponent_index:
                continue
            key = self._log_key(state, log)
            if key in self._seen_log_keys:
                continue
            self._seen_log_keys.add(key)
            instances = _card_instances_from_log(log)
            card_ids = tuple(card_id for card_id, _serial in instances)
            tags = self._tags_for_cards(card_ids)
            action_type = _log_type_name(getattr(log, "type", None))
            self._track_known_hand_from_log(log, action_type, state.turn)
            if action_type == "EVOLVE":
                for card_id in card_ids:
                    if _is_evolved_pokemon(card_id):
                        self.played_evolution_ids.add(int(card_id))
            self._remember_event(
                state,
                action_type,
                card_ids,
                tags,
                serials=tuple(serial for _card_id, serial in instances),
            )

    def _ingest_visible_deltas(self, state: AgentState) -> None:
        current_discard = Counter(_visible_discard_key(card) for card in state.opp_discard)
        for key, count in (current_discard - self._last_discard).items():
            card_id = key[0]
            tags = self._tags_for_cards((card_id,))
            for _ in range(count):
                self._remember_event(state, "VISIBLE_DISCARD", (card_id,), tags, serials=(key[1],))
        self._last_discard = current_discard

        current_board = Counter(pokemon.id for pokemon in _pokemon_views(state))
        for card_id, count in (current_board - self._last_board).items():
            tags = self._tags_for_cards((card_id,))
            if _is_evolved_pokemon(card_id):
                self.played_evolution_ids.add(int(card_id))
            for _ in range(count):
                self._remember_event(state, "VISIBLE_BOARD", (card_id,), tags)
        for card_id in current_board:
            if _is_evolved_pokemon(card_id):
                self.played_evolution_ids.add(int(card_id))
        self._last_board = current_board

        visible_discard_serials = {
            int(card.serial)
            for card in state.opp_discard
            if card.serial is not None
        }
        for serial in visible_discard_serials:
            self.known_hand_cards.pop(serial, None)

    def _track_known_hand_from_log(self, log: Any, action_type: str, turn: int) -> None:
        card_id = getattr(log, "cardId", None)
        serial = getattr(log, "serial", None)
        if card_id is None or serial is None:
            return
        from_area = _area_type_name(getattr(log, "fromArea", None))
        to_area = _area_type_name(getattr(log, "toArea", None))
        serial_id = int(serial)
        if from_area == "HAND" and to_area == "DECK":
            self.known_hand_cards.clear()
            self.hand_probability_evidence.clear()
            self._last_hand_reset_turn = int(turn)
            return
        if (from_area == "HAND" and to_area != "HAND") or action_type in {"PLAY", "ATTACH", "EVOLVE"}:
            self.known_hand_cards.pop(serial_id, None)
            return
        explicitly_revealed_move = action_type == "MOVE_CARD" and from_area in {"DECK", "DISCARD", "LOOKING"}
        if to_area == "HAND" and explicitly_revealed_move:
            self.known_hand_cards[serial_id] = int(card_id)

    def _remember_event(
        self,
        state: AgentState,
        action_type: str,
        card_ids: Iterable[int],
        tags: Iterable[str],
        serials: Iterable[int | None] = (),
    ) -> None:
        ids = tuple(int(card_id) for card_id in card_ids if card_id is not None)
        if not ids and not tuple(tags):
            return
        serial_tuple = tuple(serials)
        for index, card_id in enumerate(ids):
            serial = serial_tuple[index] if index < len(serial_tuple) else None
            if serial is not None:
                instance = (int(card_id), int(serial))
                if instance in self._observed_card_instances:
                    continue
                self._observed_card_instances.add(instance)
            self.observed_card_ids[int(card_id)] += 1
        tag_tuple = tuple(sorted(set(tags)))
        self.events.append(
            ObservedOpponentAction(
                turn=int(state.turn),
                action_count=int(state.turn_action_count),
                action_type=str(action_type),
                card_ids=ids,
                tags=tag_tuple,
            )
        )

    def _tags_for_cards(self, card_ids: Iterable[int]) -> tuple[str, ...]:
        tags: set[str] = set()
        for card_id in card_ids:
            tags.update(_classify_card(int(card_id)))
        return tuple(sorted(tags))

    def _recent_tag_score(self, tag: str, turn: int) -> float:
        score = 0.0
        for event in self.events:
            if tag in event.tags:
                score = max(score, _decayed_score(turn, event.turn))
        return max(0.0, min(1.0, score))

    def _log_key(self, state: AgentState, log: Any) -> tuple[Any, ...]:
        return (
            int(state.turn),
            int(state.turn_action_count),
            getattr(log, "type", None),
            getattr(log, "playerIndex", None),
            getattr(log, "serial", None),
            getattr(log, "serialBefore", None),
            getattr(log, "serialAfter", None),
            getattr(log, "serialTarget", None),
            getattr(log, "cardId", None),
            getattr(log, "cardIdBefore", None),
            getattr(log, "cardIdAfter", None),
            getattr(log, "cardIdTarget", None),
            getattr(log, "attackId", None),
            getattr(log, "fromArea", None),
            getattr(log, "toArea", None),
            getattr(log, "value", None),
        )


EMPTY_OPPONENT_INTENT = OpponentIntentSnapshot()
