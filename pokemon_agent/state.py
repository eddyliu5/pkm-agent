from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actions import ActionView, legal_actions
from .card_db import card_data, card_name, prize_value
from .cg_loader import ensure_cg_path

ensure_cg_path()

from cg.api import Observation  # noqa: E402


@dataclass(frozen=True)
class CardView:
    id: int
    name: str
    serial: int | None = None


@dataclass(frozen=True)
class PokemonView:
    id: int
    name: str
    hp: int
    max_hp: int
    energy_count: int
    energy_ids: tuple[int, ...]
    tool_ids: tuple[int, ...]
    prize_value: int
    is_ex: bool
    is_mega_ex: bool
    is_basic: bool
    is_stage1: bool
    is_stage2: bool
    appears_this_turn: bool

    @property
    def damage(self) -> int:
        return max(0, self.max_hp - self.hp)


@dataclass(frozen=True)
class AgentState:
    turn: int
    turn_action_count: int
    player_index: int
    opponent_index: int
    first_player: int
    result: int
    supporter_played: bool
    stadium_played: bool
    energy_attached: bool
    retreated: bool
    stadium_id: int | None
    stadium_name: str | None
    my_active: PokemonView | None
    my_bench: tuple[PokemonView, ...]
    my_hand: tuple[CardView, ...]
    my_discard: tuple[CardView, ...]
    my_prizes_remaining: int
    my_deck_count: int
    my_status: tuple[str, ...]
    opp_active: PokemonView | None
    opp_bench: tuple[PokemonView, ...]
    opp_hand_count: int
    opp_discard: tuple[CardView, ...]
    opp_prizes_remaining: int
    opp_deck_count: int
    opp_status: tuple[str, ...]
    my_bench_max: int
    opp_bench_max: int
    select_type: str
    select_context: str
    select_min_count: int
    select_max_count: int
    select_option_count: int
    legal_actions: tuple[ActionView, ...] = field(default_factory=tuple)

    @property
    def prize_diff(self) -> int:
        return self.opp_prizes_remaining - self.my_prizes_remaining

    @property
    def my_energy_in_play(self) -> int:
        mons = ([self.my_active] if self.my_active else []) + list(self.my_bench)
        return sum(mon.energy_count for mon in mons)

    @property
    def opp_energy_in_play(self) -> int:
        mons = ([self.opp_active] if self.opp_active else []) + list(self.opp_bench)
        return sum(mon.energy_count for mon in mons)


def _card_view(card: Any | None) -> CardView | None:
    if card is None:
        return None
    card_id = getattr(card, "id", None)
    if card_id is None:
        return None
    return CardView(id=int(card_id), name=card_name(card_id), serial=getattr(card, "serial", None))


def _pokemon_view(pokemon: Any | None) -> PokemonView | None:
    if pokemon is None:
        return None
    card_id = int(pokemon.id)
    data = card_data(card_id)
    energy_ids = tuple(int(getattr(card, "id", 0)) for card in getattr(pokemon, "energyCards", []) or [])
    tool_ids = tuple(int(getattr(card, "id", 0)) for card in getattr(pokemon, "tools", []) or [])
    return PokemonView(
        id=card_id,
        name=card_name(card_id),
        hp=int(getattr(pokemon, "hp", 0) or 0),
        max_hp=int(getattr(pokemon, "maxHp", 0) or 0),
        energy_count=len(getattr(pokemon, "energies", []) or []),
        energy_ids=energy_ids,
        tool_ids=tool_ids,
        prize_value=prize_value(card_id, list(energy_ids), list(tool_ids)),
        is_ex=bool(data and data.ex),
        is_mega_ex=bool(data and data.megaEx),
        is_basic=bool(data and data.basic),
        is_stage1=bool(data and data.stage1),
        is_stage2=bool(data and data.stage2),
        appears_this_turn=bool(getattr(pokemon, "appearThisTurn", False)),
    )


def _active(player: Any) -> PokemonView | None:
    if not player.active:
        return None
    return _pokemon_view(player.active[0])


def _cards(cards: list[Any] | None) -> tuple[CardView, ...]:
    if not cards:
        return ()
    return tuple(card for card in (_card_view(c) for c in cards) if card is not None)


def _pokemon_list(cards: list[Any] | None) -> tuple[PokemonView, ...]:
    if not cards:
        return ()
    return tuple(card for card in (_pokemon_view(c) for c in cards) if card is not None)


def _statuses(player: Any) -> tuple[str, ...]:
    names = []
    for attr in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
        if getattr(player, attr, False):
            names.append(attr)
    return tuple(names)


def _enum_label(value: Any) -> str:
    if value is None:
        return "NONE"
    return getattr(value, "name", str(value))


def build_agent_state(obs: Observation, perspective: int | None = None, actions: list[ActionView] | None = None) -> AgentState:
    if obs.current is None:
        raise ValueError("Cannot build AgentState without a current board state")
    state = obs.current
    player_index = state.yourIndex if perspective is None else perspective
    opponent_index = 1 - player_index
    my_player = state.players[player_index]
    opp_player = state.players[opponent_index]
    stadium = state.stadium[0] if state.stadium else None
    stadium_id = getattr(stadium, "id", None)
    select = obs.select
    return AgentState(
        turn=state.turn,
        turn_action_count=state.turnActionCount,
        player_index=player_index,
        opponent_index=opponent_index,
        first_player=state.firstPlayer,
        result=state.result,
        supporter_played=state.supporterPlayed,
        stadium_played=state.stadiumPlayed,
        energy_attached=state.energyAttached,
        retreated=state.retreated,
        stadium_id=stadium_id,
        stadium_name=card_name(stadium_id) if stadium_id is not None else None,
        my_active=_active(my_player),
        my_bench=_pokemon_list(my_player.bench),
        my_hand=_cards(my_player.hand),
        my_discard=_cards(my_player.discard),
        my_prizes_remaining=len(my_player.prize),
        my_deck_count=my_player.deckCount,
        my_status=_statuses(my_player),
        opp_active=_active(opp_player),
        opp_bench=_pokemon_list(opp_player.bench),
        opp_hand_count=opp_player.handCount,
        opp_discard=_cards(opp_player.discard),
        opp_prizes_remaining=len(opp_player.prize),
        opp_deck_count=opp_player.deckCount,
        opp_status=_statuses(opp_player),
        my_bench_max=int(getattr(my_player, "benchMax", 5) or 5),
        opp_bench_max=int(getattr(opp_player, "benchMax", 5) or 5),
        select_type=_enum_label(getattr(select, "type", None)),
        select_context=_enum_label(getattr(select, "context", None)),
        select_min_count=int(getattr(select, "minCount", 0) or 0),
        select_max_count=int(getattr(select, "maxCount", 0) or 0),
        select_option_count=len(getattr(select, "option", []) or []),
        legal_actions=tuple(actions if actions is not None else legal_actions(obs)),
    )
