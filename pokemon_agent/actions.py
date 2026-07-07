from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .card_db import attack_name, card_name
from .cg_loader import ensure_cg_path

ensure_cg_path()

from cg.api import AreaType, OptionType, Observation, SelectContext, SelectType  # noqa: E402


def enum_name(value: Any) -> str:
    if value is None:
        return "NONE"
    return getattr(value, "name", str(value))


def enum_value_name(enum_cls: Any, value: Any) -> str:
    if value is None:
        return "NONE"
    try:
        return enum_cls(value).name
    except (ValueError, TypeError):
        return enum_name(value)


def safe_get_card(obs: Observation, area: AreaType | None, index: int | None, player_index: int | None) -> Any | None:
    if obs.current is None or area is None or index is None:
        return None
    try:
        if area == AreaType.DECK:
            if obs.select and obs.select.deck:
                return obs.select.deck[index]
            return None
        if area == AreaType.LOOKING:
            if obs.current.looking:
                return obs.current.looking[index]
            return None
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if player_index is None:
            player_index = obs.current.yourIndex
        player = obs.current.players[player_index]
        if area == AreaType.HAND:
            return player.hand[index] if player.hand is not None else None
        if area == AreaType.DISCARD:
            return player.discard[index]
        if area == AreaType.ACTIVE:
            return player.active[index]
        if area == AreaType.BENCH:
            return player.bench[index]
        if area == AreaType.PRIZE:
            return player.prize[index]
    except (IndexError, TypeError):
        return None
    return None


@dataclass(frozen=True)
class ActionView:
    index: int
    option: Any
    select_type: str
    context: str
    option_type: str
    source_player_index: int | None
    source_card_id: int | None
    source_name: str | None
    target_player_index: int | None
    target_card_id: int | None
    target_name: str | None
    attack_id: int | None
    attack_name: str | None
    label: str

    @property
    def is_attack(self) -> bool:
        return self.option_type == "ATTACK"

    @property
    def is_end(self) -> bool:
        return self.option_type == "END"

    @property
    def is_play(self) -> bool:
        return self.option_type == "PLAY"

    @property
    def is_attach(self) -> bool:
        return self.option_type == "ATTACH"

    @property
    def is_evolve(self) -> bool:
        return self.option_type == "EVOLVE"

    @property
    def is_ability(self) -> bool:
        return self.option_type == "ABILITY"


def _card_id(card: Any | None) -> int | None:
    return getattr(card, "id", None)


def _card_name(card: Any | None) -> str | None:
    card_id = _card_id(card)
    return card_name(card_id) if card_id is not None else None


def _area_label(area: AreaType | None, index: int | None) -> str:
    if area is None:
        return ""
    suffix = "" if index is None else f"[{index}]"
    return f"{enum_value_name(AreaType, area)}{suffix}"


def build_action_view(obs: Observation, option: Any, index: int) -> ActionView:
    select = obs.select
    select_type = enum_value_name(SelectType, select.type) if select else enum_name(SelectType.MAIN)
    context = enum_value_name(SelectContext, select.context) if select else enum_name(SelectContext.MAIN)
    option_type = enum_value_name(OptionType, option.type)
    player_index = getattr(option, "playerIndex", None)
    if player_index is None and obs.current is not None:
        player_index = obs.current.yourIndex

    source = None
    target = None
    attack_id = getattr(option, "attackId", None)
    attack_label = attack_name(attack_id) if attack_id is not None else None

    if option.type in {
        OptionType.CARD,
        OptionType.TOOL_CARD,
        OptionType.ENERGY_CARD,
        OptionType.ENERGY,
        OptionType.PLAY,
        OptionType.ATTACH,
        OptionType.EVOLVE,
        OptionType.ABILITY,
        OptionType.DISCARD,
    }:
        source = safe_get_card(obs, getattr(option, "area", None) or AreaType.HAND, getattr(option, "index", None), player_index)

    if option.type in {OptionType.ATTACH, OptionType.EVOLVE}:
        target = safe_get_card(obs, getattr(option, "inPlayArea", None), getattr(option, "inPlayIndex", None), obs.current.yourIndex if obs.current else player_index)
    elif option.type == OptionType.CARD:
        target = source

    source_name = _card_name(source)
    target_name = _card_name(target)
    source_player_index = getattr(source, "playerIndex", player_index) if source is not None else player_index
    target_player_index = getattr(target, "playerIndex", None) if target is not None else None

    if option.type == OptionType.NUMBER:
        label = f"choose number {option.number}"
    elif option.type == OptionType.YES:
        label = "choose yes"
    elif option.type == OptionType.NO:
        label = "choose no"
    elif option.type == OptionType.PLAY:
        label = f"play {source_name or 'card'}"
    elif option.type == OptionType.ATTACH:
        label = f"attach {source_name or 'card'} to {target_name or _area_label(option.inPlayArea, option.inPlayIndex)}"
    elif option.type == OptionType.EVOLVE:
        label = f"evolve {target_name or 'pokemon'} into {source_name or 'evolution'}"
    elif option.type == OptionType.ABILITY:
        label = f"use ability on {source_name or 'pokemon'}"
    elif option.type == OptionType.RETREAT:
        label = "retreat active"
    elif option.type == OptionType.ATTACK:
        label = f"attack with {attack_label or 'attack'}"
    elif option.type == OptionType.END:
        label = "end turn"
    elif option.type in {OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD, OptionType.ENERGY}:
        label = f"select {source_name or 'card'} from {_area_label(getattr(option, 'area', None), getattr(option, 'index', None))}"
    elif option.type == OptionType.SKILL:
        label = f"select skill {option.cardId or ''}".strip()
    elif option.type == OptionType.SPECIAL_CONDITION:
        label = f"select special condition {option.specialConditionType}"
    else:
        label = f"{option_type} option {index}"

    return ActionView(
        index=index,
        option=option,
        select_type=select_type,
        context=context,
        option_type=option_type,
        source_player_index=int(source_player_index) if source_player_index is not None else None,
        source_card_id=_card_id(source),
        source_name=source_name,
        target_player_index=int(target_player_index) if target_player_index is not None else None,
        target_card_id=_card_id(target),
        target_name=target_name,
        attack_id=attack_id,
        attack_name=attack_label,
        label=label,
    )


def legal_actions(obs: Observation) -> list[ActionView]:
    if obs.select is None:
        return []
    return [build_action_view(obs, option, index) for index, option in enumerate(obs.select.option)]


def sanitize_selection(obs: Observation, selected: list[int]) -> tuple[list[int], bool]:
    if obs.select is None:
        return [], selected != []
    option_count = len(obs.select.option)
    min_count = obs.select.minCount
    max_count = obs.select.maxCount
    clean: list[int] = []
    invalid = False
    for value in selected:
        if not isinstance(value, int) or value < 0 or value >= option_count or value in clean:
            invalid = True
            continue
        clean.append(value)
        if len(clean) >= max_count:
            break
    if len(clean) < min_count:
        invalid = True
        for index in range(option_count):
            if index not in clean:
                clean.append(index)
                if len(clean) >= min_count:
                    break
    if len(clean) > max_count:
        invalid = True
        clean = clean[:max_count]
    return clean, invalid
