from __future__ import annotations

from functools import lru_cache
from typing import Any

from .cg_loader import ensure_cg_path

ensure_cg_path()

from cg.api import all_attack, all_card_data  # noqa: E402


@lru_cache(maxsize=1)
def cards_by_id() -> dict[int, Any]:
    return {card.cardId: card for card in all_card_data()}


@lru_cache(maxsize=1)
def attacks_by_id() -> dict[int, Any]:
    return {attack.attackId: attack for attack in all_attack()}


def card_data(card_id: int | None) -> Any | None:
    if card_id is None:
        return None
    return cards_by_id().get(int(card_id))


def attack_data(attack_id: int | None) -> Any | None:
    if attack_id is None:
        return None
    return attacks_by_id().get(int(attack_id))


def card_name(card_id: int | None) -> str:
    data = card_data(card_id)
    if data is None:
        return "Unknown"
    return data.name


def attack_name(attack_id: int | None) -> str:
    data = attack_data(attack_id)
    if data is None:
        return f"Attack {attack_id}"
    return data.name


def prize_value(card_id: int | None, energy_card_ids: list[int] | None = None, tool_ids: list[int] | None = None) -> int:
    data = card_data(card_id)
    if data is None:
        return 1
    count = 3 if data.megaEx else 2 if data.ex else 1
    for energy_id in energy_card_ids or []:
        if energy_id == 12:
            count -= 1
    if tool_ids and data.name and "Lillie" in data.name:
        for tool_id in tool_ids:
            if tool_id == 1172:
                count -= 1
    return max(0, count)


def is_pokemon(card_id: int | None) -> bool:
    data = card_data(card_id)
    return bool(data and data.cardType == 0)


def is_energy(card_id: int | None) -> bool:
    data = card_data(card_id)
    return bool(data and data.cardType in (5, 6))


def is_trainer(card_id: int | None) -> bool:
    data = card_data(card_id)
    return bool(data and data.cardType in (1, 2, 3, 4))

