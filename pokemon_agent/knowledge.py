from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .card_db import attack_data, card_data, prize_value


ROLE_KEYS = (
    "main_attackers",
    "main_line",
    "backup_attackers",
    "setup_basics",
    "key_evolutions",
    "search_cards",
    "draw_cards",
    "gust_cards",
    "energy_cards",
    "damage_boost_cards",
    "tool_cards",
)


@dataclass(frozen=True)
class CardKnowledge:
    card_id: int
    vector: tuple[float, ...]
    role_tags: tuple[str, ...]


def _role_sets(config: dict[str, Any]) -> dict[str, set[int]]:
    return {key: {int(value) for value in config.get(key, [])} for key in ROLE_KEYS}


def _tokens_for_card(card_id: int) -> list[str]:
    data = card_data(card_id)
    if data is None:
        return []
    text_parts = [data.name or "", data.evolvesFrom or ""]
    for skill in getattr(data, "skills", []) or []:
        text_parts.append(getattr(skill, "name", "") or "")
        text_parts.append(getattr(skill, "text", "") or "")
    for attack_id in getattr(data, "attacks", []) or []:
        attack = attack_data(attack_id)
        if attack:
            text_parts.append(getattr(attack, "name", "") or "")
            text_parts.append(getattr(attack, "text", "") or "")
    return re.findall(r"[a-z0-9]+", " ".join(text_parts).lower())


def _hash_index(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    sign = 1.0 if value & 1 else -1.0
    return value % dim, sign


@lru_cache(maxsize=4096)
def _text_hash_embedding(card_id: int, dim: int) -> tuple[float, ...]:
    values = [0.0] * dim
    tokens = _tokens_for_card(card_id)
    if not tokens:
        return tuple(values)
    for token in tokens:
        index, sign = _hash_index(token, dim)
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return tuple(value / norm for value in values)


def _structured_features(card_id: int) -> tuple[float, ...]:
    data = card_data(card_id)
    if data is None:
        return (0.0,) * 18
    attacks = [attack_data(attack_id) for attack_id in getattr(data, "attacks", []) or []]
    attacks = [attack for attack in attacks if attack is not None]
    max_damage = max((float(getattr(attack, "damage", 0) or 0) for attack in attacks), default=0.0)
    min_energy_cost = min((len(getattr(attack, "energies", []) or []) for attack in attacks), default=0)
    return (
        1.0 if data.cardType == 0 else 0.0,
        1.0 if data.cardType in (1, 2, 3, 4) else 0.0,
        1.0 if data.cardType in (5, 6) else 0.0,
        1.0 if getattr(data, "basic", False) else 0.0,
        1.0 if getattr(data, "stage1", False) else 0.0,
        1.0 if getattr(data, "stage2", False) else 0.0,
        1.0 if getattr(data, "ex", False) else 0.0,
        1.0 if getattr(data, "megaEx", False) else 0.0,
        1.0 if getattr(data, "tera", False) else 0.0,
        min(1.0, float(getattr(data, "hp", 0) or 0) / 350.0),
        min(1.0, float(getattr(data, "retreatCost", 0) or 0) / 5.0),
        min(1.0, max_damage / 350.0),
        min(1.0, float(min_energy_cost) / 5.0),
        min(1.0, float(len(attacks)) / 3.0),
        min(1.0, float(len(getattr(data, "skills", []) or [])) / 3.0),
        min(1.0, float(prize_value(card_id)) / 3.0),
        min(1.0, float(getattr(data, "energyType", 0) or 0) / 11.0),
        1.0 if getattr(data, "aceSpec", False) else 0.0,
    )


class CardKnowledgeStore:
    """Dependency-free card representation for runtime inference."""

    def __init__(self, config: dict[str, Any], text_dim: int = 32):
        self.config = config
        self.text_dim = int(text_dim)
        self.roles = _role_sets(config)

    def role_tags(self, card_id: int | None) -> tuple[str, ...]:
        if card_id is None:
            return ()
        cid = int(card_id)
        return tuple(key for key, values in self.roles.items() if cid in values)

    def vector(self, card_id: int | None) -> tuple[float, ...]:
        if card_id is None:
            return (0.0,) * (self.text_dim + 18 + len(ROLE_KEYS))
        cid = int(card_id)
        role_values = tuple(1.0 if cid in self.roles[key] else 0.0 for key in ROLE_KEYS)
        return _text_hash_embedding(cid, self.text_dim) + _structured_features(cid) + role_values

    def get(self, card_id: int | None) -> CardKnowledge:
        cid = int(card_id or 0)
        return CardKnowledge(card_id=cid, vector=self.vector(card_id), role_tags=self.role_tags(card_id))

