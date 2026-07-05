from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .card_db import attack_data, card_data, card_name, prize_value
from .cg_loader import PROJECT_ROOT


DEFAULT_DECK_DB_PATH = PROJECT_ROOT / "data" / "deck_db" / "decks.json"


def load_deck_db(path: str | Path | None = None) -> dict[str, Any]:
    db_path = Path(path) if path is not None else DEFAULT_DECK_DB_PATH
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    with db_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def archetypes(db: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = db or load_deck_db()
    return dict(payload.get("archetypes", {}))


def iter_decks(db: dict[str, Any] | None = None, *, archetype: str | None = None) -> Iterable[dict[str, Any]]:
    payload = db or load_deck_db()
    for deck in payload.get("decks", []):
        if archetype is not None and deck.get("archetype") != archetype:
            continue
        yield deck


def get_deck(deck_id: str, db: dict[str, Any] | None = None) -> dict[str, Any]:
    for deck in iter_decks(db):
        if deck.get("id") == deck_id:
            return deck
    raise KeyError(f"Unknown deck id: {deck_id}")


def deck_counts(deck: dict[str, Any]) -> Counter[int]:
    return Counter({int(row["id"]): int(row["count"]) for row in deck.get("counts", [])})


def expand_deck(deck: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for card_id, count in sorted(deck_counts(deck).items()):
        ids.extend([card_id] * count)
    return ids


@dataclass(frozen=True)
class DeckGameplanProfile:
    deck_id: str
    archetype: str
    plan_type: str
    main_attacker_ids: tuple[int, ...]
    average_first_attack_turn: float
    average_game_end_turn: float
    average_attack_cost: float
    max_attack_damage: float
    pokemon_count: int
    energy_count: int
    trainer_count: int


def _attack_costs_and_damage(card_id: int) -> list[tuple[int, int]]:
    data = card_data(card_id)
    values: list[tuple[int, int]] = []
    for attack_id in getattr(data, "attacks", []) or []:
        attack = attack_data(attack_id)
        if attack is None:
            continue
        values.append((len(getattr(attack, "energies", []) or []), int(getattr(attack, "damage", 0) or 0)))
    return values


def _pokemon_stage_delay(card_id: int) -> float:
    data = card_data(card_id)
    if data is None:
        return 0.0
    if getattr(data, "stage2", False):
        return 2.0
    if getattr(data, "stage1", False):
        return 1.0
    return 0.0


def _attacker_score(card_id: int, count: int) -> float:
    data = card_data(card_id)
    if data is None or data.cardType != 0:
        return -1.0
    attacks = _attack_costs_and_damage(card_id)
    if not attacks:
        return -1.0
    max_damage = max(damage for _cost, damage in attacks)
    min_cost = min(cost for cost, damage in attacks if damage == max_damage)
    prize_bonus = 55.0 * prize_value(card_id)
    stage_penalty = 35.0 * _pokemon_stage_delay(card_id)
    return max_damage + prize_bonus + (12.0 * count) - (22.0 * min_cost) - stage_penalty


def _estimate_plan_type(tags: Iterable[str], main_attackers: list[int], avg_first_attack: float, avg_game_end: float) -> str:
    tag_set = {str(tag).lower() for tag in tags}
    if "control" in tag_set or "deck_out" in tag_set or avg_game_end >= 11.0:
        return "control"
    if any(card_data(card_id) and getattr(card_data(card_id), "megaEx", False) for card_id in main_attackers):
        return "setup_big_attacker"
    if avg_first_attack <= 2.0 and avg_game_end <= 8.0:
        return "pressure"
    return "hybrid"


def deck_gameplan_profile(deck: dict[str, Any], db: dict[str, Any] | None = None) -> DeckGameplanProfile:
    counts = deck_counts(deck)
    composition = deck.get("composition", {})
    pokemon_count = int(composition.get("pokemon", 0) or 0)
    energy_count = int(composition.get("energy", 0) or 0)
    trainer_count = int(composition.get("trainer", 0) or 0)
    if not composition:
        for card_id, count in counts.items():
            data = card_data(card_id)
            if data is None:
                continue
            if data.cardType == 0:
                pokemon_count += count
            elif data.cardType in (5, 6):
                energy_count += count
            else:
                trainer_count += count

    attacker_scores = [
        (card_id, _attacker_score(card_id, count))
        for card_id, count in counts.items()
        if _attacker_score(card_id, count) > 0
    ]
    attacker_scores.sort(key=lambda item: item[1], reverse=True)
    main_attackers = [card_id for card_id, _score in attacker_scores[:3]]
    attack_profiles: list[tuple[float, float]] = []
    for card_id in main_attackers or [card_id for card_id, count in counts.items() if _attacker_score(card_id, count) > 0]:
        attacks = _attack_costs_and_damage(card_id)
        if not attacks:
            continue
        best_damage = max(damage for _cost, damage in attacks)
        best_cost = min(cost for cost, damage in attacks if damage == best_damage)
        attack_profiles.append((_pokemon_stage_delay(card_id) + max(1.0, float(best_cost)), float(best_damage)))

    if attack_profiles:
        avg_first_attack = sum(cost for cost, _damage in attack_profiles) / len(attack_profiles)
        avg_attack_cost = sum(cost for cost, _damage in attack_profiles) / len(attack_profiles)
        max_damage = max(damage for _cost, damage in attack_profiles)
    else:
        avg_first_attack = 3.5
        avg_attack_cost = 3.0
        max_damage = 0.0

    pressure = max_damage / max(1.0, avg_first_attack * 100.0)
    avg_game_end = max(5.5, min(13.5, 10.5 - (2.7 * pressure) + (0.22 * avg_first_attack)))
    archetype_meta = archetypes(db).get(deck.get("archetype", ""), {}) if db is not None else {}
    tags = list(archetype_meta.get("tags", []))
    plan_type = _estimate_plan_type(tags, main_attackers, avg_first_attack, avg_game_end)
    if plan_type == "control":
        avg_game_end = max(avg_game_end, 11.0)
    elif plan_type == "pressure":
        avg_game_end = min(avg_game_end, 8.0)

    return DeckGameplanProfile(
        deck_id=str(deck["id"]),
        archetype=str(deck["archetype"]),
        plan_type=plan_type,
        main_attacker_ids=tuple(main_attackers),
        average_first_attack_turn=round(avg_first_attack, 2),
        average_game_end_turn=round(avg_game_end, 2),
        average_attack_cost=round(avg_attack_cost, 2),
        max_attack_damage=round(max_damage, 1),
        pokemon_count=pokemon_count,
        energy_count=energy_count,
        trainer_count=trainer_count,
    )


def archetype_gameplan_profiles(db: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    payload = db or load_deck_db()
    by_archetype: dict[str, list[DeckGameplanProfile]] = defaultdict(list)
    for deck in iter_decks(payload):
        by_archetype[deck["archetype"]].append(deck_gameplan_profile(deck, payload))

    profiles: dict[str, dict[str, Any]] = {}
    for archetype, rows in by_archetype.items():
        main_attackers = Counter(card_id for row in rows for card_id in row.main_attacker_ids)
        plan_types = Counter(row.plan_type for row in rows)
        profiles[archetype] = {
            "archetype": archetype,
            "deck_count": len(rows),
            "plan_type": plan_types.most_common(1)[0][0],
            "main_attacker_ids": [card_id for card_id, _count in main_attackers.most_common(5)],
            "average_first_attack_turn": round(sum(row.average_first_attack_turn for row in rows) / len(rows), 2),
            "average_game_end_turn": round(sum(row.average_game_end_turn for row in rows) / len(rows), 2),
            "average_attack_cost": round(sum(row.average_attack_cost for row in rows) / len(rows), 2),
            "max_attack_damage": round(max(row.max_attack_damage for row in rows), 1),
            "average_pokemon_count": round(sum(row.pokemon_count for row in rows) / len(rows), 2),
            "average_energy_count": round(sum(row.energy_count for row in rows) / len(rows), 2),
            "average_trainer_count": round(sum(row.trainer_count for row in rows) / len(rows), 2),
        }
    return profiles


def validate_card_counts(counts: Counter[int]) -> list[str]:
    problems: list[str] = []
    total = sum(counts.values())
    if total != 60:
        problems.append(f"expected 60 cards, got {total}")

    name_counts: Counter[str] = Counter()
    ace_spec_count = 0
    basic_pokemon_count = 0
    for card_id, count in counts.items():
        data = card_data(card_id)
        if data is None:
            problems.append(f"unknown card id: {card_id}")
            continue
        name_counts[card_name(card_id)] += count
        if getattr(data, "aceSpec", False):
            ace_spec_count += count
        if data.cardType == 0 and getattr(data, "basic", False):
            basic_pokemon_count += count

    for name, count in name_counts.items():
        card_ids = [card_id for card_id in counts if card_name(card_id) == name]
        is_basic_energy = any(card_data(card_id) and card_data(card_id).cardType == 5 for card_id in card_ids)
        if count > 4 and not is_basic_energy:
            problems.append(f"more than 4 copies of non-basic-energy card {name}: {count}")

    if ace_spec_count > 1:
        problems.append(f"more than 1 ACE SPEC card: {ace_spec_count}")
    if basic_pokemon_count <= 0:
        problems.append("no Basic Pokemon")
    return problems


def validate_deck(deck: dict[str, Any]) -> list[str]:
    return validate_card_counts(deck_counts(deck))


def score_deck_against_observed(deck: dict[str, Any], observed_card_ids: Iterable[int]) -> dict[str, Any]:
    observed = Counter(int(card_id) for card_id in observed_card_ids)
    total_observed = sum(observed.values())
    counts = deck_counts(deck)
    matched = sum(min(count, counts.get(card_id, 0)) for card_id, count in observed.items())
    impossible = {
        card_id: count - counts.get(card_id, 0)
        for card_id, count in observed.items()
        if count > counts.get(card_id, 0)
    }
    impossible_count = sum(impossible.values())
    score = 0.0 if total_observed == 0 else (matched - impossible_count) / total_observed
    return {
        "deck_id": deck["id"],
        "name": deck["name"],
        "archetype": deck["archetype"],
        "archetype_name": deck.get("archetype_name", deck["archetype"]),
        "score": score,
        "matched_cards": matched,
        "observed_cards": total_observed,
        "impossible_cards": {
            str(card_id): {"count": count, "name": card_name(card_id)}
            for card_id, count in sorted(impossible.items())
        },
    }


def rank_decks_by_observed_cards(
    observed_card_ids: Iterable[int],
    db: dict[str, Any] | None = None,
    *,
    archetype: str | None = None,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    ranked = [score_deck_against_observed(deck, observed_card_ids) for deck in iter_decks(db, archetype=archetype)]
    ranked.sort(key=lambda row: (row["score"], row["matched_cards"], -len(row["impossible_cards"])), reverse=True)
    return ranked[:top_n] if top_n is not None else ranked


def _summary(db: dict[str, Any]) -> dict[str, Any]:
    by_archetype = Counter(deck["archetype"] for deck in db.get("decks", []))
    profiles = archetype_gameplan_profiles(db)
    return {
        "decks": len(db.get("decks", [])),
        "archetypes": len(by_archetype),
        "by_archetype": dict(sorted(by_archetype.items())),
        "gameplans": dict(sorted(Counter(row["plan_type"] for row in profiles.values()).items())),
        "rejected_sources": len(db.get("rejected_sources", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the local deck/archetype database.")
    parser.add_argument("--db", default=None)
    parser.add_argument("--archetype", default=None)
    parser.add_argument("--observed", nargs="*", type=int, default=None, help="Observed opponent card ids to rank decks.")
    parser.add_argument("--profiles", action="store_true", help="Print derived archetype gameplan profiles.")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    db = load_deck_db(args.db)
    if args.profiles:
        print(json.dumps(archetype_gameplan_profiles(db), indent=2, sort_keys=True))
    elif args.observed is not None:
        print(json.dumps(rank_decks_by_observed_cards(args.observed, db, archetype=args.archetype, top_n=args.top), indent=2))
    else:
        print(json.dumps(_summary(db), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
