# Local Kaggle Proxy

Kaggle score is the source of truth, but it is expensive and delayed. This proxy is the default local indicator for whether a candidate is worth submitting.

## Inputs

Run these three suites with the same agent, config, and game count:

1. Random mirror: `--opponent random`
2. Random deck-db rotation: `--opponent random --opponent-decks deck_db`
3. Heuristic mirror: `--opponent heuristic`

From each evaluator summary, collect:

- `win_rate`
- `invalid_actions`
- `average_prize_diff`
- `average_steps`

## Proxy Formula

Use this score on a 0 to 100 scale:

```text
local_kaggle_proxy =
  100 * (
    0.20 * random_mirror_win_rate
  + 0.50 * deck_db_rotation_win_rate
  + 0.30 * heuristic_mirror_win_rate
  )
  + prize_bonus
  - invalid_action_penalty
  - slow_game_penalty
```

Where:

```text
prize_bonus = clamp(average of the three average_prize_diff values, -3.0, 3.0) * 2.0
invalid_action_penalty = 25.0 if total invalid_actions > 0 else 0.0
slow_game_penalty = 5.0 if average of the three average_steps values > 450 else 0.0
```

The deck-db rotation weight is intentionally highest because varied opponent decks are a better local proxy for Kaggle than mirror-only games.

## Manual Calculation Example

If a candidate produces:

```text
random mirror win_rate = 0.92
deck-db rotation win_rate = 0.68
heuristic mirror win_rate = 0.56
average prize diffs = 2.1, 0.8, 0.2
invalid_actions = 0
average steps = 180, 220, 240
```

Then:

```text
weighted win score = 100 * ((0.20 * 0.92) + (0.50 * 0.68) + (0.30 * 0.56)) = 69.2
prize_bonus = ((2.1 + 0.8 + 0.2) / 3) * 2.0 = 2.07
invalid_action_penalty = 0
slow_game_penalty = 0
local_kaggle_proxy = 71.27
```

## Decision Rules

Use the proxy comparatively, not as an absolute Kaggle prediction.

| Result | Decision |
| --- | --- |
| Candidate proxy is lower than default `sequence` by 2+ points | Reject or revise. |
| Candidate proxy is within +/-2 points | Repeat with more games or inspect matchup-specific tradeoffs. |
| Candidate proxy is higher by 2+ points and invalid actions are zero | Promote to final 300-game run. |
| Candidate proxy is higher by 4+ points after 300-game run | Strong Kaggle submission candidate. |

## Limitations

- This proxy does not model the live Kaggle opponent distribution exactly.
- It depends on the quality and diversity of `data/deck_db/decks.json`.
- It does not measure hidden leaderboard variance.
- It can miss regressions against specific archetypes if those decks are underrepresented locally.

When in doubt, prefer a candidate that improves deck-db rotation and keeps invalid actions at zero over one that only improves random mirror.

