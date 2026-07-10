# Historical Ground Truth: 20260708T021405Z

This is the pinned general-agent baseline for the repository.

| Field | Value |
| --- | --- |
| Agent | `sequence` |
| Opponent | `sequence` |
| Self and opponent deck pool | 30 replay-observed top-winrate decks |
| Schedule | 100 games for each self deck; opponent decks rotated across the pool |
| Total games | 3,000 |
| Wins / losses / draws | 1,571 / 1,429 / 0 |
| Pooled and mean-deck win rate | 52.3667% |
| Invalid actions | 0 |
| Mean prize differential | 0.1803 |
| Mean steps | 147.6353 |
| Seed schedule | Not recorded by the historical runner |

The result was 1.9667 percentage points above the contemporaneous
`sequence-starmie-froslass` result (50.40%) on the same 3,000-game deck matrix.

Use this record as evidence of broad deck robustness. Use the fixed-seed fast
or full protocol in [`../../README.md`](../../README.md) for new comparisons;
do not claim seed-paired equivalence to this historical run.
