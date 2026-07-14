# Local RL Engine Guide

The competition-provided C++ engine source is installed locally at:

```text
data/pokemon-tcg-ai-battle/engine_source/
```

A local derivative with deterministic RL hooks is at:

```text
data/pokemon-tcg-ai-battle/rl_engine/
```

Both directories are covered by `data/pokemon-tcg-ai-battle/` in `.gitignore`. They are competition-only material, are not open source, must not be redistributed or packaged with submissions, and must be deleted when required by the competition license. Preserve their included README and license text.

## When to use it

Use the RL engine for:

- deterministic self-play trajectory generation;
- policy/value training data;
- counterfactual rollouts from a cloned state;
- paired experiments with controlled engine RNG;
- transition, reward, and legal-action debugging;
- privileged-state critic targets or diagnostics, provided the deployed policy still receives only legal public observations.

Do not use it for:

- Kaggle submission packaging or runtime inference;
- final honest evaluation intended to reproduce the competition interface;
- giving hidden state to a policy that will be deployed;
- replacing the normal `cg` wrapper in the production agent;
- non-competition projects or redistribution.

For those cases, continue using the bundled public-observation `cg` API and the existing `pokemon_agent` evaluation/submission paths.

## Build and verify

From the repository root:

```bash
data/pokemon-tcg-ai-battle/rl_engine/build_local.sh
./.venv/bin/python data/pokemon-tcg-ai-battle/rl_engine/python/smoke_test.py
```

The smoke test checks:

- identical seeds produce identical public and hidden state;
- different seeds diverge;
- `clone()` preserves full engine and RNG state;
- identical actions remain identical across original and cloned branches;
- synchronous vector stepping preserves branch equivalence.

The current wrapper supports macOS and Linux. Engine thread safety has not been established; use independent processes for parallel self-play rather than calling one loaded library concurrently from multiple threads.

## Python API

```python
import sys
sys.path.insert(0, "data/pokemon-tcg-ai-battle/rl_engine/python")

from ptcg_rl import RLEnv, RLVectorEnv

env = RLEnv()
observation = env.reset(deck0, deck1, seed=13)

branch = env.clone()
next_observation = branch.step([option_index])

# Training/debugging only. Never provide this to the deployed policy.
hidden_state = branch.full_state()

branches = RLVectorEnv.from_clone(env, count=8)
next_observations = branches.step(actions)

branches.close()
branch.close()
env.close()
```

`RLEnv` exposes deterministic `reset`, independent `clone`, `clone_many`, validated `step`, `full_state`, `done`, and `close`. `RLVectorEnv` is currently a synchronous convenience wrapper, not a native vectorized engine.

## Local implementation notes

The derivative exports three additions over the competition ABI:

- `BattleStartSeeded`: deterministic reset with a caller-provided 32-bit seed;
- `BattleClone`: deep copy of game state and `std::mt19937` state with the cloned `State.game` pointer rebound correctly;
- `FullStateJson`: privileged visualizer state for training/debugging.

The untouched source remains separately under `engine_source/` for auditing. Local build output is `libptcg_rl.dylib` on macOS or `libptcg_rl.so` on Linux.

## Next RL implementation step

Keep the first training loop modest and measurable:

1. Generate public-observation trajectories using fixed deck/seed schedules.
2. Store legal-action masks, selected actions, terminal outcomes, and seat/deck metadata.
3. Train a policy head from replay/self-play actions and a value head from terminal returns.
4. Compare against `sequence`, the retained four-layer MLP, and XGBoost v1 on paired local games.
5. Add shallow policy/value-guided search only after the standalone value model is calibrated and improves controlled gameplay.

Do not begin with unrestricted MCTS or hidden-state policy inputs. The first objective is a reproducible self-play dataset and a policy/value baseline whose causal gameplay effect can be measured.
