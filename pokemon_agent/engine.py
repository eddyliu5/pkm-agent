from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actions import legal_actions, sanitize_selection
from .cg_loader import ensure_cg_path
from .features import combine_feature_dicts, extract_action_features
from .policy import Policy
from .reward import transition_reward
from .state import build_agent_state

ensure_cg_path()

from cg.api import Observation, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


@dataclass
class Transition:
    player: int
    turn: int
    selected: list[int]
    labels: list[str]
    features: dict[str, float]
    reward: float


@dataclass
class EpisodeResult:
    winner: int
    reason: int | None
    steps: int
    invalid_actions: int
    final_prizes: tuple[int, int]
    transitions: list[Transition] = field(default_factory=list)

    def outcome_for(self, player: int) -> float:
        if self.winner == player:
            return 1.0
        if self.winner == 2:
            return 0.0
        return -1.0


def _result_reason(obs: Observation) -> int | None:
    for log in reversed(obs.logs or []):
        if getattr(log, "result", None) is not None:
            return getattr(log, "reason", None)
    return None


def run_episode(
    policy0: Policy,
    policy1: Policy,
    deck0: list[int],
    deck1: list[int],
    *,
    config: dict,
    max_steps: int = 2000,
    collect_transitions: bool = False,
) -> EpisodeResult:
    obs_dict, start_data = battle_start(deck0, deck1)
    if obs_dict is None:
        raise RuntimeError(f"battle_start failed: player={start_data.errorPlayer} type={start_data.errorType}")

    transitions: list[Transition] = []
    invalid_actions = 0
    steps = 0
    obs = to_observation_class(obs_dict)
    try:
        while obs.current is not None and obs.current.result == -1 and steps < max_steps:
            player = obs.current.yourIndex
            policy = policy0 if player == 0 else policy1
            action_views = legal_actions(obs)
            prev_state = build_agent_state(obs, perspective=player, actions=action_views)
            selected_raw = policy.choose(obs)
            selected, invalid = sanitize_selection(obs, selected_raw)
            invalid_actions += int(invalid)
            selected_views = [action for action in action_views if action.index in selected]
            feature_dicts = [extract_action_features(prev_state, action, config) for action in selected_views]
            combined_features = combine_feature_dicts(feature_dicts)
            next_obs_dict = battle_select(selected)
            next_obs = to_observation_class(next_obs_dict)
            reward = transition_reward(prev_state, next_obs, player, selected_views)
            if collect_transitions:
                transitions.append(
                    Transition(
                        player=player,
                        turn=prev_state.turn,
                        selected=selected,
                        labels=[action.label for action in selected_views],
                        features=combined_features,
                        reward=reward,
                    )
                )
            obs = next_obs
            steps += 1
    finally:
        battle_finish()

    final = obs.current
    if final is None:
        winner = -1
        final_prizes = (0, 0)
    else:
        winner = final.result
        final_prizes = (len(final.players[0].prize), len(final.players[1].prize))
    if steps >= max_steps and winner == -1:
        winner = 2
    return EpisodeResult(
        winner=winner,
        reason=_result_reason(obs),
        steps=steps,
        invalid_actions=invalid_actions,
        final_prizes=final_prizes,
        transitions=transitions,
    )


def policy_name(policy: Any) -> str:
    return policy.__class__.__name__

