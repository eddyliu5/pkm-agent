from __future__ import annotations

from .actions import ActionView
from .cg_loader import ensure_cg_path
from .state import AgentState, build_agent_state

ensure_cg_path()

from cg.api import Observation  # noqa: E402


def transition_reward(
    prev_state: AgentState,
    next_obs: Observation,
    acting_player: int,
    selected_actions: list[ActionView],
) -> float:
    """Return shaped reward from the acting player's perspective."""
    reward = 0.0
    selected_labels = {action.option_type for action in selected_actions}
    if "END" in selected_labels:
        if any(not action.is_end and not action.is_attack for action in prev_state.legal_actions):
            reward -= 3.0
    if "ATTACK" in selected_labels:
        reward += 0.3
    if "ATTACH" in selected_labels:
        reward += 0.2
    if "EVOLVE" in selected_labels:
        reward += 0.4

    if next_obs.current is None:
        return reward

    next_state = build_agent_state(next_obs, perspective=acting_player)
    prize_gain = prev_state.my_prizes_remaining - next_state.my_prizes_remaining
    prize_loss = prev_state.opp_prizes_remaining - next_state.opp_prizes_remaining
    reward += 8.0 * prize_gain
    reward -= 8.0 * prize_loss

    if next_state.my_energy_in_play > prev_state.my_energy_in_play:
        reward += 0.5 * (next_state.my_energy_in_play - prev_state.my_energy_in_play)
    if len(next_state.my_bench) > len(prev_state.my_bench):
        reward += 0.2 * (len(next_state.my_bench) - len(prev_state.my_bench))

    if next_state.result != -1:
        if next_state.result == acting_player:
            reward += 100.0
        elif next_state.result == 2:
            reward += 0.0
        else:
            reward -= 100.0

    return reward

