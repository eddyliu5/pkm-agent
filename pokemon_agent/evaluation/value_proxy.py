from __future__ import annotations

from ..state import AgentState


class HeuristicValueProxy:
    """Heuristic state-delta proxy; not a learned calibrated value model."""

    def evaluate(self, root_state: AgentState, state: AgentState | None) -> float:
        if state is None:
            return 0.0
        if state.result == root_state.player_index:
            return 1.0
        if state.result not in (-1, 2, root_state.player_index):
            return -1.0
        prize_gain = float(root_state.my_prizes_remaining - state.my_prizes_remaining)
        prize_loss = float(root_state.opp_prizes_remaining - state.opp_prizes_remaining)
        energy_delta = float(state.my_energy_in_play - root_state.my_energy_in_play) / 6.0
        board_delta = float(len(state.my_bench) - len(root_state.my_bench)) / 5.0
        prize_value = (prize_gain - prize_loss) / 6.0
        return max(-1.0, min(1.0, prize_value + 0.25 * energy_delta + 0.1 * board_delta))


StateDeltaValueProxy = HeuristicValueProxy
