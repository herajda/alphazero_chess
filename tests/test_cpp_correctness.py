from __future__ import annotations

import math
import importlib.util

import pytest

MISSING_DEPS = [
    name
    for name in ("chess", "numpy", "torch", "chess_engine")
    if importlib.util.find_spec(name) is None
]
pytestmark = pytest.mark.skipif(
    bool(MISSING_DEPS),
    reason=f"missing test dependencies: {', '.join(MISSING_DEPS)}",
)

if not MISSING_DEPS:
    import chess
    import numpy as np
    import torch
    import chess_engine
    import chess_moves
else:
    chess = None
    chess_engine = None
    chess_moves = None
    np = None
    torch = None


ROUND_TRIP_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
    "8/P6k/8/8/8/8/6K1/8 w - - 0 1",
    "8/6k1/8/8/8/8/p5K1/8 b - - 0 1",
]


def test_cpp_action_ids_round_trip_to_legal_uci() -> None:
    for fen in ROUND_TRIP_FENS:
        board = chess.Board(fen)
        for action in chess_engine.legal_actions(fen):
            uci = chess_engine.action_to_uci(fen, int(action))
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves
            assert chess_moves.uci_to_action(board, uci) == int(action)


@pytest.mark.parametrize(
    "fen",
    [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 12 24",
        "8/6k1/8/8/8/8/p5K1/8 b - - 0 1",
    ],
)
def test_cpp_tensor_matches_python_encoder_for_fen_only_positions(fen: str) -> None:
    expected = chess_moves.board_to_tensor(chess.Board(fen))
    actual = np.asarray(chess_engine.encode_tensor(fen), dtype=np.float32).reshape(8, 8, 119)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_legal_policy_normalisation_accepts_probabilities_and_logits() -> None:
    legal = [10, 20]

    probs = np.zeros(4672, dtype=np.float32)
    probs[10] = 0.2
    probs[20] = 0.3
    probs[30] = 0.5
    normalised_probs = chess_engine.normalise_legal_policy(probs.tolist(), legal)
    assert normalised_probs == pytest.approx([0.4, 0.6])

    logits = np.full(4672, -100.0, dtype=np.float32)
    logits[10] = -2.0
    logits[20] = 0.0
    normalised_logits = chess_engine.normalise_legal_policy(logits.tolist(), legal)
    expected_second = 1.0 / (1.0 + math.exp(-2.0))
    assert normalised_logits == pytest.approx([1.0 - expected_second, expected_second])


def test_terminal_value_is_from_side_to_move_perspective() -> None:
    board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1")
    board.push(chess.Move.from_uci("g6g7"))
    assert board.is_checkmate()
    assert chess_engine.terminal_value(board.fen()) == pytest.approx(-1.0)


def test_terminal_value_detects_insufficient_material() -> None:
    assert chess_engine.terminal_value("8/8/8/8/8/8/8/K6k w - - 0 1") == pytest.approx(0.0)


def test_terminal_value_detects_threefold_repetition_from_history() -> None:
    assert chess_engine.terminal_value_after_uci([
        "g1f3", "g8f6", "f3g1", "f6g8",
        "g1f3", "g8f6", "f3g1", "f6g8",
    ]) == pytest.approx(0.0)


def test_puct_score_uses_parent_perspective_for_child_value() -> None:
    child_is_losing_for_player_to_move = chess_engine.puct_score_from_parent(-1.0, 0.5, 10, 1)
    child_is_winning_for_player_to_move = chess_engine.puct_score_from_parent(1.0, 0.5, 10, 1)
    assert child_is_losing_for_player_to_move > child_is_winning_for_player_to_move


def test_mcts_selects_a_forced_mate_with_uniform_network(tmp_path) -> None:
    class UniformModel(torch.nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            policy = torch.zeros(batch, 4672, dtype=torch.float32, device=x.device)
            value = torch.zeros(batch, 1, dtype=torch.float32, device=x.device)
            return policy, value

    model_path = tmp_path / "uniform.pt"
    example = torch.zeros(1, 8, 8, 119, dtype=torch.float32)
    torch.jit.trace(UniformModel(), example).save(str(model_path))

    fen = "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1"
    selected = chess_engine.select_move(str(model_path), fen, 200, 0.3, 0.0, 30)
    board = chess.Board(fen)
    board.push(chess.Move.from_uci(chess_engine.action_to_uci(fen, int(selected))))
    assert board.is_checkmate()
