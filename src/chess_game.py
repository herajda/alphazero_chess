#!/usr/bin/env python3
import sys

import numpy as np
import chess
import chess_moves


class BoardGame:
    ACTIONS: int
    N: int
    C: int

    def clone(self, swap_players=False):
        """Clone the game state, optionally swapping the players."""

    @property
    def board(self):
        """Return the board as a NxNxC numpy array of features."""

    @property
    def board_internal(self):
        """Return the internal representation of board as a NxN numpy array."""

    @property
    def to_play(self):
        """Return the current player."""

    @property
    def winner(self):
        """Return the winner of the game, or `None` if the game is not over."""

    def valid(self, action):
        """Return whether the given action is valid."""

    def valid_actions(self):
        """Return the list of valid actions."""

    def move(self, action):
        """Execute the given action."""

class ChessGame(BoardGame):
    ACTIONS = 4672 #8x8x73

    def __init__(self):
        self._board = chess.Board()
        self._winner = None

    def clone(self, swap_players=False):
        clone = ChessGame()
        clone._board = self._board.copy()
        if swap_players:
            clone._board.turn = not self._board.turn
        return clone

    @property
    def board(self):
        return self._get_board_features()

    @property
    def board_internal(self):
        return np.array(self._board.piece_map().values())

    @property
    def to_play(self):
        return int(self._board.turn)

    @property
    def winner(self):
        if self._board.is_checkmate():
            return int(not self._board.turn)
        elif self._board.is_stalemate() or self._board.is_insufficient_material():
            return -1
        else:
            return None

    def valid(self, action):
        """Check if the action is valid."""
        valid_actions = self.valid_actions() 
        return valid_actions[action]

    def valid_actions(self):
        """Return the list of valid actions."""
        return chess_moves.get_legal_moves(self._board)

    def move(self, action):
        move = chess.Move.from_uci(action)
        if move not in self._board.legal_moves:
            raise ValueError(f"Invalid move: {action}")
        self._board.push(move)

