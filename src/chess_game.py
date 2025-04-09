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
    N = 8
    # {king, queen, rook, bishop, knight, pawn, empty} x {white and black}
    C = 119 
    

    def __init__(self):
        self._board = chess.Board()

    def clone(self, swap_players=False):
        clone = ChessGame()
        clone._board = self._board.copy()
        if swap_players:
            clone._board.turn = not self._board.turn
        return clone

    @property
    def board(self):
        return chess_moves.board_to_tensor(self._board) 

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
        return chess_moves.legal_moves_to_array(self._board)

    def move(self, action):
        move = chess.Move.from_uci(chess_moves.action_to_uci(self._board, action))
        if move not in self._board.legal_moves:
            raise ValueError(f"Invalid move: {action}")
        self._board.push(move)

    def to_play(self):
        return int(self._board.turn)

