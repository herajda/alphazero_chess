#!/usr/bin/env python3
import sys

import numpy as np
import chess
import chess_moves
import tkinter as tk
from threading import Lock

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
    
    def __init__(self, gui_enabled=False):
        """Initialize the ChessGame with an optional GUI."""
        self._board = chess.Board()
        self.lock = Lock()  # For thread safety, if needed elsewhere
        self.gui = None
        if gui_enabled:
            self.gui = ChessGUI(self)  # Create GUI instance if enabled

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

    def valid(self, action):
        """Check if the action is valid."""
        valid_actions = self.valid_actions() 
        return valid_actions[action]

    def valid_actions(self):
        """Return the list of valid actions."""
        return chess_moves.legal_moves_to_array(self._board)

    def move(self, action):
        """Make a move and update the GUI if it exists."""
        with self.lock:
            # Assuming action is in a format convertible to UCI
            move = chess.Move.from_uci(chess_moves.action_to_uci(self._board, action))
            if move not in self._board.legal_moves:
                raise ValueError(f"Invalid move: {action}")
            self._board.push(move)
        # Update GUI after the move, if GUI is enabled
        if self.gui:
            self.gui.update_display()
    def get_board_state(self):
        """Return a copy of the current board state, safely."""
        with self.lock:
            return self._board.copy()
    @property
    def winner(self):
        """Determine the winner, if any."""
        with self.lock:
            if self._board.is_checkmate():
                return int(not self._board.turn)  # 1 for Black, 0 for White
            elif self._board.is_stalemate() or self._board.is_insufficient_material():
                return -1  # Draw
            return None
    @property
    def to_play(self):
        return int(self._board.turn)

class ChessGUI:
    def __init__(self, game):
        """Initialize the GUI with a reference to the ChessGame."""
        self.game = game
        self.root = tk.Tk()
        self.root.title("Chess Game")
        self.labels = {}  # To store board square labels

        # Create an 8x8 grid of labels for the chessboard
        for row in range(8):
            for col in range(8):
                label = tk.Label(self.root, width=4, height=2, borderwidth=1, relief="solid")
                label.grid(row=7 - row, column=col)  # Flip row to match chess notation
                self.labels[(row, col)] = label

        # Initial display update
        self.update_display()

    def update_display(self):
        """Update the GUI to reflect the current board state."""
        board = self.game.get_board_state()
        for row in range(8):
            for col in range(8):
                square = chess.square(col, row)  # Convert to python-chess square index
                piece = board.piece_at(square)
                text = piece.symbol() if piece else "."
                self.labels[(row, col)].config(text=text)
        # Process GUI events without blocking
        self.root.update()

    def close(self):
        """Close the GUI window."""
        self.root.destroy()
