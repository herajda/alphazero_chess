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
            result = self._board.result(claim_draw=True)
            if result == "*":
                return None
            elif result == "1-0":
                return 1
            elif result == "0-1":
                return 0
            elif result == "1/2-1/2":
                return -1
            return None
    @property
    def to_play(self):
        return int(self._board.turn)
class ChessGUI:
    # Dictionary mapping piece symbols to Unicode chess emojis
    PIECE_EMOJIS = {
        # White pieces
        'K': '♔',  # White King
        'Q': '♕',  # White Queen
        'R': '♖',  # White Rook
        'B': '♗',  # White Bishop
        'N': '♘',  # White Knight
        'P': '♙',  # White Pawn
        # Black pieces
        'k': '♚',  # Black King
        'q': '♛',  # Black Queen
        'r': '♜',  # Black Rook
        'b': '♝',  # Black Bishop
        'n': '♞',  # Black Knight
        'p': '♟',  # Black Pawn
    }

    def __init__(self, game):
        """Initialize the GUI with a reference to the ChessGame."""
        self.game = game
        self.root = tk.Tk()
        self.root.title("Chess Game")
        self.labels = {}
        self.game_ended = False

        # Create board labels with alternating colors
        for row in range(8):
            for col in range(8):
                bg_color = 'white' if (row + col) % 2 == 0 else 'gray'
                # Increased width and height to better display emojis
                label = tk.Label(self.root, width=4, height=2, bg=bg_color, 
                               borderwidth=1, relief="solid", font=("Arial", 20))
                label.grid(row=7 - row, column=col)
                self.labels[(row, col)] = label

        # Initial update
        self.update_display()

    def update_display(self):
        """Update the GUI to reflect the current board state and handle game end."""
        # Update the board display with current pieces using emojis
        board = self.game.get_board_state()
        for row in range(8):
            for col in range(8):
                square = chess.square(col, row)
                piece = board.piece_at(square)
                if piece:
                    text = self.PIECE_EMOJIS.get(piece.symbol(), piece.symbol())
                else:
                    text = " "  # Empty space instead of dot
                self.labels[(row, col)].config(text=text)

        # Check if the game has ended
        winner = self.game.winner
        if winner is not None and not self.game_ended:
            self.game_ended = True
            # Determine the color based on the winner
            color = 'red' if winner == 1 else 'green' if winner == 0 else 'yellow'
            # Change background color of all squares
            for label in self.labels.values():
                label.config(bg=color)
            # Schedule the window to close after 10 seconds (10000 ms)
            self.root.after(10000, self.close)

        # Refresh the GUI to show changes immediately
        self.root.update()

    def close(self):
        """Close the GUI window."""
        self.root.destroy()
