import numpy as np
import chess

# Define directions and knight moves
DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]  # N, NE, E, SE, S, SW, W, NW
KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]  # Clockwise from (1, 2)

def get_moves(board, move):
    "from_square, to_square, true_from_square, true_to_square"
    if board.turn == chess.WHITE:
        return move.from_square, move.to_square, move.from_square, move.to_square
    else:
        true_from_square = move.from_square
        true_to_square = move.to_square
        # Flip the board for Black perspective
        from_square = chess.square(chess.square_file(true_from_square), 7 - chess.square_rank(true_from_square))
        to_square = chess.square(chess.square_file(true_to_square), 7 - chess.square_rank(true_to_square))
        return from_square, to_square, true_from_square, true_to_square
        
def move_to_action(board, move):
    """Map a chess.Move to (from_file, from_rank, move_type_index) or None if not in 73 types."""
    from_square, to_square, true_from_square, true_to_square = get_moves(board, move)
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    delta_file = chess.square_file(to_square) - from_file
    delta_rank = chess.square_rank(to_square) - from_rank

    # Underpromotion moves
    if move.promotion and move.promotion in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
        expected_rank = 1 
        if abs(delta_file) <= 1 and delta_rank == expected_rank:
            direction = {-1: 0, 0: 1, 1: 2}[delta_file]  # NW: 0, N: 1, NE: 2
            piece_idx = move.promotion - 2  # KNIGHT: 0, BISHOP: 1, ROOK: 2
            move_type_index = 64 + 3 * piece_idx + direction
            return flatten_action(from_file, from_rank, move_type_index)
        return None

    # Knight moves
    piece = board.piece_at(true_from_square)
    if piece and piece.piece_type == chess.KNIGHT:
        delta = (delta_file, delta_rank)
        if delta in KNIGHT_DELTAS:
            knight_idx = KNIGHT_DELTAS.index(delta)
            move_type_index = 56 + knight_idx
            return flatten_action(from_file, from_rank, move_type_index)
        return None

    # Queen moves (sliding moves)
    for dir_idx, (df, dr) in enumerate(DIRECTIONS):
        if df == 0 and delta_file == 0 and 1 <= abs(delta_rank) <= 7:
            k = abs(delta_rank)
            if delta_rank > 0:  # N
                move_type_index = 0 * 7 + (k - 1)
            else:  # S
                move_type_index = 4 * 7 + (k - 1)
            return flatten_action(from_file, from_rank, move_type_index)
        elif dr == 0 and delta_rank == 0 and 1 <= abs(delta_file) <= 7:
            k = abs(delta_file)
            if delta_file > 0:  # E
                move_type_index = 2 * 7 + (k - 1)
            else:  # W
                move_type_index = 6 * 7 + (k - 1)
            return flatten_action(from_file, from_rank, move_type_index)
        elif delta_file != 0 and delta_rank != 0 and abs(delta_file) == abs(delta_rank) and 1 <= abs(delta_file) <= 7:
            k = abs(delta_file)
            if delta_file > 0 and delta_rank > 0:  # NE
                move_type_index = 1 * 7 + (k - 1)
            elif delta_file > 0 and delta_rank < 0:  # SE
                move_type_index = 3 * 7 + (k - 1)
            elif delta_file < 0 and delta_rank < 0:  # SW
                move_type_index = 5 * 7 + (k - 1)
            elif delta_file < 0 and delta_rank > 0:  # NW
                move_type_index = 7 * 7 + (k - 1)
            return flatten_action(from_file, from_rank, move_type_index)
    return None  # Not one of the 73 move types

def legal_moves_to_array(board):
    actions = []
    for mv in board.legal_moves:
        idx = move_to_action(board, mv)
        if idx is not None:
            actions.append(idx)
    return actions

def uci_to_action(board, uci):
    """Convert UCI string to flattened action number (0–4671) or None if invalid."""
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        return None
    action = move_to_action(board, move)
    if action:
        from_file, from_rank, move_type = action
        return flatten_action(from_file, from_rank, move_type)
    return None

def action_to_uci(board, action):
    """Convert flattened action number to UCI string or None if invalid."""
    
    if not 0 <= action < 4672:
        return None
    move_type = action % 73 
    temp = action // 73
    file = temp // 8
    rank = temp % 8
    if board.turn == chess.BLACK:
        rank = 7 - rank
    from_square = chess.square(file, rank)

    if 0 <= move_type <= 55:  # Queen moves
        dir_idx = move_type // 7
        step = (move_type % 7) + 1
        df, dr = DIRECTIONS[dir_idx]
        if board.turn == chess.BLACK:
            dr = -dr
        to_file = file + df * step
        to_rank = rank + dr * step
        if 0 <= to_file < 8 and 0 <= to_rank < 8:
            to_square = chess.square(to_file, to_rank)
            move = chess.Move(from_square, to_square)
        else:
            return None
    elif 56 <= move_type <= 63:  # Knight moves
        knight_idx = move_type - 56
        df, dr = KNIGHT_DELTAS[knight_idx]
        if board.turn == chess.BLACK:
            dr = -dr
        to_file = file + df
        to_rank = rank + dr
        if 0 <= to_file < 8 and 0 <= to_rank < 8:
            to_square = chess.square(to_file, to_rank)
            move = chess.Move(from_square, to_square)
        else:
            return None
    elif 64 <= move_type <= 72:  # Underpromotions
        piece_idx = (move_type - 64) // 3
        dir_idx = (move_type - 64) % 3
        piece = [chess.KNIGHT, chess.BISHOP, chess.ROOK][piece_idx]
        delta_file = [-1, 0, 1][dir_idx]  # NW, N, NE
        delta_rank = 1 if board.turn else -1
        to_file = file + delta_file
        to_rank = rank + delta_rank
        promotion_rank = 7 if board.turn else 0
        if 0 <= to_file < 8 and to_rank == promotion_rank:
            to_square = chess.square(to_file, to_rank)
            move = chess.Move(from_square, to_square, promotion=piece)
        else:
            return None
    else:
        return None
    if move in board.legal_moves:
        return move.uci()
    else:
        move.promotion = chess.QUEEN 
        if move in board.legal_moves:
            return move.uci()

    return None

def flatten_action(file, rank, move_type):
    """
    Convert a non-flattened action (file, rank, move_type) to a flattened action number (0–4671).
    
    Args:
        file (int): File index (0–7, a–h).
        rank (int): Rank index (0–7, 1–8).
        move_type (int): Move type index (0–72).
    
    Returns:
        int: Flattened action number, or None if inputs are out of bounds.
    """
    if not (0 <= file < 8 and 0 <= rank < 8 and 0 <= move_type < 73):
        return None
    return file * 8 * 73 + rank * 73 + move_type

def unflatten_action(action):
    """
    Convert a flattened action number (0–4671) to a non-flattened (file, rank, move_type) tuple.
    
    Args:
        action (int): Flattened action number (0–4671).
    
    Returns:
        tuple: (file, rank, move_type), or None if action is out of bounds.
    """
    if not (0 <= action < 4672):
        return None
    move_type = action // 64
    temp = action % 64
    rank = temp // 8
    file = temp % 8
    return (file, rank, move_type)


def get_history_and_counts(board, max_history=8):
    """
    Retrieve the last max_history board positions and their repetition counts.
    Returns a list of boards (or None for padding) and corresponding counts.
    """
    history = []
    current_board = board.copy()
    # Collect boards from current to initial by undoing moves
    while current_board.move_stack:
        history.append(current_board.copy())
        current_board.pop()
    history.append(current_board.copy())  # Add initial board
    history.reverse()  # Reorder from initial to current
    
    # Compute cumulative repetition counts using FEN
    fen_counts = {}
    counts = []
    for b in history:
        fen = b.fen()  # FEN includes position, turn, castling, en passant
        fen_counts[fen] = fen_counts.get(fen, 0) + 1
        counts.append(fen_counts[fen])
    
    # Take the last max_history positions, padding with None if needed
    if len(history) > max_history:
        history = history[-max_history:]
        counts = counts[-max_history:]
    else:
        padding = [None] * (max_history - len(history))
        history = padding + history
        counts = [0] * len(padding) + counts
    return history, counts

def get_piece_at(board, i, j, p1):
    """
    Get the piece at position (i, j) oriented to P1's perspective.
    - P1 = White: (0,0) is a1, (7,7) is h8.
    - P1 = Black: (0,0) is a8, (7,7) is h1.
    """
    if p1 == chess.WHITE:
        square = chess.square(j, i)  # file=j, rank=i
    else:
        square = chess.square(j, 7 - i)  # file=j, rank=7-i
    return board.piece_at(square)

def board_to_tensor(board):
    """
    Transform a python-chess Board object into an 8x8x119 tensor for AlphaZero.
    """
    # Determine the current player (P1)
    p1 = board.turn  # True for White, False for Black
    p2 = not p1      # Opponent
    
    # Get history and repetition counts
    history, counts = get_history_and_counts(board, max_history=8)
    
    # Initialize the tensor
    tensor = np.zeros((8, 8, 119), dtype=np.float32)
    
    # Fill historical planes (8 time-steps, 14 planes each)
    for t in range(8):
        if history[t] is None:
            # Missing positions (before game start) remain zero
            continue
        b = history[t]
        # Piece planes
        for i in range(8):
            for j in range(8):
                piece = get_piece_at(b, i, j, p1)
                if piece is not None:
                    color = piece.color
                    # piece_type: 1=Pawn, 2=Knight, ..., 6=King; map to 0-5
                    piece_type = piece.piece_type - 1
                    if color == p1:
                        # P1's pieces: planes 0-5
                        plane_index = piece_type
                    else:
                        # P2's pieces: planes 6-11
                        plane_index = 6 + piece_type
                    tensor[i, j, t * 14 + plane_index] = 1
        # Repetition planes
        count = counts[t]
        if count >= 2:
            # Plane 12: 1 if position has repeated at least once before
            tensor[:, :, t * 14 + 12] = 1
        # Plane 13: 1 if count >= 3, but since game continues, count < 3, so always 0
    
    # Fill constant planes (indices 112-118)
    # Player color
    tensor[:, :, 112] = 1 if p1 == chess.WHITE else 0
    # Total move count (ply count)
    total_moves = len(board.move_stack)
    tensor[:, :, 113] = total_moves / 1000.0  # Normalize (assuming max 1000 moves)
    # P1 castling rights
    tensor[:, :, 114] = 1 if board.has_kingside_castling_rights(p1) else 0
    tensor[:, :, 115] = 1 if board.has_queenside_castling_rights(p1) else 0
    # P2 castling rights
    tensor[:, :, 116] = 1 if board.has_kingside_castling_rights(p2) else 0
    tensor[:, :, 117] = 1 if board.has_queenside_castling_rights(p2) else 0
    # No-progress count (halfmove clock)
    tensor[:, :, 118] = board.halfmove_clock / 50.0  # Normalize (assuming max 50)
    
    return tensor


#import unittest
#
#class TestBoardToTensor(unittest.TestCase):
#    def setUp(self):
#        # Common setup for all tests
#        self.tensor_shape = (8, 8, 119)
#
#    def test_initial_position(self):
#        """Test the starting position of a chess game."""
#        board = chess.Board()
#        
#        tensor = board_to_tensor(board)
#        # Check tensor shape
#        self.assertEqual(tensor.shape, self.tensor_shape)
#        
#        # Check piece planes (White to move, P1 = White)
#        # White pawns on rank 2 (row 1 in 0-based index)
#        self.assertTrue(np.all(tensor[1, :, 98] == 1))  # Plane 0: P1 pawns
#        self.assertTrue(np.all(tensor[0, :, 98] == 0))  # No pawns on rank 1
#        # White king on e1 (row 0, col 4)
#        self.assertEqual(tensor[0, 4, 103], 1)  # Plane 5: P1 king
#        # Black pawns on rank 7 (row 6)
#        self.assertTrue(np.all(tensor[6, :, 104] == 1))  # Plane 6: P2 pawns
#        
#        # Check constant planes
#        # P1 = White (color plane = 1)
#        self.assertTrue(np.all(tensor[:, :, 112] == 1))
#        # Total move count = 0 (normalized to 0.0)
#        self.assertTrue(np.all(tensor[:, :, 113] == 0.0))
#        # Castling rights all available
#        self.assertTrue(np.all(tensor[:, :, 114] == 1))  # P1 kingside
#        self.assertTrue(np.all(tensor[:, :, 115] == 1))  # P1 queenside
#        self.assertTrue(np.all(tensor[:, :, 116] == 1))  # P2 kingside
#        self.assertTrue(np.all(tensor[:, :, 117] == 1))  # P2 queenside
#        # No-progress count = 0
#        self.assertTrue(np.all(tensor[:, :, 118] == 0.0))
#        
#        # Repetition planes (no repeats yet)
#        self.assertTrue(np.all(tensor[:, :, 110] == 0))  # t=0, repetition plane 1
#        self.assertTrue(np.all(tensor[:, :, 111] == 0))  # t=0, repetition plane 2

    #def test_mid_game_position(self):
    #    """Test a mid-game position after a few moves."""
    #    board = chess.Board()
    #    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    #    for move in moves:
    #        board.push_uci(move)
    #    
    #    tensor = board_to_tensor(board)
    #    
    #    # Check shape
    #    self.assertEqual(tensor.shape, self.tensor_shape)
    #    
    #    # White to move (P1 = White)
    #    # e4 pawn (row 3, col 4)
    #    self.assertEqual(tensor[3, 4, 14] , 1)  # t=1 (second position), P1 pawn
    #    # Nf3 (row 2, col 5)
    #    self.assertEqual(tensor[2, 5, 15] , 1)  # t=1, P1 knight
    #    # Black e5 pawn (row 4, col 4)
    #    self.assertEqual(tensor[4, 4, 20] , 1)  # t=1, P2 pawn
    #    # Nc6 (row 5, col 2)
    #    self.assertEqual(tensor[5, 2, 21] , 1)  # t=1, P2 knight
    #    
    #    # Constant planes
    #    self.assertTrue(np.all(tensor[:, :, 112] == 1))  # P1 = White
    #    self.assertTrue(np.all(tensor[:, :, 113] == 4 / 1000.0))  # 4 plies
    #    self.assertTrue(np.all(tensor[:, :, 114] == 1))  # Castling intact
    #    self.assertTrue(np.all(tensor[:, :, 118] == 2 / 50.0))  # 2 half-moves

    #def test_black_to_move(self):
    #    """Test a position with Black to move (orientation flipped)."""
    #    board = chess.Board()
    #    board.push_uci("e2e4")
    #    tensor = board_to_tensor(board)
    #    
    #    # Black to move (P1 = Black)
    #    # Board flipped: Black pieces at bottom (rows 0-1)
    #    self.assertEqual(tensor.shape, self.tensor_shape)
    #    self.assertTrue(np.all(tensor[:, :, 112] == 0))  # P1 = Black
    #    # Black pawns on rank 7 -> row 1
    #    self.assertTrue(np.all(tensor[1, :, 0] == 1))  # P1 pawns
    #    # White pawn on e4 -> row 5, col 4
    #    self.assertEqual(tensor[5, 4, 6] , 1)  # t=0, P2 pawn
    #    
    #    self.assertTrue(np.all(tensor[:, :, 113] == 1 / 1000.0))  # 1 move

    #def test_castling_rights_lost(self):
    #    """Test a position where castling rights are lost."""
    #    board = chess.Board()
    #    # Move king, losing castling rights for White
    #    moves = ["e2e4", "e7e5", "e1e2"]
    #    for move in moves:
    #        board.push_uci(move)
    #    
    #    tensor = board_to_tensor(board)
    #    
    #    # Black to move (P1 = Black)
    #    self.assertTrue(np.all(tensor[:, :, 112] == 0))
    #    # P1 (Black) castling intact
    #    self.assertTrue(np.all(tensor[:, :, 114] == 1))
    #    self.assertTrue(np.all(tensor[:, :, 115] == 1))
    #    # P2 (White) castling lost
    #    self.assertTrue(np.all(tensor[:, :, 116] == 0))
    #    self.assertTrue(np.all(tensor[:, :, 117] == 0))

    #def test_repetitions(self):
    #    """Test a position with repetitions."""
    #    board = chess.Board()
    #    # Create a threefold repetition: Nf3, Ng5, Nf3, Ng5
    #    moves = ["g1f3", "g8f6", "f3g5", "f6g4", "g5f3", "g4f6"]
    #    for move in moves:
    #        board.push_uci(move)
    #    
    #    tensor = board_to_tensor(board)
    #    
    #    # Check repetition planes
    #    # t=5 (current position), should show repetition
    #    self.assertTrue(np.all(tensor[:, :, 5 * 14 + 12] == 1))  # Count >= 2
    #    
    #    # Earlier positions should reflect their counts
    #    self.assertTrue(np.all(tensor[:, :, 0 * 14 + 12] == 0))  # t=0, no repeats

    #def test_short_history(self):
    #    """Test a position with fewer than 8 moves."""
    #    board = chess.Board()
    #    board.push_uci("e2e4")
    #    tensor = board_to_tensor(board)
    #    
    #    # Check that earlier time-steps (t=0 to t=6) are zero
    #    for t in range(7):  # t=0 to t=6
    #        self.assertTrue(np.all(tensor[:, :, t * 14:(t + 1) * 14] == 0))
    #    # t=7 should have the current position
    #    self.assertEqual(tensor[3, 4, 7 * 14], 1)  # e4 pawn

#if __name__ == "__main__":
#    unittest.main()
