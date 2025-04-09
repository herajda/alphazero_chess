import numpy as np
import chess

# Define directions and knight moves
DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]  # N, NE, E, SE, S, SW, W, NW
KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]  # Clockwise from (1, 2)

def move_to_action(board, move):
    """Map a chess.Move to (from_file, from_rank, move_type_index) or None if not in 73 types."""
    from_square = move.from_square
    to_square = move.to_square
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    delta_file = chess.square_file(to_square) - from_file
    delta_rank = chess.square_rank(to_square) - from_rank

    # Underpromotion moves
    if move.promotion and move.promotion in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
        expected_rank = 1 if board.turn else -1  # White: to rank 8, Black: to rank 1
        if abs(delta_file) <= 1 and delta_rank == expected_rank:
            direction = {-1: 0, 0: 1, 1: 2}[delta_file]  # NW: 0, N: 1, NE: 2
            piece_idx = move.promotion - 2  # KNIGHT: 0, BISHOP: 1, ROOK: 2
            move_type_index = 64 + 3 * piece_idx + direction
            return (from_file, from_rank, move_type_index)
        return None

    # Knight moves
    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.KNIGHT:
        delta = (delta_file, delta_rank)
        if delta in KNIGHT_DELTAS:
            knight_idx = KNIGHT_DELTAS.index(delta)
            move_type_index = 56 + knight_idx
            return (from_file, from_rank, move_type_index)
        return None

    # Queen moves (sliding moves)
    for dir_idx, (df, dr) in enumerate(DIRECTIONS):
        if df == 0 and delta_file == 0 and 1 <= abs(delta_rank) <= 7:
            k = abs(delta_rank)
            if delta_rank > 0:  # N
                move_type_index = 0 * 7 + (k - 1)
            else:  # S
                move_type_index = 4 * 7 + (k - 1)
            return (from_file, from_rank, move_type_index)
        elif dr == 0 and delta_rank == 0 and 1 <= abs(delta_file) <= 7:
            k = abs(delta_file)
            if delta_file > 0:  # E
                move_type_index = 2 * 7 + (k - 1)
            else:  # W
                move_type_index = 6 * 7 + (k - 1)
            return (from_file, from_rank, move_type_index)
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
            return (from_file, from_rank, move_type_index)
    return None  # Not one of the 73 move types

def legal_moves_to_array(board):
    """Create a [8, 8, 73] boolean array of legal moves."""
    array = np.zeros((8, 8, 73), dtype=bool)
    for move in board.legal_moves:
        action = move_to_action(board, move)
        if action:
            array[action] = True
    return array

def uci_to_action(board, uci):
    """Convert UCI string to flattened action number (0–4671) or None if invalid."""
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        return None
    action = move_to_action(board, move)
    if action:
        from_file, from_rank, move_type = action
        return from_file + 8 * from_rank + 64 * move_type
    return None

def action_to_uci(board, action):
    """Convert flattened action number to UCI string or None if invalid."""
    if not 0 <= action < 4672:
        return None
    move_type = action // 64
    temp = action % 64
    rank = temp // 8
    file = temp % 8
    from_square = chess.square(file, rank)

    if 0 <= move_type <= 55:  # Queen moves
        dir_idx = move_type // 7
        step = (move_type % 7) + 1
        df, dr = DIRECTIONS[dir_idx]
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
    return file + 8 * rank + 64 * move_type

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
