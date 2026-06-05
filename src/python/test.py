import os
import torch
import chess, numpy as np, chess_moves
import chess_engine


board = chess.Board()
board.push_uci("a2a4")  # e4
fen = chess.STARTING_FEN          # any FEN is fine
print("FEN:", fen)
py_t = chess_moves.board_to_tensor(board)   # [8,8,119] H×W×C
print("Python tensor shape:", py_t.shape)
print("Python tensor sum:", py_t.sum())
print(py_t[0,0,:] )
cpp_t = chess_engine.cpp_encode_tensor(fen)            # current C++ output
print("C++ tensor shape:", cpp_t.shape)
print("C++ tensor sum:", cpp_t.sum())
print(cpp_t[0,0,:])

diff = np.abs(py_t - cpp_t).max()
print("max |Python − C++|  =", diff)
























##import struct, numpy as np
##import torch
##
##HEADER = 24
##STATE  = 8*8*119 * 4          # 304, 64 bytes
##REC    = STATE + 4672*4 + 4   # full record size
##
##with open("games.bin", "rb") as f:
##    f.seek(HEADER)            # position of very first record
##    raw = f.read(STATE)       # read *only* the state part
##state_cpp = np.frombuffer(raw, np.float32)
##
##
##
##wrong = state_cpp.reshape(8, 8, 119)                 # what you had
##fixed = state_cpp.reshape(119, 8, 8).transpose(1,2,0)  # correct layout
##
##def piece_counts(t):
##    # first 12 planes are pieces: 0-5 = P1, 6-11 = P2
##    return [t[:,:,p].sum() for p in range(12)]
##
##print("wrong :", piece_counts(wrong))
##print("fixed :", piece_counts(fixed))
##
##
##import chess, chess.svg
##
##def tensor_to_board(t):
##    b = chess.Board.empty()
##    # only P1 planes here; they already encode from the side-to-move view
##    piece_map = {
##        0: chess.PAWN,   1: chess.KNIGHT, 2: chess.BISHOP,
##        3: chess.ROOK,   4: chess.QUEEN,  5: chess.KING
##    }
##    for plane, ptype in piece_map.items():
##        for r in range(8):
##            for f in range(8):
##                if t[r,f,plane]:
##                    sq = chess.square(f, r)
##                    b.set_piece_at(sq, chess.Piece(ptype, chess.WHITE))
##    for plane, ptype in piece_map.items():
##        for r in range(8):
##            for f in range(8):
##                if t[r,f,6+plane]:
##                    sq = chess.square(f, r)
##                    b.set_piece_at(sq, chess.Piece(ptype, chess.BLACK))
##    return b
##
##print(tensor_to_board(fixed))        # should look normal
##ok = all(count <= 8 for count in piece_counts(fixed))
##print("All piece counts <= 8:", ok)


