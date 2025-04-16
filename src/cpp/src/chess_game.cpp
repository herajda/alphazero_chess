#include "chess_game.h"
#include <algorithm>
#include <stdexcept>

using U64 = std::uint64_t;

// Clone the game state, optionally swapping players
ChessGame ChessGame::clone() const {
    ChessGame clone;
    return clone;
}

chess::Board ChessGame::current_board() const {
    if () {
        throw std::runtime_error("No board history available.");
    }
    return _boards.back();

}
chess::Board ChessGame::current_board() {
    return board_history.get_latest_board();
}

// Return the current player (0 for White, 1 for Black)
int ChessGame::to_play() const {
    return static_cast<int>(_board.sideToMove());
}

// Check if an action is valid
bool ChessGame::valid(int action) const {
    chess::Move move = action_to_move(action);
    if (move == chess::Move::NO_MOVE) return false;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, _board);
    return std::find(moves.begin(), moves.end(), move) != moves.end();
}

// Return a list of valid action indices
std::vector<int> ChessGame::valid_actions() const {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, _board);
    std::vector<int> actions;
    actions.reserve(moves.size());
    for (const auto& move : moves) {
        int action = move_to_action(move);
        if (action != -1) {
            actions.push_back(action);
        }
    }
    return actions;
}

// Execute a move based on the action index
void ChessGame::move(int action) {
    chess::Move move = action_to_move(action);
    if (move == chess::Move::NO_MOVE || !valid(action)) {
        throw std::invalid_argument("Invalid action: " + std::to_string(action));
    }
    _board.makeMove(move);
}

// Determine the winner: -2 (not over), -1 (draw), 0 (White), 1 (Black)
int ChessGame::winner() const {
    auto [reason, result] = _board.isGameOver();
    if (result == chess::GameResult::WIN) {
        return static_cast<int>(_board.sideToMove());
    } else if (result == chess::GameResult::LOSE) {
        return static_cast<int>(~_board.sideToMove());
    } else if (result == chess::GameResult::DRAW) {
        return -1;
    } else {
        return -2;
    }
}

// Convert a Move object to an action index
int ChessGame::move_to_action(const chess::Move& move) const {
    chess::Square from = move.from();
    chess::Square to = move.to();
    int from_file = from.file();
    int from_rank = from.rank();
    int delta_file = to.file() - from_file;
    int delta_rank = to.rank() - from_rank;

    // Underpromotions
    if (move.typeOf() == chess::Move::PROMOTION && move.promotionType() != chess::PieceType::QUEEN) {
        int expected_rank = (_board.sideToMove() == chess::Color::WHITE) ? 7 : 0;
        if (to.rank() == expected_rank && std::abs(delta_file) <= 1) {
            int direction = (delta_file == -1) ? 0 : (delta_file == 0) ? 1 : 2;
            int piece_idx = static_cast<int>(move.promotionType()) - 2;
            int move_type_index = 64 + 3 * piece_idx + direction;
            return from_file + 8 * from_rank + 64 * move_type_index;
        }
        return -1;
    }

    // Knight moves
    chess::Piece piece = _board.at<chess::Piece>(from);
    if (piece.type() == chess::PieceType::KNIGHT) {
        for (int k = 0; k < 8; ++k) {
            if (delta_file == KNIGHT_DELTAS[k].first && delta_rank == KNIGHT_DELTAS[k].second) {
                int move_type_index = 56 + k;
                return from_file + 8 * from_rank + 64 * move_type_index;
            }
        }
        return -1;
    }

    // Sliding moves
    for (int dir_idx = 0; dir_idx < 8; ++dir_idx) {
        int df = DIRECTIONS[dir_idx].first;
        int dr = DIRECTIONS[dir_idx].second;
        if ((df == 0 && delta_file == 0 && dr * delta_rank > 0) ||
            (dr == 0 && delta_rank == 0 && df * delta_file > 0) ||
            (df != 0 && dr != 0 && delta_file * dr == delta_rank * df && delta_file * df > 0)) {
            int steps = (df == 0) ? std::abs(delta_rank) : std::abs(delta_file);
            if (1 <= steps && steps <= 7) {
                int move_type_index = dir_idx * 7 + (steps - 1);
                return from_file + 8 * from_rank + 64 * move_type_index;
            }
        }
    }
    return -1;
}

// Convert an action index to a Move object
chess::Move ChessGame::action_to_move(int action) const {
    if (action < 0 || action >= ACTIONS) return chess::Move::NO_MOVE;
    int move_type = action / 64;
    int temp = action % 64;
    int rank = temp / 8;
    int file = temp % 8;
    chess::Square from = chess::Square(file, rank);

    if (move_type <= 55) { // Sliding moves
        int dir_idx = move_type / 7;
        int step = (move_type % 7) + 1;
        int df = DIRECTIONS[dir_idx].first;
        int dr = DIRECTIONS[dir_idx].second;
        int to_file = file + df * step;
        int to_rank = rank + dr * step;
        if (to_file >= 0 && to_file < 8 && to_rank >= 0 && to_rank < 8) {
            chess::Square to = chess::Square(to_file, to_rank);
            chess::Piece piece = _board.at<chess::Piece>(from);
            if (piece.type() == chess::PieceType::PAWN && to.rank() == (_board.sideToMove() == chess::Color::WHITE ? 7 : 0)) {
                return chess::Move::make<chess::Move::PROMOTION>(from, to, chess::PieceType::QUEEN);
            }
            return chess::Move::make<chess::Move::NORMAL>(from, to);
        }
    } else if (move_type <= 63) { // Knight moves
        int knight_idx = move_type - 56;
        int df = KNIGHT_DELTAS[knight_idx].first;
        int dr = KNIGHT_DELTAS[knight_idx].second;
        int to_file = file + df;
        int to_rank = rank + dr;
        if (to_file >= 0 && to_file < 8 && to_rank >= 0 && to_rank < 8) {
            chess::Square to = chess::Square(to_file, to_rank);
            return chess::Move::make<chess::Move::NORMAL>(from, to);
        }
    } else if (move_type <= 72) { // Underpromotions
        int piece_idx = (move_type - 64) / 3;
        int dir_idx = (move_type - 64) % 3;
        chess::PieceType promotion = static_cast<chess::PieceType>(chess::PieceType::KNIGHT + piece_idx);
        int delta_file = (dir_idx == 0) ? -1 : (dir_idx == 1) ? 0 : 1;
        int delta_rank = (_board.sideToMove() == chess::Color::WHITE) ? 1 : -1;
        int to_file = file + delta_file;
        int to_rank = rank + delta_rank;
        int promotion_rank = (_board.sideToMove() == chess::Color::WHITE) ? 7 : 0;
        if (to_file >= 0 && to_file < 8 && to_rank == promotion_rank) {
            chess::Square to = chess::Square(to_file, to_rank);
            return chess::Move::make<chess::Move::PROMOTION>(from, to, promotion);
        }
    }
    return chess::Move::NO_MOVE;
}

// Fill piece planes for a board state
void ChessGame::fill_tensor_for_board(const chess::Board& board, chess::Color p1,
                                      std::vector<std::vector<std::vector<float>>>& tensor,
                                      int channel_offset) const {
    chess::Color p2 = ~p1;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            chess::Square sq = (p1 == chess::Color::WHITE) ? chess::Square(j, i) : chess::Square(j, 7 - i);
            chess::Piece piece = board.at<chess::Piece>(sq);
            if (piece != chess::Piece::NONE) {
                chess::Color color = piece.color();
                int plane = (color == p1) ? static_cast<int>(piece.type()) - 1 : 6 + static_cast<int>(piece.type()) - 1;
                tensor[i][j][channel_offset + plane] = 1.0f;
            }
        }
    }
}

// Fill constant planes
void ChessGame::fill_constant_planes(std::vector<std::vector<std::vector<float>>>& tensor, chess::Color p1) const {
    float color_value = (p1 == chess::Color::WHITE) ? 1.0f : 0.0f;
    int total_moves = _board.moveStack().size();
    float move_count_value = total_moves / 1000.0f;
    chess::CastlingRights cr = _board.castlingRights();
    float p1_kingside = cr.hasKingside(p1) ? 1.0f : 0.0f;
    float p1_queenside = cr.hasQueenside(p1) ? 1.0f : 0.0f;
    float p2_kingside = cr.hasKingside(~p1) ? 1.0f : 0.0f;
    float p2_queenside = cr.hasQueenside(~p1) ? 1.0f : 0.0f;
    float halfmove_value = _board.halfMoveClock() / 50.0f;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            tensor[i][j][112] = color_value;
            tensor[i][j][113] = move_count_value;
            tensor[i][j][114] = p1_kingside;
            tensor[i][j][115] = p1_queenside;
            tensor[i][j][116] = p2_kingside;
            tensor[i][j][117] = p2_queenside;
            tensor[i][j][118] = halfmove_value;
        }
    }
}
