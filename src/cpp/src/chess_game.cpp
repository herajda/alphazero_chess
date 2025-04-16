#include "chess_game.h"
#include <unordered_map>
#include <algorithm>

// Action mapping functions (ported from chess_moves.py)
int uci_to_action(const Board& board, const std::string& uci) {
    // Port uci_to_action from chess_moves.py
    // Input: UCI string (e.g., "e2e4")
    // Output: Action index (0–4671) or -1 if invalid
    // Implement logic for queen moves, knight moves, underpromotions
    // Example (simplified):
    Move move = Move::from_uci(uci);
    if (!board.is_legal(move)) return -1;
    int from_square = move.from_square();
    int to_square = move.to_square();
    int from_file = from_square % 8;
    int from_rank = from_square / 8;
    int delta_file = (to_square % 8) - from_file;
    int delta_rank = (to_square / 8) - from_rank;
    // Map to one of 73 move types (see chess_moves.py)
    // Return from_file + 8 * from_rank + 64 * move_type
    return -1; // Placeholder
}

std::string action_to_uci(const Board& board, int action) {
    // Port action_to_uci from chess_moves.py
    // Input: Action index (0–4671)
    // Output: UCI string (e.g., "e2e4") or empty if invalid
    if (action < 0 || action >= ACTIONS) return "";
    int move_type = action / 64;
    int temp = action % 64;
    int rank = temp / 8;
    int file = temp % 8;
    // Map to UCI based on move_type (queen, knight, underpromotion)
    return ""; // Placeholder
}

ChessGame::ChessGame() : board(std::make_unique<Board>()) {}

std::unique_ptr<ChessGame> ChessGame::clone(bool swap_players) const {
    auto clone = std::make_unique<ChessGame>();
    clone->board = std::make_unique<Board>(*board);
    if (swap_players) clone->board->set_turn(!board->turn());
    clone->history.resize(history.size());
    for (size_t i = 0; i < history.size(); ++i) {
        clone->history[i] = std::make_unique<Board>(*history[i]);
    }
    return clone;
}

std::vector<float> ChessGame::get_tensor() const {
    std::vector<float> tensor(N * N * C, 0.0f);
    bool p1 = board->turn();
    bool p2 = !p1;

    // Get history (up to 8 positions)
    std::vector<std::string> fens;
    fens.push_back(board->fen());
    for (size_t i = 0; i < history.size() && i < 7; ++i) {
        fens.push_back(history[history.size() - 1 - i]->fen());
    }
    std::reverse(fens.begin(), fens.end());

    // Count repetitions
    std::unordered_map<std::string, int> fen_counts;
    std::vector<int> counts;
    for (const auto& fen : fens) {
        fen_counts[fen]++;
        counts.push_back(fen_counts[fen]);
    }

    // Fill piece planes
    for (size_t t = 0; t < std::min<size_t>(8, fens.size()); ++t) {
        Board& b = t == fens.size() - 1 ? *board : *history[t];
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                int square = p1 ? (j + i * 8) : (j + (7 - i) * 8);
                auto piece = b.piece_at(square);
                if (piece) {
                    bool color = piece.color();
                    int piece_type = piece.type() - 1; // 0=pawn, ..., 5=king
                    int plane = color == p1 ? piece_type : 6 + piece_type;
                    tensor[i * N * C + j * C + t * 14 + plane] = 1.0f;
                }
            }
        }
        if (counts[t] >= 2) {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    tensor[i * N * C + j * C + t * 14 + 12] = 1.0f;
        }
    }

    // Constant planes
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            tensor[i * N * C + j * C + 112] = p1 ? 1.0f : 0.0f;
            tensor[i * N * C + j * C + 113] = board->fullmove_number() / 1000.0f;
            tensor[i * N * C + j * C + 114] = board->has_kingside_castling_rights(p1) ? 1.0f : 0.0f;
            tensor[i * N * C + j * C + 115] = board->has_queenside_castling_rights(p1) ? 1.0f : 0.0f;
            tensor[i * N * C + j * C + 116] = board->has_kingside_castling_rights(p2) ? 1.0f : 0.0f;
            tensor[i * N * C + j * C + 117] = board->has_queenside_castling_rights(p2) ? 1.0f : 0.0f;
            tensor[i * N * C + j * C + 118] = board->halfmove_clock() / 50.0f;
        }
    }
    return tensor;
}

int ChessGame::to_play() const {
    return board->turn() ? 1 : 0;
}

std::optional<int> ChessGame::winner() const {
    auto result = board->result();
    if (result == "*") return std::nullopt;
    if (result == "1-0") return 1;
    if (result == "0-1") return 0;
    return -1;
}

std::vector<int> ChessGame::legal_actions() const {
    std::vector<int> actions;
    for (const auto& uci : board->legal_moves()) {
        int action = uci_to_action(*board, uci);
        if (action >= 0 && action < ACTIONS) actions.push_back(action);
    }
    return actions;
}

void ChessGame::make_move(int action) {
    std::string uci = action_to_uci(*board, action);
    if (!uci.empty()) {
        history.push_back(std::make_unique<Board>(*board));
        board->move(uci);
    }
}

void ChessGame::undo_move() {
    if (!history.empty()) {
        board = std::move(history.back());
        history.pop_back();
    }
}

bool ChessGame::is_terminal() const {
    return board->result() != "*";
}
