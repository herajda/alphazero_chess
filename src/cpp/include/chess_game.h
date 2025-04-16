#ifndef CHESSGAME_H
#define CHESSGAME_H

#include "chess.hpp"
#include "rolling_board_history.h"

#include <vector>
#include <array>
#include <deque>
#include <unordered_map>

class ChessGame {
public:
    // Constants matching the Python implementation
    static const int ACTIONS = 4672; // 8x8x73 possible moves
    static const int N = 8;          // Board size
    static const int C = 119;        // Channels in board tensor (8*14 historical + 7 constant)
    const size_t MAX_HISTORY_SIZE = 8;

    // Constructor: Initializes with the standard starting position
    explicit ChessGame(size_t history_size = 8, const std::string& initial_fen = chess::constants::STARTPOS);

    // Clone the game state, optionally swapping players
    ChessGame clone() const;

    chess::Board current_board() const;

    // TODO: think about if this should really be this data structure
    // Return the board as an 8x8x119 tensor
    std::vector<std::vector<std::vector<float>>> board() const;

    // Return the current player (0 for White, 1 for Black)
    int to_play() const;

    // Check if an action is valid
    bool valid(int action) const;

    // Return a list of valid action indices
    std::vector<int> valid_actions() const;

    // Execute a move based on the action index
    void move(int action);

    // Determine the winner: -2 (not over), -1 (draw), 0 (White), 1 (Black)
    int winner() const;

private:
    RollingBoardHistory board_history;

    // Knight move deltas (clockwise from (1, 2))
    static constexpr std::array<std::pair<int, int>, 8> KNIGHT_DELTAS = {
        {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}
    };

    // Directions for sliding moves (N, NE, E, SE, S, SW, W, NW)
    static constexpr std::array<std::pair<int, int>, 8> DIRECTIONS = {
        {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}}
    };

    // Convert a Move object to an action index (0–4671)
    int move_to_action(const chess::Move& move) const;

    // Convert an action index to a Move object
    chess::Move action_to_move(int action) const;

    // Fill piece planes for a board state, oriented to p1's perspective
    void fill_tensor_for_board(const chess::Board& board, chess::Color p1,
                               std::vector<std::vector<std::vector<float>>>& tensor,
                               int channel_offset) const;

    // Fill constant planes (112–118)
    void fill_constant_planes(std::vector<std::vector<std::vector<float>>>& tensor, chess::Color p1) const;
};

#endif // CHESSGAME_H
