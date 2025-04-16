#ifndef CHESS_GAME_H
#define CHESS_GAME_H

#include <vector>
#include <optional>
#include <memory>
#include <chess.hpp>

using namespace chess;

class ChessGame {
public:
    static const int ACTIONS = 4672; // 8x8x73
    static const int N = 8;         // 8x8 board
    static const int C = 119;       // Tensor channels

    ChessGame();
    std::unique_ptr<ChessGame> clone(bool swap_players = false) const;
    std::vector<float> get_tensor() const; // Flattened 8*8*119
    int to_play() const; // 0 (black), 1 (white)
    std::optional<int> winner() const; // 0 (black), 1 (white), -1 (draw), nullopt
    std::vector<int> legal_actions() const;
    void make_move(int action);
    void undo_move();
    bool is_terminal() const;

private:
    std::unique_ptr<Board> board; // chess-library Board
    std::vector<std::unique_ptr<Board>> history; // For tensor history
};

#endif
