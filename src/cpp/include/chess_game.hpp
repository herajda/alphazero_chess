#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "chess.hpp" // Disservin chess-library

class ChessGame {
public:
    // 8×8×119 tensor in NHWC ("channels-last") order – AlphaZero format
    using Tensor = std::array<float, 8 * 8 * 119>;

    [[nodiscard]] const chess::Board& currentBoard() const noexcept {
        return history_.back();
    }

    explicit ChessGame(std::string_view fen = chess::constants::STARTPOS);

    // Push a legal move and update all incremental data structures
    void makeMove(const chess::Move& m);

    // Encode history_.back() and the seven previous plies
    [[nodiscard]] Tensor encodeTensor() const;

private:
    // Board after every ply (history_[0] is the initial one)
    std::vector<chess::Board> history_;

    // Zobrist hash of every entry in history_
    std::vector<std::uint64_t> hashes_;
};
