#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <optional>
#include <vector>

#include "chess.hpp" // Disservin single-header library

namespace az73 {
    [[nodiscard]] std::uint16_t encode(const chess::Board& board, const chess::Move& mv);
    [[nodiscard]] chess::Move decode_action(std::uint16_t action, const chess::Board& board);
} // namespace az73

class ChessGame {
    public:
        using Tensor = std::array<float, 8 * 8 * 119>;

        explicit ChessGame(std::string_view fen = chess::constants::STARTPOS);

        [[nodiscard]] const chess::Board& currentBoard() const noexcept { return history_.back(); }
        void makeMove(const chess::Move& m);
        void makeMove(const std::uint16_t m);
        [[nodiscard]] Tensor encodeTensor() const;
        [[nodiscard]] std::vector<std::uint16_t> legalMoves() const;

        [[nodiscard]] std::optional<int> winner() const;
        [[nodiscard]] int to_play() const noexcept;

    private:
        std::vector<chess::Board> history_;
        std::vector<std::uint64_t> hashes_;
};
