#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "chess.hpp" // Disservin single-header library

// 8×8×73 AlphaZero policy-head encoding (4,672 action slots)
namespace az73 {

    namespace detail {
        // Constant offset tables (defined once in .cpp)
        extern const int  DIR[8];      // N, NE, E, SE, S, SW, W, NW (square-index deltas)
        extern const int  KNIGHT[8];   // Clockwise: UUR, RRU, …
        extern const char PROMO[3];    // Under-promotion piece order: n, b, r

        // Helpers working on a raw square index (0–63)
        [[nodiscard]] constexpr int file(int idx) noexcept { return idx & 7; }
        [[nodiscard]] constexpr int rank(int idx) noexcept { return idx >> 3; }
        [[nodiscard]] constexpr bool on_board(int f, int r) noexcept {
            return unsigned(f) < 8u && unsigned(r) < 8u;
        }
    }

    // Mapping helpers – definitions in chess_game.cpp
    [[nodiscard]] std::uint32_t encode(const chess::Move& mv);
    [[nodiscard]] std::string decode(std::uint32_t action);
    [[nodiscard]] std::uint32_t from_uci(std::string_view u);

} // namespace az73

class ChessGame {
    public:
        using Tensor = std::array<float, 8 * 8 * 119>;

        explicit ChessGame(std::string_view fen = chess::constants::STARTPOS);

        [[nodiscard]] const chess::Board& currentBoard() const noexcept {
            return history_.back();
        }

        void makeMove(const chess::Move& m);
        [[nodiscard]] Tensor encodeTensor() const;
        [[nodiscard]] std::vector<std::uint32_t> legalMoves() const;

        // Handy bridges
        [[nodiscard]] std::string actionToUci(std::uint32_t a) const;
        [[nodiscard]] std::uint32_t uciToAction(std::string_view u) const;

    private:
        std::vector<chess::Board> history_;
        std::vector<std::uint64_t> hashes_;
};
