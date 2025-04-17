#include "chess_game.hpp"

namespace {

using chess::Color;
using chess::PieceType;
using chess::Square;
using chess::Bitboard;

/* plane * 64 + 8 * rank + file  (row‑major, “channels‑last”) */
constexpr int flat(int plane, int r, int f) { return plane * 64 + r * 8 + f; }

/* fill an 8×8 plane with a constant */
inline void fill(float* base, int plane, float v) noexcept {
    std::fill(base + plane * 64, base + (plane + 1) * 64, v);
}

/* square → (rank, file) with *view*’s back‑rank = row‑0 */
inline std::pair<int, int> orient(Square sq, Color view) {
    int r = static_cast<int>(sq.rank());
    int f = static_cast<int>(sq.file());
    if (view == Color::BLACK) {
        r = 7 - r;
        f = 7 - f;
    }
    return {r, f};
}

} // namespace

/* ---------------------------------------------------- */
/* ***  ChessGame implementation                     *** */
/* ---------------------------------------------------- */

ChessGame::ChessGame(std::string_view fen) {
    history_.emplace_back(std::string{fen});
    hashes_.push_back(history_.back().hash());
}

/* push one ply and remember the board + hash */
void ChessGame::makeMove(const chess::Move& m) {
    chess::Board b = history_.back();
    b.makeMove(m);
    history_.push_back(b);
    hashes_.push_back(b.hash());
}

/* ---------------------------------------------------- */
/* *********  AlphaZero 8×8×119 encoder  *************** */
/* ---------------------------------------------------- */
ChessGame::Tensor ChessGame::encodeTensor() const {
    constexpr int PLANES_PER_STEP = 14; // 6 P1 + 6 P2 + 2 repetition
    ChessGame::Tensor x{};

    const chess::Board& cur = history_.back();
    const Color P1 = cur.sideToMove(); // player to move *now*
    const Color P2 = (P1 == Color::WHITE ? Color::BLACK : Color::WHITE);

    // dynamic planes : current + 7 previous plies
    for (int t = 0; t < 8; ++t) {
        int hIdx = static_cast<int>(history_.size()) - 1 - t;
        if (hIdx < 0) break;

        const chess::Board& b = history_[hIdx];
        const bool flip = (b.sideToMove() != P1);
        const Color view = flip ? P2 : P1;
        const int base = t * PLANES_PER_STEP;

        auto addPieces = [&](Color c, int planeBase) {
            using PT = PieceType::underlying;
            static constexpr PT order[6] = {PT::PAWN, PT::KNIGHT, PT::BISHOP, PT::ROOK, PT::QUEEN, PT::KING};

            for (int i = 0; i < 6; ++i) {
                Bitboard bb = b.pieces(PieceType{order[i]}, c);
                while (bb) {
                    int idx64 = bb.pop(); // 0–63
                    Square sq = Square(idx64);
                    auto [r, f] = orient(sq, view);
                    x[flat(planeBase + i, r, f)] = 1.0f;
                }
            }
        };

        addPieces(view, base);
        addPieces(view == P1 ? P2 : P1, base + 6);

        std::uint64_t h = hashes_[hIdx];
        int repeats = std::count(hashes_.begin(), hashes_.begin() + hIdx, h);
        fill(x.data(), base + 12, repeats >= 1 ? 1.0f : 0.0f);
        fill(x.data(), base + 13, repeats >= 2 ? 1.0f : 0.0f);
    }

    // constant planes
    constexpr int CONST0 = 8 * PLANES_PER_STEP; // 112

    fill(x.data(), CONST0 + 0, (P1 == Color::WHITE) ? 1.0f : 0.0f);
    fill(x.data(), CONST0 + 1, static_cast<float>(cur.fullMoveNumber()));

    bool p1Ks = false, p1Qs = false, p2Ks = false, p2Qs = false;
    for (char c : cur.getCastleString()) {
        switch (c) {
            case 'K': (P1 == Color::WHITE ? p1Ks : p2Ks) = true; break;
            case 'Q': (P1 == Color::WHITE ? p1Qs : p2Qs) = true; break;
            case 'k': (P1 == Color::BLACK ? p1Ks : p2Ks) = true; break;
            case 'q': (P1 == Color::BLACK ? p1Qs : p2Qs) = true; break;
            default: break;
        }
    }

    fill(x.data(), CONST0 + 2, p1Ks ? 1.0f : 0.0f);
    fill(x.data(), CONST0 + 3, p1Qs ? 1.0f : 0.0f);
    fill(x.data(), CONST0 + 4, p2Ks ? 1.0f : 0.0f);
    fill(x.data(), CONST0 + 5, p2Qs ? 1.0f : 0.0f);
    fill(x.data(), CONST0 + 6, static_cast<float>(cur.halfMoveClock()));

    return x;
}
