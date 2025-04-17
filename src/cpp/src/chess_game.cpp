#include "chess_game.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric> // std::count

namespace az73::detail {
    const int  DIR[8]    = {+8, +9, +1, -7, -8, -9, -1, +7};
    const int  KNIGHT[8] = {+17, +10, -6, -15, -17, -10, +6, +15};
    const char PROMO[3]  = {'n', 'b', 'r'};
}

static inline int promo_index(chess::PieceType pt) noexcept {
    using PT = chess::PieceType;
    if (pt == PT::KNIGHT) return 0;
    if (pt == PT::BISHOP) return 1;
    if (pt == PT::ROOK)   return 2;
    return -1;
}

namespace az73 {

    std::uint32_t encode(const chess::Move& mv) {
        const int fromIdx = mv.from().index();
        const int toIdx = mv.to().index();
        const int df = detail::file(toIdx) - detail::file(fromIdx);
        const int dr = detail::rank(toIdx) - detail::rank(fromIdx);

        if ((std::abs(df) == 1 && std::abs(dr) == 2) || (std::abs(df) == 2 && std::abs(dr) == 1)) {
            const int offset = dr * 8 + df;
            for (int i = 0; i < 8; ++i)
                if (detail::KNIGHT[i] == offset)
                    return fromIdx * 73u + 56u + i;
            throw std::logic_error("encode: knight offset mismatch");
        }

        if (const int pidx = promo_index(mv.promotionType()); pidx != -1) {
            const bool white = dr > 0;
            int dir_idx = (df == 0) ? 1 : (df == -1 ? (white ? 0 : 2) : (white ? 2 : 0));
            return fromIdx * 73u + 64u + pidx * 3 + dir_idx;
        }

        if (df == 0 || dr == 0 || std::abs(df) == std::abs(dr)) {
            int dir = -1, dist = 0;
            if (df == 0) dir = dr > 0 ? 0 : 4, dist = std::abs(dr);
            else if (dr == 0) dir = df > 0 ? 2 : 6, dist = std::abs(df);
            else {
                if (df > 0 && dr > 0) dir = 1;
                else if (df > 0 && dr < 0) dir = 3;
                else if (df < 0 && dr < 0) dir = 5;
                else dir = 7;
                dist = std::abs(df);
            }
            if (dist == 0 || dist > 7) throw std::logic_error("encode: distance out of range");
            return fromIdx * 73u + dir * 7u + (dist - 1);
        }

        throw std::logic_error("encode: unsupported move pattern");
    }

    std::string decode(std::uint32_t action) {
        const int fromIdx = static_cast<int>(action / 73);
        const int slot = static_cast<int>(action % 73);
        const int ff = detail::file(fromIdx);
        const int fr = detail::rank(fromIdx);
        int tf = ff, tr = fr;
        char promo = 0;

        if (slot < 56) {
            const int dir = slot / 7;
            const int dist = (slot % 7) + 1;
            int dest = fromIdx + detail::DIR[dir] * dist;
            tf = detail::file(dest); tr = detail::rank(dest);
        } else if (slot < 64) {
            const int k = slot - 56;
            int dest = fromIdx + detail::KNIGHT[k];
            tf = detail::file(dest); tr = detail::rank(dest);
        } else {
            int local = slot - 64;
            int pidx = local / 3;
            int dir_idx = local % 3;
            promo = detail::PROMO[pidx];
            bool white = (fr == 6);
            int df = (dir_idx == 0 ? -1 : dir_idx == 2 ? 1 : 0) * (white ? 1 : -1);
            tf = ff + df;
            tr = fr + (white ? 1 : -1);
        }

        if (!detail::on_board(tf, tr)) throw std::logic_error("decode: off-board result");

        std::string uci;
        uci.reserve(5);
        uci.push_back('a' + ff); uci.push_back('1' + fr);
        uci.push_back('a' + tf); uci.push_back('1' + tr);
        if (promo) uci.push_back(promo);
        return uci;
    }

    std::uint32_t from_uci(std::string_view u) {
        if (u.size() != 4 && u.size() != 5)
            throw std::invalid_argument("UCI must be 4 or 5 chars");

        int ff = u[0] - 'a', fr = u[1] - '1';
        int tf = u[2] - 'a', tr = u[3] - '1';
        if (!detail::on_board(ff, fr) || !detail::on_board(tf, tr))
            throw std::invalid_argument("coords off board");

        int df = tf - ff, dr = tr - fr, fromIdx = fr * 8 + ff;

        if (u.size() == 5 && (u[4] == 'n' || u[4] == 'N' || u[4] == 'b' || u[4] == 'B' || u[4] == 'r' || u[4] == 'R')) {
            int pidx = (std::tolower(u[4]) == 'n') ? 0 : (std::tolower(u[4]) == 'b' ? 1 : 2);
            int dir_idx;
            if (dr == 1) dir_idx = (df == -1 ? 0 : df == 0 ? 1 : 2);
            else if (dr == -1) dir_idx = (df == 1 ? 0 : df == 0 ? 1 : 2);
            else throw std::invalid_argument("promo delta bad");
            return fromIdx * 73u + 64u + pidx * 3 + dir_idx;
        }

        if ((std::abs(df) == 1 && std::abs(dr) == 2) || (std::abs(df) == 2 && std::abs(dr) == 1)) {
            const int offset = dr * 8 + df;
            for (int i = 0; i < 8; ++i)
                if (detail::KNIGHT[i] == offset)
                    return fromIdx * 73u + 56u + i;
            throw std::logic_error("from_uci: knight offset mismatch");
        }

        const bool diag = std::abs(df) == std::abs(dr) && df != 0;
        const bool horiz = dr == 0 && df != 0;
        const bool vert = df == 0 && dr != 0;
        if (!diag && !horiz && !vert)
            throw std::invalid_argument("displacement not queen-like");

        int dir = -1, dist;
        if (vert) dir = dr > 0 ? 0 : 4, dist = std::abs(dr);
        else if (horiz) dir = df > 0 ? 2 : 6, dist = std::abs(df);
        else {
            if (df > 0 && dr > 0) dir = 1;
            else if (df > 0 && dr < 0) dir = 3;
            else if (df < 0 && dr < 0) dir = 5;
            else dir = 7;
            dist = std::abs(df);
        }

        return fromIdx * 73u + dir * 7u + (dist - 1);
    }

} // namespace az73

namespace {

    using chess::Color;
    using chess::PieceType;
    using chess::Square;
    using chess::Bitboard;

    constexpr int flat(int plane, int r, int f) { return plane * 64 + r * 8 + f; }

    inline void fill(float* base, int plane, float v) noexcept {
        std::fill(base + plane * 64, base + (plane + 1) * 64, v);
    }

    inline std::pair<int, int> orient(Square sq, Color view) {
        int r = static_cast<int>(sq.rank());
        int f = static_cast<int>(sq.file());
        if (view == Color::BLACK) { r = 7 - r; f = 7 - f; }
        return {r, f};
    }

} // anonymous namespace

ChessGame::ChessGame(std::string_view fen) {
    history_.emplace_back(std::string{fen});
    hashes_.push_back(history_.back().hash());
}

void ChessGame::makeMove(const chess::Move& m) {
    chess::Board b = history_.back();
    b.makeMove(m);
    history_.push_back(b);
    hashes_.push_back(b.hash());
}

ChessGame::Tensor ChessGame::encodeTensor() const {
    constexpr int PLANES_PER_STEP = 14;
    Tensor x{};

    const chess::Board& cur = history_.back();
    const Color P1 = cur.sideToMove();
    const Color P2 = (P1 == Color::WHITE ? Color::BLACK : Color::WHITE);

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
                    int idx64 = bb.pop();
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

    constexpr int CONST0 = 8 * PLANES_PER_STEP;

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

std::vector<std::uint32_t> ChessGame::legalMoves() const {
    chess::Movelist mvlist;
    chess::movegen::legalmoves(mvlist, currentBoard());

    std::vector<std::uint32_t> out;
    out.reserve(mvlist.size());
    for (const auto& mv : mvlist)
        out.push_back(az73::encode(mv));

    std::sort(out.begin(), out.end());
    return out;
}

std::string ChessGame::actionToUci(std::uint32_t a) const {
    return az73::decode(a);
}

std::uint32_t ChessGame::uciToAction(std::string_view u) const {
    return az73::from_uci(u);
}
