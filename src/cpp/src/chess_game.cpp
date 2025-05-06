#include "chess_game.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <iostream>

namespace az73 {
namespace {
    static constexpr std::array<std::pair<int, int>, 8> DIRECTIONS = {{{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}}};
    static constexpr std::array<std::pair<int, int>, 8> KNIGHT_DELTAS = {{{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}};
    static constexpr std::array<int, 3> UNDERPROMO_DF{{-1, 0, 1}};
}

std::uint16_t encode(const chess::Board &board, const chess::Move &mv) {
    using namespace chess;
    const Color toPlay = board.sideToMove();
    const Square trueFrom = mv.from();
    const Square trueTo = mv.to();
    const Square fromSq = trueFrom.relative_square(toPlay);
    const Square toSq = trueTo.relative_square(toPlay);
    const int file = fromSq.file();
    const int rank = fromSq.rank();
    const int df = toSq.file() - file;
    const int dr = toSq.rank() - rank;
    if (mv.typeOf() == chess::Move::PROMOTION && mv.promotionType() != chess::PieceType::QUEEN) {
        int p = static_cast<int>(mv.promotionType());
        int piece_idx = p - static_cast<int>(chess::PieceType::KNIGHT);
        if (abs(df) <= 1 && dr == 1) {
            int dir = (df == -1 ? 0 : df == 0 ? 1 : 2);
            int moveType = 64 + piece_idx * 3 + dir;
            return file * 8 * 73 + rank * 73 + moveType;
        }
        return UINT16_MAX;
    }
    if (board.at(trueFrom).type() == chess::PieceType::KNIGHT) {
        auto it = std::find(KNIGHT_DELTAS.begin(), KNIGHT_DELTAS.end(), std::make_pair(df, dr));
        if (it != KNIGHT_DELTAS.end()) {
            int idx = static_cast<int>(std::distance(KNIGHT_DELTAS.begin(), it));
            int moveType = 56 + idx;
            return file * 8 * 73 + rank * 73 + moveType;
        }
        return UINT16_MAX;
    }
    for (int d = 0; d < 8; ++d) {
        auto [dx, dy] = DIRECTIONS[d];
        if (dx == 0 && df == 0 && dr * dy > 0 && abs(dr) <= 7) {
            int k = abs(dr);
            int moveType = d * 7 + (k - 1);
            return file * 8 * 73 + rank * 73 + moveType;
        } else if (dy == 0 && dr == 0 && df * dx > 0 && abs(df) <= 7) {
            int k = abs(df);
            int moveType = d * 7 + (k - 1);
            return file * 8 * 73 + rank * 73 + moveType;
        } else if (dx != 0 && dy != 0 && df * dx > 0 && dr * dy > 0 && abs(df) == abs(dr) && abs(df) <= 7) {
            int k = abs(df);
            int moveType = d * 7 + (k - 1);
            return file * 8 * 73 + rank * 73 + moveType;
        }
    }
    return UINT16_MAX;
}

chess::Move decode_action(std::uint16_t action, const chess::Board &board) {
    constexpr int TOTAL = 8 * 8 * 73;
    assert(action < TOTAL);
    int mt = action % 73;
    int tmp = action / 73;
    int file = tmp / 8;
    int rank = tmp % 8;
    if (board.sideToMove() == chess::Color::BLACK)
        rank = 7 - rank;
    chess::Square from(chess::File(static_cast<chess::File::underlying>(file)), chess::Rank(static_cast<chess::Rank::underlying>(rank)));
    chess::Square to;
    if (mt < 56) {
        int dir = mt / 7;
        int step = (mt % 7) + 1;
        auto [df, dr] = az73::DIRECTIONS[dir];
        if (board.sideToMove() == chess::Color::BLACK)
            dr = -dr;
        int tf = file + df * step;
        int tr = rank + dr * step;
        to = chess::Square(chess::File(static_cast<chess::File::underlying>(tf)), chess::Rank(static_cast<chess::Rank::underlying>(tr)));
    } else if (mt < 64) {
        int idx = mt - 56;
        auto [df, dr] = az73::KNIGHT_DELTAS[idx];
        if (board.sideToMove() == chess::Color::BLACK)
            dr = -dr;
        int tf = file + df;
        int tr = rank + dr;
        to = chess::Square(chess::File(static_cast<chess::File::underlying>(tf)), chess::Rank(static_cast<chess::Rank::underlying>(tr)));
    } else {
        int promo_idx = (mt - 64) / 3;
        int dir_idx = (mt - 64) % 3;
        int df = az73::UNDERPROMO_DF[dir_idx];
        int dr = (board.sideToMove() == chess::Color::WHITE ? 1 : -1);
        int tf = file + df;
        int tr = (board.sideToMove() == chess::Color::WHITE ? 7 : 0);
        to = chess::Square(chess::File(static_cast<chess::File::underlying>(tf)), chess::Rank(static_cast<chess::Rank::underlying>(tr)));
    }
    const chess::Piece fromPiece = board.at(from);
    const chess::Piece toPiece = to.is_valid() ? board.at(to) : chess::Piece::NONE;
    if (fromPiece.type() == chess::PieceType::KING && toPiece != chess::Piece::NONE && toPiece.color() == fromPiece.color() && toPiece.type() == chess::PieceType::ROOK) {
        return chess::Move::make<chess::Move::CASTLING>(from, to);
    }
    if (fromPiece.type() == chess::PieceType::PAWN && to == board.enpassantSq()) {
        return chess::Move::make<chess::Move::ENPASSANT>(from, to);
    }
    if (fromPiece.type() == chess::PieceType::PAWN) {
        if (to.rank() == (board.sideToMove() == chess::Color::WHITE ? chess::Rank::RANK_8 : chess::Rank::RANK_1)) {
            return chess::Move::make<chess::Move::PROMOTION>(from, to, chess::PieceType::QUEEN);
        }
    }
    if (mt >= 64) {
        int promo_idx = (mt - 64) / 3;
        chess::PieceType pt = promo_idx == 0 ? chess::PieceType::KNIGHT : promo_idx == 1 ? chess::PieceType::BISHOP : chess::PieceType::ROOK;
        return chess::Move::make<chess::Move::PROMOTION>(from, to, pt);
    }
    if (fromPiece.type() == chess::PieceType::KNIGHT || mt >= 56) {
        return chess::Move::make<chess::Move::NORMAL>(from, to);
    }
    return chess::Move::make<chess::Move::NORMAL>(from, to);
}

} // namespace az73

namespace {
    using chess::Bitboard;
    using chess::Color;
    using chess::PieceType;
    using chess::Square;
    constexpr int flat(int plane, int r, int f) { return plane * 64 + r * 8 + f; }
    inline void fill(float *base, int plane, float v) noexcept { std::fill(base + plane * 64, base + (plane + 1) * 64, v); }
    inline std::pair<int, int> orient(Square sq, Color view) {
        int r = static_cast<int>(sq.rank());
        int f = static_cast<int>(sq.file());
        if (view == Color::BLACK) {
            r = 7 - r;
            f = 7 - f;
        }
        return {r, f};
    }
}

ChessGame::ChessGame(std::string_view fen) {
    history_.emplace_back(std::string{fen});
    hashes_.push_back(history_.back().hash());
}

void ChessGame::makeMove(const chess::Move &m) {
    chess::Board b = currentBoard();
    b.makeMove(m);
    history_.push_back(b);
    hashes_.push_back(b.hash());
}

void ChessGame::makeMove(const std::uint16_t a) {
    chess::Move m = az73::decode_action(a, currentBoard());
    makeMove(m);
}

ChessGame::Tensor ChessGame::encodeTensor() const {
    constexpr int PLANES_PER_STEP = 14;
    Tensor x{};
    const chess::Board &cur = history_.back();
    const Color P1 = cur.sideToMove();
    const Color P2 = (P1 == Color::WHITE ? Color::BLACK : Color::WHITE);
    for (int t = 0; t < 8; ++t) {
        int hIdx = static_cast<int>(history_.size()) - 1 - t;
        if (hIdx < 0) break;
        const chess::Board &b = history_[hIdx];
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

std::vector<std::uint16_t> ChessGame::legalMoves() const {
    using namespace chess;
    Movelist movelist;
    movegen::legalmoves<movegen::MoveGenType::ALL>(movelist, currentBoard(), PieceGenType::PAWN | PieceGenType::KNIGHT | PieceGenType::BISHOP | PieceGenType::ROOK | PieceGenType::QUEEN | PieceGenType::KING);
    std::vector<std::uint16_t> actions;
    actions.reserve(static_cast<size_t>(movelist.size()));
    for (const auto &mv : movelist) {
        std::uint16_t a = az73::encode(currentBoard(), mv);
        if (a < 4672) actions.push_back(a);
    }
    return actions;
}

std::optional<int> ChessGame::winner() const {
    const auto &b = currentBoard();
    using namespace chess;
    if (b.isHalfMoveDraw()) return -1;
    if (b.isRepetition()) return -1;
    Movelist movelist;
    movegen::legalmoves<movegen::MoveGenType::ALL>(movelist, b, PieceGenType::PAWN | PieceGenType::KNIGHT | PieceGenType::BISHOP | PieceGenType::ROOK | PieceGenType::QUEEN | PieceGenType::KING);
    if (movelist.empty()) {
        if (b.inCheck()) return (b.sideToMove() == Color::WHITE) ? 0 : 1;
        else return -1;
    }
    return std::nullopt;
}

int ChessGame::to_play() const noexcept {
    return (currentBoard().sideToMove() == chess::Color::WHITE) ? 1 : 0;
}
