#include "chess_game.hpp"
#include <algorithm>
#include <memory>
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
    if (mt >= 64) {
        int promo_idx = (mt - 64) / 3;
        chess::PieceType pt = promo_idx == 0 ? chess::PieceType::KNIGHT : promo_idx == 1 ? chess::PieceType::BISHOP : chess::PieceType::ROOK;
        return chess::Move::make<chess::Move::PROMOTION>(from, to, pt);
    }
    if (fromPiece.type() == chess::PieceType::PAWN) {
        if (to.rank() == (board.sideToMove() == chess::Color::WHITE ? chess::Rank::RANK_8 : chess::Rank::RANK_1)) {
            return chess::Move::make<chess::Move::PROMOTION>(from, to, chess::PieceType::QUEEN);
        }
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
    //constexpr int flat(int plane, int r, int f) { return plane * 64 + r * 8 + f; }
    //constexpr int flat(int plane,int r,int f){ return r*8*119 + f*119 + plane; }
    constexpr int flat(int plane, int r, int f)
    {   return r * 8 * 119 + f * 119 + plane;   }


    inline void fill(float *base, int plane, float v) noexcept
    {   std::fill(base + plane * 64, base + (plane + 1) * 64, v); }

    inline std::pair<int,int> orient(Square sq, Color p1)
    {
        int r = int(sq.rank());
        int f = int(sq.file());
        if (p1 == Color::BLACK)   r = 7 - r;          // vertical flip only
        return {r, f};
    }
    inline void fillPlane(ChessGame::Tensor &x, int plane, float v) noexcept
    {
    for (int r = 0; r < 8; ++r)
        for (int f = 0; f < 8; ++f)
            x[flat(plane, r, f)] = v;
    }
}

ChessGame::ChessGame(std::string_view fen) {
    chess::Board b {std::string{fen}};
    history_.push_back(chess::Board::Compact::encode(b));
    hashes_.push_back(b.hash());
    half_move_clock_.push_back(b.halfMoveClock());
    full_move_number_.push_back(b.fullMoveNumber());
}

void ChessGame::makeMove(const chess::Move &m) {
    chess::Board b = currentBoard();
    b.makeMove(m);
    history_.push_back(chess::Board::Compact::encode(b));
    hashes_.push_back(b.hash());
    half_move_clock_.push_back(b.halfMoveClock());
    full_move_number_.push_back(b.fullMoveNumber());
}

void ChessGame::makeMove(const std::uint16_t a) {
    chess::Move m = az73::decode_action(a, currentBoard());
    makeMove(m);
}
ChessGame::Tensor ChessGame::encodeTensor() const
{
    constexpr int PLANES_PER_STEP = 14;          // 8 steps × 14 planes  = 112
    constexpr int CONST_BASE      = 8 * PLANES_PER_STEP;   // 112 … 118

    Tensor x{};                                  // all zeros by default

    // ── current players ──────────────────────────────────────────────────────
    const chess::Board  cur = currentBoard();
    const chess::Color  P1  = cur.sideToMove();          // player to move now
    const chess::Color  P2  = (P1 == chess::Color::WHITE ? chess::Color::BLACK
                                                         : chess::Color::WHITE);

    // ── which 8 positions do we need?  (oldest → newest) ─────────────────────
    const int total   = int(history_.size());            // inc. initial board
    const int pad     = total < 8 ? 8 - total : 0;       // “None” slots in front
    const int start   = total > 8 ? total - 8 : 0;       // drop older boards

    // ── to count repetitions exactly the way Python does it ──────────────────
    std::unordered_map<std::uint64_t,int> seen_hashes;

    auto addPieces = [&](const chess::Board& b,
                         chess::Color colour,
                         int           planeBase)
    {
        using PTu = chess::PieceType::underlying;
        static constexpr PTu ORDER[6] = { PTu::PAWN, PTu::KNIGHT, PTu::BISHOP,
                                          PTu::ROOK, PTu::QUEEN, PTu::KING };

        for (int i = 0; i < 6; ++i)
        {
            Bitboard bb = b.pieces(chess::PieceType{ORDER[i]}, colour);
            while (bb)
            {
                int idx64 = bb.pop();
                chess::Square sq(idx64);
                auto [r,f] = orient(sq, P1);            // always from P1’s view
                x[ flat(planeBase + i, r, f) ] = 1.f;
            }
        }
    };

    // ── 8 historical time-steps (t = 0 = oldest … 7 = current) ───────────────
    for (int t = 0; t < 8; ++t)
    {
        if (t < pad)                       // before the game even started
            continue;                      // → planes stay zero

        const int hIdx = start + (t - pad);
        const chess::Board b = boardAt(hIdx);

        const int base = t * PLANES_PER_STEP;

        // 0-5 : P1 pieces | 6-11 : P2 pieces
        addPieces(b, P1, base);
        addPieces(b, P2, base + 6);

        // repetition plane 12  (count ≥ 2)
        int cnt = ++seen_hashes[ hashes_[hIdx] ];
        if (cnt >= 2)
            fillPlane(x, base + 12, 1.f);

        // plane 13 is left at 0 – exactly what the Python code does
    }

    // ── constant planes 112 … 118 ────────────────────────────────────────────
    // ─ historical repetition plane ─

// ─ constant planes 112-118 ─
    fillPlane(x, CONST_BASE + 0, P1 == chess::Color::WHITE ? 1.f : 0.f);  // 112
    fillPlane(x, CONST_BASE + 1, float(history_.size() - 1));    // 113
    using Side = chess::Board::CastlingRights::Side; 
    const auto cr = cur.castlingRights();                                       // NEW
    fillPlane(x, CONST_BASE + 2, cr.has(P1, Side::KING_SIDE)  ? 1.f : 0.f); // 114
    fillPlane(x, CONST_BASE + 3, cr.has(P1, Side::QUEEN_SIDE) ? 1.f : 0.f); // 115
    fillPlane(x, CONST_BASE + 4, cr.has(P2, Side::KING_SIDE)  ? 1.f : 0.f); // 116
    fillPlane(x, CONST_BASE + 5, cr.has(P2, Side::QUEEN_SIDE) ? 1.f : 0.f); // 117

    fillPlane(x, CONST_BASE + 6, float(cur.halfMoveClock()) / 50.f);        // 118


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

chess::Board ChessGame::boardAt(std::size_t idx) const
{
    chess::Board pos = chess::Board::Compact::decode(history_[idx]);

    std::string fen = pos.getFen(/*move_counters = */ false);

    fen += ' ';
    fen += std::to_string(half_move_clock_[idx]);   // 50-move counter
    fen += ' ';
    fen += std::to_string(full_move_number_[idx]);  // full move counter 

    return chess::Board::fromFen(fen);
}

chess::Board ChessGame::currentBoard() const
{
    return boardAt(history_.size() - 1);
}


