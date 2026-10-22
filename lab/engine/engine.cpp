// 非NN評価の将棋エンジン（研究用）
//
// 評価 = 固定の駒の価値 tanh(差×100/1200) + λ × 線形（駒数＋玉×駒×位置＋脅威）
//   線形の特徴量は lab/nine/x7_expressivity.py の features() と同じ作り方（Python で学習した重みをそのまま読む）
// 探索 = 反復深化 αβ（置換表・駒取り優先・キラー手）＋ 静止探索（駒を取る手。王手されていれば全ての応手）
// 盤面・指し手生成・1手詰めは cshogi（Apery 由来、GPL）の C++ 部分を使う
//
// USI 以外の命令:
//   evaltest           : 標準入力から1行ずつ SFEN を読み、評価値（手番側から見た値、内部の単位）を出力（Python 版との照合用）
//
// ビルド: lab/engine/build.sh

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "generateMoves.hpp"
#include "init.hpp"
#include "position.hpp"
#include "usi.hpp"

// ---- 特徴量の次元（x7_expressivity.py と同じ） ----
constexpr int NP = 28;
constexpr int P_DIM = NP * 81 + 2 * 7 * 19;
constexpr int KP_DIM = 2 * 81 * P_DIM;
constexpr int NT = 17;
constexpr int T_B = 16;
constexpr int PHASES = 4;
constexpr int T_DIM = NT * T_B * (1 + PHASES);
constexpr int OFF_P = 42;
constexpr int OFF_KP = OFF_P + P_DIM;
constexpr int OFF_T = OFF_KP + KP_DIM;
constexpr int DIM = OFF_T + T_DIM;

static const int PV[15] = {0, 1, 3, 4, 5, 8, 10, 6, 0, 6, 6, 6, 6, 10, 12};
static const int HV[7] = {1, 3, 4, 5, 6, 8, 10};

struct Weights {
    float lam = 0.5f;
    float bias = 0.0f;
    std::vector<float> w;
    bool loaded = false;
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        int dim = 0;
        f.read((char*)&lam, 4);
        f.read((char*)&bias, 4);
        f.read((char*)&dim, 4);
        if (dim != DIM) { std::cerr << "info string weight dim mismatch " << dim << " != " << DIM << std::endl; return false; }
        w.resize(DIM);
        f.read((char*)w.data(), 4 * DIM);
        loaded = (bool)f;
        return loaded;
    }
};
static Weights W;
static float LAMBDA_OVERRIDE = -1.0f;
static bool USE_LINEAR = true;

static inline bool inNeigh(int a, int b) {  // b が a の8近傍か
    int fa = a / 9, ra = a % 9, fb = b / 9, rb = b % 9;
    int df = fa - fb, dr = ra - rb;
    return (df || dr) && df >= -1 && df <= 1 && dr >= -1 && dr <= 1;
}

static int materialBalance(const Position& pos, Color c) {
    int s = 0;
    for (int sq = 0; sq < 81; ++sq) {
        int p = pos.piece(Square(sq));
        if (p && (p & 15) != 8) {
            int v = PV[p & 15];
            bool white = p >= 16;
            s += (white == (c == White)) ? v : -v;
        }
    }
    for (int k = 0; k < 7; ++k) {
        int nb = pos.hand(Black).numOf(HandPiece(k));
        int nw = pos.hand(White).numOf(HandPiece(k));
        s += (nb - nw) * HV[k] * (c == Black ? 1 : -1);
    }
    return s;
}

static double classicEval(const Position& pos) { return std::tanh(materialBalance(pos, pos.turn()) * 100.0 / 1200.0); }

struct MoveInfo { int n = 0, press = 0, caps = 0; };

static MoveInfo movesInfo(const Position& pos, int kingTarget) {
    MoveInfo mi;
    for (MoveList<PseudoLegal> ml(pos); !ml.end(); ++ml) {
        Move m = ml.move();
        mi.n++;
        int to = m.to();
        if (inNeigh(kingTarget, to)) mi.press++;
        if (!m.isDrop() && pos.piece(Square(to))) mi.caps++;
    }
    return mi;
}

static int kingEscapes(const Position& pos, int kingSq) {
    int n = 0;
    for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
        Move m = ml.move();
        if (!m.isDrop() && (int)m.from() == kingSq) n++;
    }
    return n;
}

// 線形部分の値 b + w·x（Python の Evaluator("M+KP+T") と同じ）
static double linearEval(Position& pos) {
    const Color turn = pos.turn();
    int pcode[40], psq[40], np = 0;
    float mat[42] = {0};
    for (int sq = 0; sq < 81; ++sq) {
        int p = pos.piece(Square(sq));
        if (!p) continue;
        bool own = (p < 16) == (turn == Black);
        int code = (p & 15) - 1 + (own ? 0 : 14);
        int s = (turn == Black) ? sq : 80 - sq;
        pcode[np] = code;
        psq[np] = s;
        np++;
        mat[code] += 1;
    }
    int hm[7], ho[7];
    for (int k = 0; k < 7; ++k) {
        hm[k] = pos.hand(turn).numOf(HandPiece(k));
        ho[k] = pos.hand(oppositeColor(turn)).numOf(HandPiece(k));
        mat[28 + k] = hm[k];
        mat[35 + k] = ho[k];
    }
    int ksReal = pos.kingSquare(turn), koReal = pos.kingSquare(oppositeColor(turn));
    int ks = (turn == Black) ? ksReal : 80 - ksReal;
    int ko = (turn == Black) ? koReal : 80 - koReal;

    // 脅威の特徴量
    int t[NT];
    MoveInfo mine = movesInfo(pos, koReal);
    bool check = pos.inCheck();
    t[0] = mine.n;
    t[7] = mine.press;
    t[9] = mine.caps;
    t[2] = check;
    t[3] = check ? 0 : (pos.mateMoveIn1Ply().value() != 0);
    t[5] = kingEscapes(pos, ksReal);
    if (!check) {
        StateInfo st;
        pos.doNullMove<true>(st);
        MoveInfo op = movesInfo(pos, ksReal);
        t[1] = op.n;
        t[8] = op.press;
        t[10] = op.caps;
        t[4] = pos.mateMoveIn1Ply().value() != 0;
        t[6] = kingEscapes(pos, koReal);
        pos.doNullMove<false>(st);
    } else {
        t[1] = t[8] = t[10] = t[4] = t[6] = 0;
    }
    auto goldLike = [](int c) { return c == 3 || c == 6 || c == 8 || c == 9 || c == 10 || c == 11; };
    int guard = 0, opGuard = 0, promoted = 0;
    for (int i = 0; i < np; ++i) {
        int c = pcode[i], s = psq[i];
        if (goldLike(c) && (s == ks || inNeigh(ks, s))) guard++;
        if (goldLike(c - 14) && (s == ko || inNeigh(ko, s))) opGuard++;
        if (c % 14 >= 8) promoted++;
    }
    t[11] = guard;
    t[12] = opGuard;
    t[13] = 8 - ks % 9;
    t[14] = ko % 9;
    int sh = 0, so = 0;
    for (int k = 0; k < 7; ++k) { sh += hm[k]; so += ho[k]; }
    t[15] = sh;
    t[16] = so;
    int ph = std::min((sh + so + promoted) / 4, PHASES - 1);

    const float* w = W.w.data();
    double v = W.bias;
    for (int j = 0; j < 42; ++j) v += mat[j] / 4.0 * w[j];
    // 駒×位置と持ち駒（温度計表現）、玉×駒×位置
    auto addP = [&](int pi) {
        v += w[OFF_P + pi];
        v += w[OFF_KP + ks * P_DIM + pi];
        v += w[OFF_KP + 81 * P_DIM + ko * P_DIM + pi];
    };
    for (int i = 0; i < np; ++i) addP(pcode[i] * 81 + psq[i]);
    const int base = NP * 81;
    for (int side = 0; side < 2; ++side)
        for (int k = 0; k < 7; ++k) {
            int cnt = std::min(side == 0 ? hm[k] : ho[k], 19);
            for (int j = 0; j < cnt; ++j) addP(base + (side * 7 + k) * 19 + j);
        }
    for (int i = 0; i < NT; ++i) {
        int val = std::min(t[i], T_B - 1);
        v += w[OFF_T + i * T_B + val];
        v += w[OFF_T + NT * T_B * (1 + ph) + i * T_B + val];
    }
    return v;
}

static long long EVALS = 0;

static double evaluate(Position& pos) {
    EVALS++;
    double v = classicEval(pos);
    if (USE_LINEAR && W.loaded) {
        float lam = LAMBDA_OVERRIDE >= 0 ? LAMBDA_OVERRIDE : W.lam;
        if (lam != 0) v += lam * linearEval(pos);
    }
    return v;
}

// ---- 探索 ----
constexpr double INF = 1e9;
constexpr double MATE = 1e6;
constexpr int MAX_PLY = 64;

struct TTEntry { Key key = 0; double v = 0; int depth = -1; u32 move = 0; u8 flag = 0; };
static std::vector<TTEntry> TT(1 << 20);

struct Search {
    long long nodes = 0, nodeLimit = 0;
    std::chrono::steady_clock::time_point start;
    double timeLimitMs = 0;
    bool stop = false;
    Move killer[MAX_PLY][2];

    bool outOfBudget() {
        if ((nodes & 1023) == 0 && timeLimitMs > 0) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (ms > timeLimitMs) stop = true;
        }
        if (nodeLimit && nodes >= nodeLimit) stop = true;
        return stop;
    }

    int score(const Position& pos, Move m, Move ttMove, int ply) {
        if (m == ttMove) return 1 << 30;
        if (!m.isDrop() && pos.piece(m.to())) return (1 << 20) + PV[pos.piece(m.to()) & 15] * 16 - PV[m.pieceTypeFrom()];
        if (m.isPromotion()) return 1 << 19;
        if (ply < MAX_PLY && (m == killer[ply][0] || m == killer[ply][1])) return 1 << 18;
        return 0;
    }

    double qsearch(Position& pos, double alpha, double beta, int qd, int ply) {
        nodes++;
        if (outOfBudget()) return 0;
        bool check = pos.inCheck();
        double best = -INF;
        if (!check) {
            double stand = evaluate(pos);
            if (stand >= beta || qd <= 0) return stand;
            if (stand > alpha) alpha = stand;
            best = stand;
        }
        std::vector<std::pair<int, Move>> moves;
        if (check) {
            for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) moves.push_back({score(pos, ml.move(), Move::moveNone(), ply), ml.move()});
            if (moves.empty()) return -MATE + ply;
            if (qd <= -4) return evaluate(pos);
        } else {
            for (MoveList<CapturePlusPro> ml(pos); !ml.end(); ++ml)
                if (pos.pseudoLegalMoveIsLegal<false, false>(ml.move(), pos.pinnedBB())) moves.push_back({score(pos, ml.move(), Move::moveNone(), ply), ml.move()});
        }
        std::sort(moves.begin(), moves.end(), [](auto& a, auto& b) { return a.first > b.first; });
        for (auto& [_, m] : moves) {
            StateInfo st;
            pos.doMove(m, st);
            double s = -qsearch(pos, -beta, -alpha, qd - 1, ply + 1);
            pos.undoMove(m);
            if (stop) return 0;
            if (s > best) best = s;
            if (s > alpha) alpha = s;
            if (alpha >= beta) break;
        }
        return best;
    }

    double search(Position& pos, double alpha, double beta, int depth, int ply, Move* bestOut) {
        if (depth <= 0) return qsearch(pos, alpha, beta, 4, ply);
        nodes++;
        if (outOfBudget()) return 0;
        if (ply > 0) {
            RepetitionType rep = pos.isDraw(16);
            if (rep == RepetitionDraw) return 0;
            if (rep == RepetitionWin || rep == RepetitionSuperior) return MATE / 2;
            if (rep == RepetitionLose || rep == RepetitionInferior) return -MATE / 2;
        }
        Key key = pos.getKey();
        TTEntry& e = TT[key & (TT.size() - 1)];
        Move ttMove = Move::moveNone();
        if (e.key == key) {
            ttMove = Move(e.move);
            if (e.depth >= depth && ply > 0) {
                if (e.flag == 0) return e.v;
                if (e.flag == 1 && e.v >= beta) return e.v;
                if (e.flag == 2 && e.v <= alpha) return e.v;
            }
        }
        if (ply > 0 && !pos.inCheck()) {
            Move m1 = pos.mateMoveIn1Ply();
            if (m1.value()) return MATE - ply - 1;
        }
        std::vector<std::pair<int, Move>> moves;
        for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) moves.push_back({score(pos, ml.move(), ttMove, ply), ml.move()});
        if (moves.empty()) return -MATE + ply;
        std::sort(moves.begin(), moves.end(), [](auto& a, auto& b) { return a.first > b.first; });
        double best = -INF, a0 = alpha;
        Move bestM = moves[0].second;
        for (auto& [_, m] : moves) {
            StateInfo st;
            pos.doMove(m, st);
            double s = -search(pos, -beta, -alpha, depth - 1, ply + 1, nullptr);
            pos.undoMove(m);
            if (stop) return 0;
            if (s > best) { best = s; bestM = m; }
            if (s > alpha) alpha = s;
            if (alpha >= beta) {
                if (ply < MAX_PLY && !(!m.isDrop() && pos.piece(m.to())) && m != killer[ply][0]) { killer[ply][1] = killer[ply][0]; killer[ply][0] = m; }
                break;
            }
        }
        e.key = key; e.v = best; e.depth = depth; e.move = bestM.value();
        e.flag = best >= beta ? 1 : (best <= a0 ? 2 : 0);
        if (bestOut) *bestOut = bestM;
        return best;
    }

    Move think(Position& pos, int maxDepth, double& outScore, int& outDepth) {
        start = std::chrono::steady_clock::now();
        stop = false;
        nodes = 0;
        std::memset(killer, 0, sizeof killer);
        Move best = Move::moveNone();
        {
            MoveList<LegalAll> ml(pos);
            if (ml.size() == 0) return best;
            best = ml.move();
        }
        Move m1 = pos.inCheck() ? Move::moveNone() : pos.mateMoveIn1Ply();
        if (m1.value()) { outScore = MATE; outDepth = 1; return m1; }
        for (int d = 1; d <= maxDepth; ++d) {
            Move m = Move::moveNone();
            double v = search(pos, -INF, INF, d, 0, &m);
            if (stop) break;
            best = m;
            outScore = v;
            outDepth = d;
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            std::cout << "info depth " << d << " score cp " << (int)std::round(v * 1000) << " nodes " << nodes << " time " << (int)ms << " pv " << m.toUSI() << std::endl;
            if (std::abs(v) >= MATE / 2) break;
        }
        return best;
    }
};

static void setPosition(Position& pos, std::istringstream& is, std::vector<StateInfo>& states) {
    std::string token, sfen;
    is >> token;
    if (token == "startpos") {
        sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        is >> token;
    } else if (token == "sfen") {
        while (is >> token && token != "moves") sfen += token + " ";
    }
    pos.set(sfen);
    states.clear();
    states.reserve(1024);
    while (is >> token) {
        Move m = usiToMove(pos, token);
        if (!m.value()) break;
        states.emplace_back();
        pos.doMove(m, states.back());
    }
}

int main() {
    initTable();
    Position::initZobrist();
    std::string evalFile = "weights.bin";
    Position pos;
    std::vector<StateInfo> states;
    pos.set(std::string("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"));
    Search S;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd;
        is >> cmd;
        if (cmd == "usi") {
            std::cout << "id name NonNN-Linear\nid author shogi-ai lab\n";
            std::cout << "option name EvalFile type string default weights.bin\n";
            std::cout << "option name Lambda type string default -1\n";
            std::cout << "option name UseLinear type check default true\n";
            std::cout << "usiok" << std::endl;
        } else if (cmd == "setoption") {
            std::string tok, name, value;
            is >> tok >> name >> tok >> value;
            if (name == "EvalFile") evalFile = value;
            if (name == "Lambda") LAMBDA_OVERRIDE = std::stof(value);
            if (name == "UseLinear") USE_LINEAR = (value == "true");
        } else if (cmd == "isready") {
            if (USE_LINEAR && !W.loaded && !W.load(evalFile)) std::cout << "info string cannot load " << evalFile << std::endl;
            std::cout << "readyok" << std::endl;
        } else if (cmd == "usinewgame") {
            std::fill(TT.begin(), TT.end(), TTEntry());
        } else if (cmd == "position") {
            setPosition(pos, is, states);
        } else if (cmd == "go") {
            std::string tok;
            int depth = MAX_PLY - 8;
            S.nodeLimit = 0;
            S.timeLimitMs = 0;
            double btime = 0, wtime = 0, byoyomi = 0, binc = 0, winc = 0;
            while (is >> tok) {
                if (tok == "depth") is >> depth;
                else if (tok == "nodes") is >> S.nodeLimit;
                else if (tok == "movetime") is >> S.timeLimitMs;
                else if (tok == "btime") is >> btime;
                else if (tok == "wtime") is >> wtime;
                else if (tok == "byoyomi") is >> byoyomi;
                else if (tok == "binc") is >> binc;
                else if (tok == "winc") is >> winc;
            }
            if (S.timeLimitMs == 0 && (btime || wtime || byoyomi)) {
                double mine = pos.turn() == Black ? btime : wtime;
                double inc = pos.turn() == Black ? binc : winc;
                S.timeLimitMs = std::max(50.0, mine / 30 + inc + byoyomi - 100);
            }
            double v = 0;
            int d = 0;
            Move m = S.think(pos, depth, v, d);
            std::cout << "bestmove " << (m.value() ? m.toUSI() : std::string("resign")) << std::endl;
        } else if (cmd == "evaltest") {
            if (!W.loaded) W.load(evalFile);
            std::string s;
            while (std::getline(std::cin, s) && !s.empty()) {
                pos.set(s);
                double lin = W.loaded ? linearEval(pos) : 0;
                std::printf("%.6f %.6f\n", classicEval(pos), lin);
            }
            std::fflush(stdout);
        } else if (cmd == "quit") {
            break;
        }
    }
    return 0;
}
