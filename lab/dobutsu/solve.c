// どうぶつしょうぎ 完全解析（後退解析・距離付き）
//
// 局面は常に「手番側から見た向き」で正規化し、さらに左右反転の小さい方を代表にする。
// 盤: 3列x4段, idx = r*3+c, r=0 が手番側から見て奥（相手陣=トライ/成りの段）。
// 駒コード(4bit): 0空 1L 2G 3E 4C(ひよこ) 5H(にわとり)、相手駒は +8。
// key = 盤 48bit | 手番側持ち駒 G,E,C (2bit x3) <<48 | 相手持ち駒 <<54
//
// 値 vals[i]: 0=未確定, それ以外は d+1 (d=終局までの手数、d奇数=手番側勝ち、偶数=負け)
// 未確定のまま残った局面は千日手による引き分け。
//
// 出力: <outdir>/keys.bin (mix(key) 昇順 uint64), vals.bin (uint8), summary.txt
// 列挙: 即勝ちのある局面は終局扱いで展開しない（勝ちを見送る手順は最善手順に現れないため）
// 使い方: solve <outdir>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

typedef uint64_t u64;

enum { EMPTY = 0, L = 1, G = 2, E = 3, C = 4, H = 5, OPP = 8 };

typedef struct {
    uint8_t sq[12];
    uint8_t hm[3], ho[3]; // G,E,C
} Pos;

static const int DIRS[6][8][2] = {
    {{0}},
    {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}}, // L
    {{-1,0},{1,0},{0,-1},{0,1}},                             // G
    {{-1,-1},{-1,1},{1,-1},{1,1}},                           // E
    {{-1,0}},                                                // C
    {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,0}},              // H
};
static const int NDIR[6] = {0, 8, 4, 4, 1, 6};

static u64 encode(const Pos *p) {
    u64 k = 0;
    for (int i = 0; i < 12; i++) k |= (u64)p->sq[i] << (4 * i);
    for (int j = 0; j < 3; j++) {
        k |= (u64)p->hm[j] << (48 + 2 * j);
        k |= (u64)p->ho[j] << (54 + 2 * j);
    }
    return k;
}

static void decode(u64 k, Pos *p) {
    for (int i = 0; i < 12; i++) p->sq[i] = (k >> (4 * i)) & 15;
    for (int j = 0; j < 3; j++) {
        p->hm[j] = (k >> (48 + 2 * j)) & 3;
        p->ho[j] = (k >> (54 + 2 * j)) & 3;
    }
}

static u64 mirror_key(u64 k) {
    u64 r = k & ~((1ULL << 48) - 1);
    for (int row = 0; row < 4; row++)
        for (int c = 0; c < 3; c++) {
            u64 v = (k >> (4 * (row * 3 + c))) & 15;
            r |= v << (4 * (row * 3 + (2 - c)));
        }
    return r;
}

static u64 canon(u64 k) {
    u64 m = mirror_key(k);
    return m < k ? m : k;
}

// 手を指した後の局面を相手手番視点に回転して key 化
static u64 flip_encode(const Pos *p) {
    Pos q;
    for (int i = 0; i < 12; i++) {
        uint8_t v = p->sq[i];
        q.sq[11 - i] = v ? (v ^ OPP) : 0;
    }
    for (int j = 0; j < 3; j++) { q.hm[j] = p->ho[j]; q.ho[j] = p->hm[j]; }
    return canon(encode(&q));
}

// 相手駒が s に利いているか
static int opp_attacks(const Pos *p, int s) {
    int sr = s / 3, sc = s % 3;
    for (int i = 0; i < 12; i++) {
        uint8_t v = p->sq[i];
        if (!(v & OPP)) continue;
        int t = v & 7, r = i / 3, c = i % 3;
        for (int d = 0; d < NDIR[t]; d++) {
            // 相手駒は向きが逆
            if (r - DIRS[t][d][0] == sr && c + DIRS[t][d][1] == sc) return 1;
        }
    }
    return 0;
}

// splitmix64 の finalizer（全単射）とその逆
static u64 mix(u64 x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}
static u64 inv_odd(u64 a) { u64 x = a; for (int i = 0; i < 6; i++) x *= 2 - a * x; return x; }
static u64 unxs(u64 y, int s) { u64 x = y; for (int i = 0; i < 64 / s + 1; i++) x = y ^ (x >> s); return x; }
static u64 INV1, INV2;
static u64 unmix(u64 x) {
    x = unxs(x, 31); x *= INV2;
    x = unxs(x, 27); x *= INV1;
    x = unxs(x, 30); return x;
}

// expand() が子局面ごとに記録する手の情報: from(4bit, 15=打ち) | to<<4 | 駒<<8 | 取った駒<<12
static uint16_t mvinfo[128];

// 局面を展開する。戻り値: 1=即勝ち（ライオン捕獲 or トライ成立 or 既に自ライオンが奥の段）、
// 0=通常。通常時 children[] に子局面(mix済み)を入れ *n に数を返す。
// enumerate=1 のときは即勝ちでも子を全て列挙する（到達局面の列挙用）
static int expand(u64 key, u64 *children, int *n, int enumerate) {
    Pos p; decode(key, &p);
    int win = 0, cnt = 0;
    // 自ライオンが既に奥の段にいる = 相手がトライを止めなかった → 終局（勝ち）
    for (int i = 0; i < 3; i++) if (p.sq[i] == L) { *n = 0; return 1; }
    for (int i = 0; i < 12; i++) {
        uint8_t v = p.sq[i];
        if (v == EMPTY || (v & OPP)) continue;
        int r = i / 3, c = i % 3;
        for (int d = 0; d < NDIR[v]; d++) {
            int nr = r + DIRS[v][d][0], nc = c + DIRS[v][d][1];
            if (nr < 0 || nr > 3 || nc < 0 || nc > 2) continue;
            int s = nr * 3 + nc;
            uint8_t cap = p.sq[s];
            if (cap && !(cap & OPP)) continue;
            if (cap == (L | OPP)) { win = 1; if (!enumerate) { *n = 0; return 1; } continue; }
            Pos q = p;
            q.sq[i] = EMPTY;
            q.sq[s] = (v == C && nr == 0) ? H : v;
            if (cap) {
                int t = cap & 7; if (t == H) t = C;
                q.hm[t - G]++;
            }
            if (v == L && nr == 0 && !opp_attacks(&q, s)) {
                win = 1; if (!enumerate) { *n = 0; return 1; } continue;
            }
            mvinfo[cnt] = i | (s << 4) | (v << 8) | ((cap & 7) << 12);
            children[cnt++] = mix(flip_encode(&q));
        }
    }
    for (int j = 0; j < 3; j++) {
        if (!p.hm[j]) continue;
        for (int s = 0; s < 12; s++) {
            if (p.sq[s]) continue;
            Pos q = p; q.sq[s] = G + j; q.hm[j]--;
            mvinfo[cnt] = 15 | (s << 4) | ((G + j) << 8);
            children[cnt++] = mix(flip_encode(&q));
        }
    }
    *n = cnt;
    return win;
}

static int cmp_u64(const void *a, const void *b) {
    u64 x = *(const u64 *)a, y = *(const u64 *)b;
    return x < y ? -1 : x > y;
}
static size_t uniq(u64 *a, size_t n) {
    if (!n) return 0;
    qsort(a, n, sizeof(u64), cmp_u64);
    size_t m = 1;
    for (size_t i = 1; i < n; i++) if (a[i] != a[m - 1]) a[m++] = a[i];
    return m;
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double rss_mb(void) { struct rusage u; getrusage(RUSAGE_SELF, &u); return u.ru_maxrss / 1048576.0; }

static u64 *all; static size_t nall;
#define BUCKET_BITS 24
static uint32_t *boff;

static long find(u64 m) {
    size_t b = m >> (64 - BUCKET_BITS);
    size_t lo = boff[b], hi = boff[b + 1];
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (all[mid] < m) lo = mid + 1; else hi = mid;
    }
    return (lo < nall && all[lo] == m) ? (long)lo : -1;
}

// 大きい配列はファイルに mmap する（OS がいつでもページを捨てられるので、スワップを圧迫しない）
typedef struct { int fd; void *p; size_t bytes; } FMap;
static void *fmap_resize(FMap *m, const char *path, size_t bytes) {
    if (m->fd < 0) { m->fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644); if (m->fd < 0) { perror(path); exit(1); } m->p = NULL; }
    if (m->p) munmap(m->p, m->bytes);
    if (bytes < 16384) bytes = 16384;
    if (ftruncate(m->fd, bytes)) { perror("ftruncate"); exit(1); }
    m->p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, m->fd, 0);
    if (m->p == MAP_FAILED) { perror("mmap"); exit(1); }
    m->bytes = bytes;
    return m->p;
}

static Pos initial(void) {
    Pos p; memset(&p, 0, sizeof p);
    p.sq[9] = G; p.sq[10] = L; p.sq[11] = E; p.sq[7] = C;
    p.sq[0] = E | OPP; p.sq[1] = L | OPP; p.sq[2] = G | OPP; p.sq[4] = C | OPP;
    return p;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: solve <outdir>\n"); return 1; }
    const char *out = argv[1];
    INV1 = inv_odd(0xbf58476d1ce4e5b9ULL); INV2 = inv_odd(0x94d049bb133111ebULL);
    double t0 = now();
    u64 ch[128]; int nc;

    Pos ip = initial();
    u64 start = mix(canon(encode(&ip)));
    if (unmix(start) != canon(encode(&ip))) { fprintf(stderr, "unmix broken\n"); return 1; }
    expand(unmix(start), ch, &nc, 1);
    printf("initial legal moves (non-winning): %d\n", nc);

    // ---- 1. 到達可能局面の列挙（BFS） ----
    char path[1024], pa[1024], pb[1024];
    snprintf(path, sizeof path, "%s/keys.bin", out);
    snprintf(pa, sizeof pa, "%s/tmp_front_a.bin", out);
    snprintf(pb, sizeof pb, "%s/tmp_front_b.bin", out);
    FMap mall = {-1}, mfa = {-1}, mfb = {-1};
    FMap *mfront = &mfa, *mnext = &mfb;
    const char *pfront = pa, *pnext = pb;
    size_t cap_all = 1 << 20; all = fmap_resize(&mall, path, cap_all * 8); all[0] = start; nall = 1;
    u64 *front = fmap_resize(mfront, pfront, 8); front[0] = start; size_t nfront = 1;
    const size_t BUF = 1 << 24; u64 *buf = malloc(BUF * 8);
    int layer = 0;
    while (nfront) {
        size_t cap_next = 1 << 20, nnext = 0;
        u64 *next = fmap_resize(mnext, pnext, cap_next * 8);
        size_t nb = 0;
        for (size_t f = 0; f <= nfront; f++) {
            if (f < nfront) {
                // 即勝ちのある局面（ライオンを取れる/トライ成立）は終局として扱い、その先は展開しない
                if (expand(unmix(front[f]), ch, &nc, 0)) nc = 0;
                memcpy(buf + nb, ch, nc * 8); nb += nc;
            }
            if (nb > BUF - 128 || (f == nfront && nb)) {
                nb = uniq(buf, nb);
                // all との差分（両方ソート済みなので二分探索で開始位置を決めてマージ）
                size_t i = 0, w = 0;
                for (size_t j = 0; j < nb; j++) {
                    while (i < nall && all[i] < buf[j]) i++;
                    if (i < nall && all[i] == buf[j]) continue;
                    buf[w++] = buf[j];
                }
                if (nnext + w > cap_next) { cap_next = (nnext + w) * 3 / 2; next = fmap_resize(mnext, pnext, cap_next * 8); }
                memcpy(next + nnext, buf, w * 8); nnext += w; nb = 0;
            }
        }
        nnext = uniq(next, nnext);
        // all に next をマージ（後ろから）
        if (nall + nnext > cap_all) { cap_all = (nall + nnext) * 5 / 4; all = fmap_resize(&mall, path, cap_all * 8); }
        long i = (long)nall - 1, j = (long)nnext - 1, w = (long)(nall + nnext) - 1;
        while (j >= 0) {
            if (i >= 0 && all[i] > next[j]) all[w--] = all[i--];
            else all[w--] = next[j--];
        }
        nall += nnext;
        // front と next を入れ替え
        FMap *tm = mfront; mfront = mnext; mnext = tm;
        const char *tp = pfront; pfront = pnext; pnext = tp;
        front = next; nfront = nnext;
        layer++;
        printf("bfs layer %d: new %zu, total %zu, %.0fs, rss %.0fMB\n", layer, nfront, nall, now() - t0, rss_mb());
        fflush(stdout);
    }
    free(buf);
    munmap(mfa.p, mfa.bytes); close(mfa.fd); unlink(pa);
    munmap(mfb.p, mfb.bytes); close(mfb.fd); unlink(pb);
    all = fmap_resize(&mall, path, nall * 8);

    boff = calloc(((size_t)1 << BUCKET_BITS) + 1, 4);
    for (size_t i = 0; i < nall; i++) boff[(all[i] >> (64 - BUCKET_BITS)) + 1]++;
    for (size_t b = 0; b < ((size_t)1 << BUCKET_BITS); b++) boff[b + 1] += boff[b];

    // ---- 2. 後退解析（距離 d ごとの層別パス） ----
    char vpath[1024]; snprintf(vpath, sizeof vpath, "%s/vals.bin", out);
    FMap mv = {-1}; uint8_t *vals = fmap_resize(&mv, vpath, nall);
    memset(vals, 0, nall);
    size_t nwin1 = 0, nloss0 = 0;
    for (size_t i = 0; i < nall; i++) {
        if (expand(unmix(all[i]), ch, &nc, 0)) { vals[i] = 2; nwin1++; }
        else if (nc == 0) { vals[i] = 1; nloss0++; }
    }
    printf("d=1 wins %zu, d=0 losses %zu, %.0fs\n", nwin1, nloss0, now() - t0); fflush(stdout);
    int empty_streak = 0, maxd = 1;
    for (int d = 2; d < 254 && empty_streak < 2; d++) {
        size_t changed = 0;
        for (size_t i = 0; i < nall; i++) {
            if (vals[i]) continue;
            expand(unmix(all[i]), ch, &nc, 0);
            if (d & 1) { // 勝ち: 負けの子が1つでもあれば
                for (int k = 0; k < nc; k++) {
                    long x = find(ch[k]);
                    if (x < 0) { fprintf(stderr, "missing child\n"); return 1; }
                    uint8_t v = vals[x];
                    if (v && ((v - 1) & 1) == 0 && v - 1 < d) { vals[i] = d + 1; changed++; break; }
                }
            } else { // 負け: 子が全て確定した勝ち
                int ok = 1;
                for (int k = 0; k < nc && ok; k++) {
                    uint8_t v = vals[find(ch[k])];
                    if (!v || ((v - 1) & 1) == 0 || v - 1 >= d) ok = 0;
                }
                if (ok) { vals[i] = d + 1; changed++; }
            }
        }
        if (changed) { empty_streak = 0; maxd = d; } else empty_streak++;
        printf("d=%d: %zu, %.0fs\n", d, changed, now() - t0); fflush(stdout);
    }

    // ---- 3. 集計・保存 ----
    size_t nw = 0, nl = 0, nd = 0;
    for (size_t i = 0; i < nall; i++) {
        if (!vals[i]) nd++; else if ((vals[i] - 1) & 1) nw++; else nl++;
    }
    long si = find(start);
    int sv = vals[si];
    FILE *fp;
    msync(all, nall * 8, MS_SYNC); msync(vals, nall, MS_SYNC);
    if (ftruncate(mall.fd, nall * 8) || ftruncate(mv.fd, nall)) perror("ftruncate");
    snprintf(path, sizeof path, "%s/summary.txt", out); fp = fopen(path, "w");
    fprintf(fp, "positions(mirror-canonical, side-to-move normalized): %zu\n", nall);
    fprintf(fp, "win: %zu\nloss: %zu\ndraw(unresolved): %zu\nmax_d: %d\n", nw, nl, nd, maxd);
    fprintf(fp, "initial: %s d=%d\n", !sv ? "DRAW" : (((sv - 1) & 1) ? "WIN(side to move)" : "LOSS(side to move)"), sv - 1);
    fprintf(fp, "elapsed_sec: %.0f\npeak_rss_mb: %.0f\n", now() - t0, rss_mb());
    fclose(fp);
    printf("done. initial vals=%d (d=%d). see %s/summary.txt\n", sv, sv - 1, out);
    return 0;
}
