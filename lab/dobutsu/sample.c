// 解析済みDBから局面をランダムに抜き出す（実験用）
// 使い方: sample <datadir> <n> <seed> <with_children:0|1> > out.bin
// レコード: u64 key, u8 val, u8 immediate_win, u8 nc, [nc x (u64 key, u8 val, u16 mvinfo)]
#define main solve_main
#include "solve.c"
#undef main
#include <sys/stat.h>

static u64 rng_state;
static u64 rnd(void) { return mix(rng_state += 0x9e3779b97f4a7c15ULL); }

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: sample <datadir> <n> <seed> <with_children>\n"); return 1; }
    INV1 = inv_odd(0xbf58476d1ce4e5b9ULL); INV2 = inv_odd(0x94d049bb133111ebULL);
    long n = atol(argv[2]); rng_state = atol(argv[3]); int wc = atoi(argv[4]);
    // DB は読み取り専用で mmap（ファイルキャッシュとして扱われ、スワップを圧迫しない）
    char path[1024]; int fd; struct stat st;
    snprintf(path, sizeof path, "%s/keys.bin", argv[1]); fd = open(path, O_RDONLY); fstat(fd, &st);
    nall = st.st_size / 8; all = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    snprintf(path, sizeof path, "%s/vals.bin", argv[1]); fd = open(path, O_RDONLY); fstat(fd, &st);
    uint8_t *vals = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (all == MAP_FAILED || vals == MAP_FAILED) { perror("mmap"); return 1; }
    boff = calloc(((size_t)1 << BUCKET_BITS) + 1, 4);
    for (size_t i = 0; i < nall; i++) boff[(all[i] >> (64 - BUCKET_BITS)) + 1]++;
    for (size_t b = 0; b < ((size_t)1 << BUCKET_BITS); b++) boff[b + 1] += boff[b];

    u64 ch[128]; int nc;
    for (long s = 0; s < n; s++) {
        size_t i = rnd() % nall;
        u64 key = unmix(all[i]);
        uint8_t v = vals[i], imm = 0, ncb = 0;
        if (wc) { imm = expand(key, ch, &nc, 0); ncb = imm ? 0 : nc; }
        fwrite(&key, 8, 1, stdout); fwrite(&v, 1, 1, stdout); fwrite(&imm, 1, 1, stdout); fwrite(&ncb, 1, 1, stdout);
        for (int k = 0; k < ncb; k++) {
            long x = find(ch[k]);
            u64 ck = unmix(ch[k]); uint8_t cv = vals[x];
            fwrite(&ck, 8, 1, stdout); fwrite(&cv, 1, 1, stdout); fwrite(&mvinfo[k], 2, 1, stdout);
        }
    }
    return 0;
}
