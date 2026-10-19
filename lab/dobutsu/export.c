// 真値DBを「両ライオンの位置」ごとのグループに並べ直して書き出す（パターン検証用の索引）
// 出力: <dir>/plain_keys.bin (u64, mix を戻した正規化 key), plain_vals.bin (u8), groups.bin (u64 x 145 の開始位置)
// グループ番号 = 手番側ライオンのマス * 12 + 相手ライオンのマス
// 使い方: export <datadir>
#define main solve_main
#include "solve.c"
#undef main
#include <sys/stat.h>

static int group_of(u64 k) {
    int my = 0, op = 0;
    for (int i = 0; i < 12; i++) {
        int v = (k >> (4 * i)) & 15;
        if (v == L) my = i;
        if (v == (L | OPP)) op = i;
    }
    return my * 12 + op;
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    INV1 = inv_odd(0xbf58476d1ce4e5b9ULL); INV2 = inv_odd(0x94d049bb133111ebULL);
    char path[1024]; int fd; struct stat st;
    snprintf(path, sizeof path, "%s/keys.bin", argv[1]); fd = open(path, O_RDONLY); fstat(fd, &st);
    size_t n = st.st_size / 8; u64 *keys = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    snprintf(path, sizeof path, "%s/vals.bin", argv[1]); fd = open(path, O_RDONLY); fstat(fd, &st);
    uint8_t *vals = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);

    u64 cnt[145] = {0};
    for (size_t i = 0; i < n; i++) cnt[group_of(unmix(keys[i])) + 1]++;
    for (int g = 0; g < 144; g++) cnt[g + 1] += cnt[g];
    u64 pos[144]; memcpy(pos, cnt, sizeof pos);

    snprintf(path, sizeof path, "%s/plain_keys.bin", argv[1]);
    FMap mk = {-1}; u64 *ok = fmap_resize(&mk, path, n * 8);
    snprintf(path, sizeof path, "%s/plain_vals.bin", argv[1]);
    FMap mv = {-1}; uint8_t *ov = fmap_resize(&mv, path, n);
    for (size_t i = 0; i < n; i++) {
        u64 k = unmix(keys[i]);
        u64 p = pos[group_of(k)]++;
        ok[p] = k; ov[p] = vals[i];
    }
    msync(ok, n * 8, MS_SYNC); msync(ov, n, MS_SYNC);
    snprintf(path, sizeof path, "%s/groups.bin", argv[1]);
    FILE *fp = fopen(path, "wb"); fwrite(cnt, 8, 145, fp); fclose(fp);
    printf("exported %zu positions into 144 groups\n", n);
    return 0;
}
