/*
============================================================================
本ソフトウェアは実験・研究・個人利用のみを目的とし、無保証で提供される。
使用は必ず自身が管理・所有するネットワークに限ること。
本ソフトウェアの使用により生じたいかなる損害についても、作者は一切の責任を負わない。
============================================================================

PPP CHAP (RFC 1994) のシークレット(パスワード)を CUDA で総当たり復元する。
Response = MD5( Identifier(1byte) || Secret || Challenge )
Identifier / Challenge / Response(= 探索対象の MD5) はキャプチャから読み取り、コマンドライン引数(--id / --challenge / --target)で与える。いずれも必須。

windows:  nvcc -O3 -arch=native -Xcompiler /utf-8,/WX chap_md5_bulldozer.cu -o chap_md5_bulldozer.exe
Linux:    nvcc -O3 -arch=native chap_md5_bulldozer.cu -o chap_md5_bulldozer
(-arch=native は実行ホストの GPU 世代を自動選択。/utf-8 は日本語コメントをMSVC が CP932 誤読するのを防ぐため必須)
早期終了 + RK 事前加算により素朴実装比で約 +20〜26%。単一ブロック MD5。

実行例:
./chap_md5_bulldozer --id <ID> --challenge <hex> --target <md5hex>
                                                既定文字集合(a-z0-9)で長さ1〜8を探索
./chap_md5_bulldozer --id <ID> --challenge <hex> --target <md5hex> --charset "abcdefghijklmnopqrstuvwxyz0123456789" --min 1 --max 12
./chap_md5_bulldozer ... --charset-regex "a-z0-9"   // 正規表現の文字クラス風(範囲/ \d\l\u\w\p\a)
./chap_md5_bulldozer ... --charset-classes aA0!     // 代表1文字→クラス展開(a-z A-Z 0-9 記号)
./chap_md5_bulldozer --id <ID> --challenge <hex> --test <password>
                                            // 特定パスワードのハッシュを検算(GPU不要)
./chap_md5_bulldozer ... --batch 1024         // 1スレッドが処理する候補数(除算の償却量)
./chap_md5_bulldozer ... --gpus 4             // 先頭4基のGPUを使用(既定は全基)
./chap_md5_bulldozer ... --devices 0,2,3      // 使用するGPUを明示指定
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <cuda_runtime.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h> // SetConsoleOutputCP: 免責文(UTF-8)をコンソールで正しく表示するため
#endif

#define MAX_PW_LEN 16 // 高速パスが特殊化するパスワード長の上限(必要なら拡張可)
#define MAX_CHAL_LEN 55
#define TPB 256 // threads per block

#define CUDA_CHECK(x)                                                                               \
    do                                                                                              \
    {                                                                                               \
        cudaError_t e = (x);                                                                        \
        if (e != cudaSuccess)                                                                       \
        {                                                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            exit(1);                                                                                \
        }                                                                                           \
    } while (0)

// ---- MD5 圧縮関数 (単一ブロック: メッセージ長 <= 55) --------------------------
__host__ __device__ __forceinline__ uint32_t rotl32(uint32_t x, int c)
{
    return (x << c) | (x >> (32 - c));
}

static inline uint32_t rotr32(uint32_t x, int c)
{
    return (x >> c) | (x << (32 - c));
}

static const uint32_t MD5_K[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};

static const int MD5_S[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

static inline int md5_word_index(int i)
{
    if (i < 16)
        return i;
    if (i < 32)
        return (5 * i + 1) & 15;
    if (i < 48)
        return (3 * i + 5) & 15;
    return (7 * i) & 15;
}

// M[16] は呼び出し側のレジスタ配列を想定。__forceinline__ + 完全アンロールにより
// M[g]/out[] の添字がコンパイル時定数へ畳み込まれ、ptxas がレジスタへ昇格する。
static inline void md5_compress(const uint32_t M[16], uint32_t out[4])
{
    const uint32_t K[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
    const int s[64] = {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

    const uint32_t a0 = 0x67452301, b0 = 0xefcdab89, c0 = 0x98badcfe, d0 = 0x10325476;
    uint32_t A = a0, B = b0, C = c0, D = d0;
#pragma unroll
    for (int i = 0; i < 64; i++)
    {
        uint32_t f;
        int g;
        if (i < 16)
        {
            f = (B & C) | (~B & D);
            g = i;
        }
        else if (i < 32)
        {
            f = (D & B) | (~D & C);
            g = (5 * i + 1) & 15;
        }
        else if (i < 48)
        {
            f = B ^ C ^ D;
            g = (3 * i + 5) & 15;
        }
        else
        {
            f = C ^ (B | ~D);
            g = (7 * i) & 15;
        }
        f = f + A + K[i] + M[g];
        A = D;
        D = C;
        C = B;
        B = B + rotl32(f, s[i]);
    }
    out[0] = a0 + A;
    out[1] = b0 + B;
    out[2] = c0 + C;
    out[3] = d0 + D;
}

// メッセージ(msg,len) から M を組み立てて圧縮 (ホストの --test 検算用)
static void md5_single(const unsigned char *msg, int len, uint32_t out[4])
{
    uint32_t M[16];
    for (int i = 0; i < 16; i++)
        M[i] = 0;
    for (int i = 0; i < len; i++)
        M[i >> 2] |= (uint32_t)msg[i] << ((i & 3) * 8);
    M[len >> 2] |= (uint32_t)0x80 << ((len & 3) * 8);
    M[14] = (uint32_t)(len * 8);
    md5_compress(M, out);
}

// ---- 定数メモリ --------------------------------------------------------------
__constant__ unsigned char d_charset[128]; // 文字集合(共有メモリへ複製して使用)
__constant__ uint32_t d_RK[64];            // MD5_K[i] + 定数メッセージ語

// 普段は EARLY_STEPS 段だけ計算し、32-bit 比較で除外する。
// LEN<=3 は末尾15段、LEN<=7 は末尾8段が固定語のみなのでホスト側で逆算できる。
// LEN>=8 は 61 段後の B が最終 A と同じであることを使い、末尾3段を省略する。
template <int LEN>
__device__ __forceinline__ bool md5_candidate(
    const uint32_t pwWords[5], uint32_t earlyA, uint32_t ta, uint32_t tb, uint32_t tc, uint32_t td)
{
    enum
    {
        EARLY_STEPS = (LEN <= 3) ? 49 : ((LEN <= 7) ? 56 : 61)
    };
    const int s[64] = {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

    uint32_t A = 0x67452301, B = 0xefcdab89, C = 0x98badcfe, D = 0x10325476;
#pragma unroll
    for (int i = 0; i < EARLY_STEPS; i++)
    {
        uint32_t f;
        int g;
        if (i < 16)
        {
            f = (B & C) | (~B & D);
            g = i;
        }
        else if (i < 32)
        {
            f = (D & B) | (~D & C);
            g = (5 * i + 1) & 15;
        }
        else if (i < 48)
        {
            f = B ^ C ^ D;
            g = (3 * i + 5) & 15;
        }
        else
        {
            f = C ^ (B | ~D);
            g = (7 * i) & 15;
        }
        f = f + A + d_RK[i];
        if (g <= (LEN >> 2))
            f += pwWords[g];
        A = D;
        D = C;
        C = B;
        B = B + rotl32(f, s[i]);
    }

    if (LEN <= 7)
    {
        if (A != earlyA)
            return false;
    }
    else if (B != ta)
        return false;

    // 32-bit 偶然一致は約50億候補に1回だけ。その場合のみ残りを完走する。
#pragma unroll
    for (int i = EARLY_STEPS; i < 64; i++)
    {
        uint32_t f;
        int g;
        if (i < 16)
        {
            f = (B & C) | (~B & D);
            g = i;
        }
        else if (i < 32)
        {
            f = (D & B) | (~D & C);
            g = (5 * i + 1) & 15;
        }
        else if (i < 48)
        {
            f = B ^ C ^ D;
            g = (3 * i + 5) & 15;
        }
        else
        {
            f = C ^ (B | ~D);
            g = (7 * i) & 15;
        }
        f = f + A + d_RK[i];
        if (g <= (LEN >> 2))
            f += pwWords[g];
        A = D;
        D = C;
        C = B;
        B = B + rotl32(f, s[i]);
    }
    return A == ta && B == tb && C == tc && D == td;
}

// ---- 特殊化カーネル ----------------------------------------------------------
// LEN はパスワード長(コンパイル時定数)。1スレッドが batch 個の候補を連続処理する。
template <int LEN>
__global__ void __launch_bounds__(TPB) crack(
    uint64_t offset, uint64_t count, int batch, int clen, uint32_t earlyA, uint32_t ta, uint32_t tb, uint32_t tc, uint32_t td, int *found, unsigned char *result)
{
    // 文字集合を共有メモリへ (ダイバージェント参照の直列化を回避)
    __shared__ unsigned char s_charset[128];
    for (int i = threadIdx.x; i < clen; i += blockDim.x)
        s_charset[i] = d_charset[i];
    __syncthreads();

    if (*found)
        return;

    const uint64_t globalEnd = offset + count;
    uint64_t start = offset + (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) * (uint64_t)batch;
    if (start >= globalEnd)
        return;
    const uint64_t remaining = globalEnd - start;
    const int work = remaining < (uint64_t)batch ? (int)remaining : batch;

    // オドメーター初期化: start を base-clen 展開 (スレッドあたり1回だけ除算)
    int dig[LEN];
    uint32_t pwWords[5] = {0, 0, 0, 0, 0}; // ID の後ろのパスワードバイトだけ保持
    {
        uint64_t code = start;
#pragma unroll
        for (int i = LEN - 1; i >= 0; i--)
        {
            int d = (int)(code % (uint64_t)clen);
            code /= (uint64_t)clen;
            dig[i] = d;
            const int o = 1 + i;
            pwWords[o >> 2] |= (uint32_t)s_charset[d] << ((o & 3) * 8);
        }
    }

    for (int n = 0; n < work; n++)
    {
        if (md5_candidate<LEN>(pwWords, earlyA, ta, tb, tc, td))
        {
            if (atomicExch(found, 1) == 0)
            {
#pragma unroll
                for (int i = 0; i < LEN; i++)
                    result[i] = s_charset[dig[i]];
                result[LEN] = 0;
            }
            return;
        }

        // オドメーター +1。大半の候補では最下位桁だけが変わるため、桁上がり時だけ上位桁へ進む。
        int d = dig[LEN - 1] + 1;
        if (d < clen)
        {
            dig[LEN - 1] = d;
            const int o = 1 + (LEN - 1);
            const uint32_t shift = (o & 3) * 8;
            const uint32_t mask = 0xffu << shift;
            pwWords[o >> 2] = (pwWords[o >> 2] & ~mask) |
                              ((uint32_t)s_charset[d] << shift);
            continue;
        }
        dig[LEN - 1] = 0;
        {
            const int o = 1 + (LEN - 1);
            const uint32_t shift = (o & 3) * 8;
            const uint32_t mask = 0xffu << shift;
            pwWords[o >> 2] = (pwWords[o >> 2] & ~mask) |
                              ((uint32_t)s_charset[0] << shift);
        }
#pragma unroll
        for (int p = LEN - 2; p >= 0; p--)
        {
            d = dig[p] + 1;
            if (d < clen)
            {
                dig[p] = d;
                const int o = 1 + p;
                const uint32_t shift = (o & 3) * 8;
                const uint32_t mask = 0xffu << shift;
                pwWords[o >> 2] = (pwWords[o >> 2] & ~mask) |
                                  ((uint32_t)s_charset[d] << shift);
                break;
            }
            dig[p] = 0;
            const int o = 1 + p;
            const uint32_t shift = (o & 3) * 8;
            const uint32_t mask = 0xffu << shift;
            pwWords[o >> 2] = (pwWords[o >> 2] & ~mask) |
                              ((uint32_t)s_charset[0] << shift);
        }
    }
}

// ---- ディスパッチ(長さ→特殊化カーネル) --------------------------------------
static void launch(int len, unsigned int blocks, uint64_t off, uint64_t count, int batch, int clen, uint32_t earlyA, uint32_t ta, uint32_t tb, uint32_t tc, uint32_t td, int *d_found, unsigned char *d_result)
{
#define LAUNCH(L)                                                                                      \
    case L:                                                                                            \
        crack<L><<<blocks, TPB>>>(off, count, batch, clen, earlyA, ta, tb, tc, td, d_found, d_result); \
        break;
    switch (len)
    {
        LAUNCH(1)
        LAUNCH(2)
        LAUNCH(3)
        LAUNCH(4)
        LAUNCH(5)
        LAUNCH(6)
        LAUNCH(7)
        LAUNCH(8)
        LAUNCH(9)
        LAUNCH(10)
        LAUNCH(11)
        LAUNCH(12)
        LAUNCH(13)
        LAUNCH(14)
        LAUNCH(15)
        LAUNCH(16)
    default:
        fprintf(stderr, "length %d exceeds MAX_PW_LEN(%d); extend the LAUNCH table.\n", len, MAX_PW_LEN);
        exit(1);
    }
#undef LAUNCH
}

// ---- hex → bytes -------------------------------------------------------------
static int hex2bytes(const char *hex, unsigned char *out, int maxlen)
{
    if (hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X'))
        hex += 2;
    int n = (int)strlen(hex);
    if (n % 2 != 0)
        return -1;
    int bytes = n / 2;
    if (bytes > maxlen)
        return -1;
    for (int i = 0; i < bytes; i++)
    {
        unsigned v;
        if (sscanf(hex + i * 2, "%2x", &v) != 1)
            return -1;
        out[i] = (unsigned char)v;
    }
    return bytes;
}

// ---- 文字集合の展開ヘルパ ----------------------------------------------------
// クラス定義(記号は空白を除く ASCII パンクチュエーション 32 種)
static const char *CLS_LOWER = "abcdefghijklmnopqrstuvwxyz";
static const char *CLS_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const char *CLS_DIGIT = "0123456789";
static const char *CLS_SYMBOL = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

static void addUnique(std::string &dst, char c)
{
    if (dst.find(c) == std::string::npos)
        dst.push_back(c);
}
static void addUniqueStr(std::string &dst, const char *s)
{
    for (; *s; ++s)
        addUnique(dst, *s);
}

// 代表1文字 → クラス全体へ展開 (--charset-classes)。
//   小文字→a-z, 大文字→A-Z, 数字→0-9, それ以外(記号)→ASCII記号一式。
//   例: "aA0!" → a-z A-Z 0-9 と全記号(計 94 文字)。
static std::string expandClasses(const std::string &reps)
{
    std::string out;
    for (unsigned char c : reps)
    {
        if (c >= 'a' && c <= 'z')
            addUniqueStr(out, CLS_LOWER);
        else if (c >= 'A' && c <= 'Z')
            addUniqueStr(out, CLS_UPPER);
        else if (c >= '0' && c <= '9')
            addUniqueStr(out, CLS_DIGIT);
        else
            addUniqueStr(out, CLS_SYMBOL);
    }
    return out;
}

// 正規表現の文字クラス風文字列 → 文字集合へ展開 (--charset-regex)。
//   対応: 範囲 X-Y, ショートハンド \d(数字) \l(小文字) \u(大文字)
//         \w(英数字+_) \p(記号) \a(全部), エスケープ \x(その文字そのもの),
//         先頭/末尾の [ ] は任意(あれば無視)。否定 [^...] は非対応。
//   戻り値 false は不正(逆順範囲など)。
static bool expandRegexCharset(const std::string &in, std::string &out)
{
    size_t i = 0, n = in.size();
    if (n >= 2 && in[0] == '[' && in[n - 1] == ']')
    {
        i = 1;
        n -= 1;
    }
    while (i < n)
    {
        char c = in[i];
        if (c == '\\' && i + 1 < n)
        {
            switch (in[i + 1])
            {
            case 'd':
                addUniqueStr(out, CLS_DIGIT);
                break;
            case 'l':
                addUniqueStr(out, CLS_LOWER);
                break;
            case 'u':
                addUniqueStr(out, CLS_UPPER);
                break;
            case 'p':
                addUniqueStr(out, CLS_SYMBOL);
                break;
            case 'w':
                addUniqueStr(out, CLS_LOWER);
                addUniqueStr(out, CLS_UPPER);
                addUniqueStr(out, CLS_DIGIT);
                addUnique(out, '_');
                break;
            case 'a':
                addUniqueStr(out, CLS_LOWER);
                addUniqueStr(out, CLS_UPPER);
                addUniqueStr(out, CLS_DIGIT);
                addUniqueStr(out, CLS_SYMBOL);
                break;
            default:
                addUnique(out, in[i + 1]);
                break; // \-, \\, \] などの実文字
            }
            i += 2;
            continue;
        }
        // 範囲 X-Y (ハイフンが両端でなく、直後に文字がある場合のみ)
        if (i + 2 < n && in[i + 1] == '-')
        {
            unsigned char lo = (unsigned char)c;
            unsigned char hi = (unsigned char)in[i + 2];
            if (lo > hi)
                return false;
            for (int v = lo; v <= hi; v++)
                addUnique(out, (char)v);
            i += 3;
            continue;
        }
        addUnique(out, c);
        i++;
    }
    return true;
}

// 最終内部状態から、パスワード語がもう現れないラウンドを逆に戻す。
// 戻り値はカーネルの早期判定に使う A レジスタ。
static uint32_t reverse_early_a(int len, const uint32_t M[16], uint32_t ta, uint32_t tb, uint32_t tc, uint32_t td)
{
    const int steps = (len <= 3) ? 49 : 56;
    uint32_t A = ta, B = tb, C = tc, D = td;
    for (int i = 63; i >= steps; i--)
    {
        const uint32_t oldD = A;
        const uint32_t oldB = C;
        const uint32_t oldC = D;
        uint32_t f;
        if (i < 16)
            f = (oldB & oldC) | (~oldB & oldD);
        else if (i < 32)
            f = (oldD & oldB) | (~oldD & oldC);
        else if (i < 48)
            f = oldB ^ oldC ^ oldD;
        else
            f = oldC ^ (oldB | ~oldD);

        const int g = md5_word_index(i);
        const uint32_t oldA = rotr32(B - oldB, MD5_S[i]) - f - MD5_K[i] - M[g];
        A = oldA;
        B = oldB;
        C = oldC;
        D = oldD;
    }
    return A;
}

int main(int argc, char **argv)
{
    // 免責事項(実行時の 1 行目に必ず表示する)。文面:
    //   本ソフトウェアは実験・研究・個人利用のみを目的とし、無保証で提供される。
    //   使用は必ず自身が管理・所有するネットワークに限ること。
    //   本ソフトウェアの使用により生じたいかなる損害についても、作者は一切の責任を負わない。
    // nvcc のデバイス側フロントエンド(cicc)は日本語の生文字列リテラルを正しく解釈できず
    // 閉じ引用符を食う(-Xcompiler /utf-8 はホスト cl.exe にしか効かない)。そのため文面は
    // UTF-8 バイトを \x で直接埋め込み、ソースを純 ASCII に保つ。Windows では出力の間だけ
    // コンソールを UTF-8 に切り替え、直後に元の文字コードへ戻す。
#ifdef _WIN32
    UINT prevOutputCP = GetConsoleOutputCP();
    SetConsoleOutputCP(CP_UTF8);
#endif
    fprintf(stderr,
            "\xe6\x9c\xac\xe3\x82\xbd\xe3\x83\x95\xe3\x83\x88\xe3\x82\xa6\xe3\x82\xa7\xe3\x82\xa2\xe3\x81\xaf\xe5\xae\x9f\xe9\xa8\x93\xe3\x83\xbb\xe7\xa0\x94\xe7\xa9\xb6\xe3\x83\xbb\xe5\x80\x8b\xe4\xba\xba\xe5\x88\xa9\xe7\x94\xa8\xe3\x81\xae\xe3\x81\xbf\xe3\x82\x92\xe7\x9b\xae\xe7\x9a\x84\xe3\x81\xa8\xe3\x81\x97\xe3\x80\x81\xe7\x84\xa1\xe4\xbf\x9d\xe8\xa8\xbc\xe3\x81\xa7\xe6\x8f\x90\xe4\xbe\x9b\xe3\x81\x95\xe3\x82\x8c\xe3\x82\x8b\xe3\x80\x82"
            "\xe4\xbd\xbf\xe7\x94\xa8\xe3\x81\xaf\xe5\xbf\x85\xe3\x81\x9a\xe8\x87\xaa\xe8\xba\xab\xe3\x81\x8c\xe7\xae\xa1\xe7\x90\x86\xe3\x83\xbb\xe6\x89\x80\xe6\x9c\x89\xe3\x81\x99\xe3\x82\x8b\xe3\x83\x8d\xe3\x83\x83\xe3\x83\x88\xe3\x83\xaf\xe3\x83\xbc\xe3\x82\xaf\xe3\x81\xab\xe9\x99\x90\xe3\x82\x8b\xe3\x81\x93\xe3\x81\xa8\xe3\x80\x82"
            "\xe6\x9c\xac\xe3\x82\xbd\xe3\x83\x95\xe3\x83\x88\xe3\x82\xa6\xe3\x82\xa7\xe3\x82\xa2\xe3\x81\xae\xe4\xbd\xbf\xe7\x94\xa8\xe3\x81\xab\xe3\x82\x88\xe3\x82\x8a\xe7\x94\x9f\xe3\x81\x98\xe3\x81\x9f\xe3\x81\x84\xe3\x81\x8b\xe3\x81\xaa\xe3\x82\x8b\xe6\x90\x8d\xe5\xae\xb3\xe3\x81\xab\xe3\x81\xa4\xe3\x81\x84\xe3\x81\xa6\xe3\x82\x82\xe3\x80\x81\xe4\xbd\x9c\xe8\x80\x85\xe3\x81\xaf\xe4\xb8\x80\xe5\x88\x87\xe3\x81\xae\xe8\xb2\xac\xe4\xbb\xbb\xe3\x82\x92\xe8\xb2\xa0\xe3\x82\x8f\xe3\x81\xaa\xe3\x81\x84\xe3\x80\x82\n");
    fflush(stderr);
#ifdef _WIN32
    SetConsoleOutputCP(prevOutputCP);
#endif

    // ---- 既定パラメータ(id/challenge/target は既定なし。必須) ----
    std::string charset = "abcdefghijklmnopqrstuvwxyz0123456789";
    int minLen = 1, maxLen = 8;
    int id = -1;              // 必須: --id <0..255>
    std::string challengeHex; // 必須: --challenge <hex>
    std::string targetHex;    // 探索モードでは必須: --target <md5hex>
    const char *testPw = nullptr;
    int batch = 512;             // 1スレッドあたりの候補数(除算/起動オーバーヘッドの償却量)
    int gpusRequested = 0;       // >0 なら先頭 N 基を使用(既定 0 = 全基)
    std::vector<int> deviceList; // --devices で明示指定した GPU 番号

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--charset") && i + 1 < argc)
            charset = argv[++i];
        else if (!strcmp(argv[i], "--charset-regex") && i + 1 < argc)
        {
            std::string ex;
            if (!expandRegexCharset(argv[++i], ex) || ex.empty())
            {
                fprintf(stderr, "bad --charset-regex (invalid range or empty result)\n");
                return 1;
            }
            charset = ex;
        }
        else if (!strcmp(argv[i], "--charset-classes") && i + 1 < argc)
        {
            std::string ex = expandClasses(argv[++i]);
            if (ex.empty())
            {
                fprintf(stderr, "bad --charset-classes (empty)\n");
                return 1;
            }
            charset = ex;
        }
        else if (!strcmp(argv[i], "--min") && i + 1 < argc)
            minLen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max") && i + 1 < argc)
            maxLen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--id") && i + 1 < argc)
            id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--challenge") && i + 1 < argc)
            challengeHex = argv[++i];
        else if (!strcmp(argv[i], "--target") && i + 1 < argc)
            targetHex = argv[++i];
        else if (!strcmp(argv[i], "--test") && i + 1 < argc)
            testPw = argv[++i];
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc)
            batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpus") && i + 1 < argc)
            gpusRequested = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--devices") && i + 1 < argc)
        {
            // "0,2,3" のようなカンマ区切りを解析
            const char *p = argv[++i];
            while (*p)
            {
                char *end;
                long v = strtol(p, &end, 10);
                if (end == p)
                    break;
                deviceList.push_back((int)v);
                p = (*end == ',') ? end + 1 : end;
            }
        }
        else
        {
            fprintf(stderr, "unknown/incomplete arg: %s\n", argv[i]);
            return 1;
        }
    }

    // charset の重複を除去(展開・手動指定いずれでも無駄打ちと重複報告を防ぐ)
    {
        std::string uniq;
        for (char c : charset)
            addUnique(uniq, c);
        charset = uniq;
    }

    if (id < 0 || id > 255)
    {
        fprintf(stderr, "provide --id <0..255> (Identifier from the CHAP capture)\n");
        return 1;
    }
    if (challengeHex.empty())
    {
        fprintf(stderr, "provide --challenge <hex> (Challenge value from the CHAP capture)\n");
        return 1;
    }

    unsigned char chal[MAX_CHAL_LEN];
    int chlen = hex2bytes(challengeHex.c_str(), chal, MAX_CHAL_LEN);
    if (chlen < 0)
    {
        fprintf(stderr, "bad --challenge hex\n");
        return 1;
    }

    // --target は探索モードでは必須、検算(--test)モードでは任意。
    unsigned char tgt[16] = {0};
    const bool haveTarget = !targetHex.empty();
    uint32_t ta = 0, tb = 0, tc = 0, td = 0;
    if (haveTarget)
    {
        if (hex2bytes(targetHex.c_str(), tgt, 16) != 16)
        {
            fprintf(stderr, "bad --target (need 16 bytes)\n");
            return 1;
        }
        uint32_t t0 = (uint32_t)tgt[0] | (uint32_t)tgt[1] << 8 |
                      (uint32_t)tgt[2] << 16 | (uint32_t)tgt[3] << 24;
        uint32_t t1 = (uint32_t)tgt[4] | (uint32_t)tgt[5] << 8 |
                      (uint32_t)tgt[6] << 16 | (uint32_t)tgt[7] << 24;
        uint32_t t2 = (uint32_t)tgt[8] | (uint32_t)tgt[9] << 8 |
                      (uint32_t)tgt[10] << 16 | (uint32_t)tgt[11] << 24;
        uint32_t t3 = (uint32_t)tgt[12] | (uint32_t)tgt[13] << 8 |
                      (uint32_t)tgt[14] << 16 | (uint32_t)tgt[15] << 24;
        ta = t0 - 0x67452301u;
        tb = t1 - 0xefcdab89u;
        tc = t2 - 0x98badcfeu;
        td = t3 - 0x10325476u;
    }

    // ---- 検算モード(GPU 不要): --test <password> ----
    if (testPw)
    {
        int L = (int)strlen(testPw);
        if (1 + L + chlen > 55)
        {
            fprintf(stderr, "message too long for single-block MD5\n");
            return 1;
        }
        unsigned char buf[64];
        int p = 0;
        buf[p++] = (unsigned char)id;
        for (int i = 0; i < L; i++)
            buf[p++] = (unsigned char)testPw[i];
        for (int j = 0; j < chlen; j++)
            buf[p++] = chal[j];
        uint32_t out[4];
        md5_single(buf, p, out);
        unsigned char dg[16];
        for (int i = 0; i < 4; i++)
        {
            dg[i * 4 + 0] = out[i] & 0xff;
            dg[i * 4 + 1] = (out[i] >> 8) & 0xff;
            dg[i * 4 + 2] = (out[i] >> 16) & 0xff;
            dg[i * 4 + 3] = (out[i] >> 24) & 0xff;
        }
        printf("password : %s\n", testPw);
        printf("MD5      : ");
        for (int i = 0; i < 16; i++)
            printf("%02x", dg[i]);
        printf("\n");
        if (haveTarget)
        {
            printf("target   : %s\n", targetHex.c_str());
            printf("match    : %s\n", memcmp(dg, tgt, 16) == 0 ? "YES" : "no");
        }
        return 0;
    }

    if ((int)charset.size() == 0)
    {
        fprintf(stderr, "empty charset\n");
        return 1;
    }
    if ((int)charset.size() > 128)
    {
        fprintf(stderr, "charset too large (<=128)\n");
        return 1;
    }
    if (!haveTarget)
    {
        fprintf(stderr, "provide --target <md5hex> (the CHAP Response to crack)\n");
        return 1;
    }
    if (minLen < 1 || minLen > maxLen)
    {
        fprintf(stderr, "need 1 <= --min <= --max\n");
        return 1;
    }
    if (maxLen > MAX_PW_LEN)
    {
        fprintf(stderr, "max length too large for fast path (<= %d)\n", MAX_PW_LEN);
        return 1;
    }
    if (1 + maxLen + chlen > 55)
    {
        fprintf(stderr, "message would exceed single-block MD5 (id+pw+challenge > 55)\n");
        return 1;
    }
    if (batch < 1)
        batch = 1;

    int clen = (int)charset.size();

    // ---- 使用する GPU を決定 ----
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount < 1)
    {
        fprintf(stderr, "no CUDA device found\n");
        return 1;
    }
    std::vector<int> devices;
    if (!deviceList.empty())
        devices = deviceList; // --devices が最優先
    else
    {
        int n = (gpusRequested > 0) ? gpusRequested : devCount;
        if (n > devCount)
            n = devCount;
        for (int i = 0; i < n; i++)
            devices.push_back(i);
    }
    for (int d : devices)
    {
        if (d < 0 || d >= devCount)
        {
            fprintf(stderr, "device index %d out of range (0..%d)\n", d, devCount - 1);
            return 1;
        }
    }
    const int numDev = (int)devices.size();

    printf("charset size = %d, length %d..%d, id=%d, challenge=%d bytes, batch=%d\n", clen, minLen, maxLen, id, chlen, batch);
    printf("charset = \"%s\"\n", charset.c_str());
    printf("using %d GPU(s):", numDev);
    for (int d : devices)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, d) == cudaSuccess)
            printf(" [%d]%s", d, prop.name);
        else
            printf(" [%d]", d);
    }
    printf("\n");

    const uint64_t CHUNK = 1ULL << 30; // 1カーネル起動あたりの候補数(進捗/停止の粒度)
    auto t_start = std::chrono::high_resolution_clock::now();
    std::atomic<uint64_t> g_tried{0}; // 全 GPU で試行した候補数の合計
    std::atomic<int> g_found{0};      // 最初の発見者が 1 に立てる
    bool solved = false;
    unsigned char result[MAX_PW_LEN + 1] = {0};

    for (int len = minLen; len <= maxLen && !solved; len++)
    {
        // 定数部分を len ごとに構築: ID || (pw領域=0) || challenge || pad || bitlen
        uint32_t Mt[16] = {0};
        auto putb = [&](int k, unsigned char v)
        { Mt[k >> 2] |= (uint32_t)v << ((k & 3) * 8); };
        putb(0, (unsigned char)id);
        for (int j = 0; j < chlen; j++)
            putb(1 + len + j, chal[j]);
        int msgBytes = 1 + len + chlen;
        putb(msgBytes, 0x80);
        Mt[14] = (uint32_t)(msgBytes * 8);

        uint32_t RK[64];
        for (int i = 0; i < 64; i++)
            RK[i] = MD5_K[i] + Mt[md5_word_index(i)];
        const uint32_t earlyA = (len <= 7) ? reverse_early_a(len, Mt, ta, tb, tc, td) : 0;

        // keyspace = clen^len (オーバーフロー検出)
        uint64_t total = 1;
        bool ovf = false;
        for (int i = 0; i < len; i++)
        {
            if (total > (UINT64_MAX / (uint64_t)clen))
            {
                ovf = true;
                break;
            }
            total *= (uint64_t)clen;
        }
        if (ovf)
        {
            fprintf(stderr, "keyspace for len=%d overflows 64-bit; stop.\n", len);
            break;
        }

        printf("[len=%d] keyspace = %llu\n", len, (unsigned long long)total);

        // この長さのキースペースを CHUNK 単位で全 GPU に動的配分する。
        // 各 GPU スレッドが g_next から次のチャンクを atomic に取得するため、
        // 速度の異なる GPU が混在しても自動的に負荷分散される。
        std::atomic<uint64_t> g_next{0};
        std::vector<std::thread> pool;
        pool.reserve(numDev);

        for (int ti = 0; ti < numDev; ti++)
        {
            const int dev = devices[ti];
            pool.emplace_back([&, dev, ti]()
                              {
                CUDA_CHECK(cudaSetDevice(dev));
                // 定数メモリはデバイスごとに存在するので各自コピーする
                CUDA_CHECK(cudaMemcpyToSymbol(d_charset, charset.data(), clen));
                CUDA_CHECK(cudaMemcpyToSymbol(d_RK, RK, sizeof(RK)));

                int *d_found = nullptr;
                unsigned char *d_result = nullptr;
                CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_result, MAX_PW_LEN + 1));
                CUDA_CHECK(cudaMemset(d_found, 0, sizeof(int)));

                for (;;)
                {
                    if (g_found.load(std::memory_order_relaxed))
                        break;
                    uint64_t off = g_next.fetch_add(CHUNK);
                    if (off >= total)
                        break;

                    uint64_t count = (total - off < CHUNK) ? (total - off) : CHUNK;
                    uint64_t threads = (count + (uint64_t)batch - 1) / (uint64_t)batch;
                    uint64_t blocks = (threads + TPB - 1) / TPB;

                    launch(len, (unsigned int)blocks, off, count, batch, clen, earlyA, ta, tb, tc, td, d_found, d_result);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());

                    int found = 0;
                    CUDA_CHECK(cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
                    uint64_t tried = g_tried.fetch_add(count) + count;

                    if (found)
                    {
                        // 最初の発見者だけが結果を確定させる
                        if (g_found.exchange(1) == 0)
                            CUDA_CHECK(cudaMemcpy(result, d_result, MAX_PW_LEN + 1, cudaMemcpyDeviceToHost));
                        break;
                    }

                    if (ti == 0) // 進捗は先頭スレッドだけが集計表示する
                    {
                        auto now = std::chrono::high_resolution_clock::now();
                        double sec = std::chrono::duration<double>(now - t_start).count();
                        double rate = sec > 0 ? tried / sec / 1e6 : 0;
                        printf("  tried %.3e   %.1f Mh/s\r", (double)tried, rate);
                        fflush(stdout);
                    }
                }

                cudaFree(d_found);
                cudaFree(d_result); });
        }

        for (auto &th : pool)
            th.join();

        if (g_found.load())
            solved = true;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t_end - t_start).count();
    uint64_t tried = g_tried.load();
    printf("\n");
    if (solved)
        printf("[FOUND] password = \"%s\"\n", result);
    else
        printf("[NOT FOUND] in given keyspace\n");
    printf("tried %.4e candidates in %.2f s (%.1f Mh/s)\n", (double)tried, sec, sec > 0 ? tried / sec / 1e6 : 0);

    return 0;
}
