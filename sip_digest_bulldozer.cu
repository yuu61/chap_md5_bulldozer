/*
============================================================================
本ソフトウェアは実験・研究・個人利用のみを目的とし、無保証で提供される。
使用は必ず自身が管理・所有するネットワークに限ること。
本ソフトウェアの使用により生じたいかなる損害についても、作者は一切の責任を負わない。
============================================================================

SIP Digest 認証 (RFC 2617 / RFC 3261, algorithm=MD5) のパスワードを CUDA で総当たり復元する。
CHAP 版 (chap_md5_bulldozer.cu) と同じハーネス(charset展開/マルチGPU/オドメーター/CLI)を流用し、
コア計算だけを SIP Digest 用に差し替えたもの。

  HA1      = MD5( username ":" realm ":" password )            ← 探索対象。password は HA1 の末尾
  HA2      = MD5( method ":" uri )                             ← 定数(ホストで事前計算)
  response = MD5( HA1 ":" nonce ":" nc ":" cnonce ":" qop ":" HA2 )   (qop=auth)
           = MD5( HA1 ":" nonce ":" HA2 )                             (qop 無し)
  ※ HA1/HA2 は 32 文字の小文字 hex 文字列として連結する。
  ※ nonce/nc/cnonce/qop/realm/username/uri/method はすべて "そのままの ASCII 文字列" として使う
     (nonce が hex に見えてもデコードしない)。--response のみバイト列の hex。

1候補あたり MD5 圧縮は HA1(1ブロック) + response(2〜3ブロック) = 計3〜4回。CHAP(1回)より重く、
概ね 1/4 程度のハッシュレートになる。HA1 は単一ブロック前提(prefix長 + max長 <= 55)に限定。

windows:  nvcc -O3 -arch=native -Xcompiler /utf-8 sip_digest_bulldozer.cu -o sip_digest_bulldozer.exe
Linux:    nvcc -O3 -arch=native sip_digest_bulldozer.cu -o sip_digest_bulldozer
(-arch=native は実行ホストの GPU 世代を自動選択。/utf-8 は日本語コメントを MSVC が CP932 誤読するのを防ぐため必須)

実行例(voip.pcapng の REGISTER をクラック):
./sip_digest_bulldozer --username GNK14KB849 --realm ims.eonet.ne.jp \
    --method REGISTER --uri sip:10.255.232.181 \
    --nonce 727249E3B472796A00000000F8916B78 --qop auth --nc 00000001 --cnonce 7e8b6147 \
    --response e3f39edcdc5ce6a3f775bc15a85e69f2 \
    --charset "abcdefghijklmnopqrstuvwxyz0123456789" --min 1 --max 8

特定パスワードの検算(GPU不要):
./sip_digest_bulldozer --username ... --realm ... --method ... --uri ... --nonce ... \
    [--qop auth --nc ... --cnonce ...] --test <password> [--response <md5hex>]

charset 指定は CHAP 版と共通: --charset / --charset-regex / --charset-classes / --min / --max /
--batch / --gpus / --devices。
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cuda_runtime.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h> // SetConsoleOutputCP: 免責文(UTF-8)をコンソールで正しく表示するため
#endif

#define MAX_PW_LEN 16  // 特殊化するパスワード長の上限(dispatch テーブルと連動)
#define MAX_RESP_BLK 3 // response メッセージの最大ブロック数(qop=auth・標準nonceで 3)
#define TPB 256

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

// ---- MD5 基本ブロック(単一512bitブロックを処理し state を更新) ----------------
__host__ __device__ __forceinline__ uint32_t rotl32(uint32_t x, int c)
{
    return (x << c) | (x >> (32 - c));
}

// state(st) を初期値または前ブロックの結果とし、M(16語)で1ブロック分の圧縮を行い加算する。
// ループは #pragma unroll で完全展開され、g(語添字)がコンパイル時定数になるため
// M[16] がレジスタに載る(呼び出し側はレジスタ配列を渡すこと)。
__host__ __device__ __forceinline__ void md5_block(uint32_t st[4], const uint32_t M[16])
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

    uint32_t A = st[0], B = st[1], C = st[2], D = st[3];
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
    st[0] += A;
    st[1] += B;
    st[2] += C;
    st[3] += D;
}

// ---- ホスト用 汎用 MD5(任意長・複数ブロック) --------------------------------
static void md5_bytes(const unsigned char *msg, size_t len, uint32_t out[4])
{
    size_t nblk = (len + 8) / 64 + 1;
    std::vector<uint32_t> M(nblk * 16, 0);
    for (size_t i = 0; i < len; i++)
        M[i >> 2] |= (uint32_t)msg[i] << ((i & 3) * 8);
    M[len >> 2] |= (uint32_t)0x80 << ((len & 3) * 8);
    uint64_t bits = (uint64_t)len * 8;
    M[nblk * 16 - 2] = (uint32_t)(bits & 0xffffffffULL);
    M[nblk * 16 - 1] = (uint32_t)(bits >> 32);
    uint32_t st[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    for (size_t b = 0; b < nblk; b++)
        md5_block(st, &M[b * 16]);
    out[0] = st[0];
    out[1] = st[1];
    out[2] = st[2];
    out[3] = st[3];
}

// 文字列 → MD5 の 32 文字小文字 hex
static std::string md5_hex(const std::string &s)
{
    uint32_t o[4];
    md5_bytes((const unsigned char *)s.data(), s.size(), o);
    unsigned char dg[16];
    for (int i = 0; i < 4; i++)
    {
        dg[i * 4 + 0] = o[i] & 0xff;
        dg[i * 4 + 1] = (o[i] >> 8) & 0xff;
        dg[i * 4 + 2] = (o[i] >> 16) & 0xff;
        dg[i * 4 + 3] = (o[i] >> 24) & 0xff;
    }
    static const char *H = "0123456789abcdef";
    std::string r(32, '0');
    for (int i = 0; i < 16; i++)
    {
        r[i * 2 + 0] = H[dg[i] >> 4];
        r[i * 2 + 1] = H[dg[i] & 15];
    }
    return r;
}

// ---- 定数メモリ --------------------------------------------------------------
__constant__ unsigned char d_charset[128];             // 文字集合
__constant__ uint32_t d_ha1_base[16];                  // HA1 メッセージ語(password 領域=0, pad/bitlen 込み・単一ブロック)
__constant__ int d_pw_word[MAX_PW_LEN];                // password 文字 i → HA1 メッセージの語添字
__constant__ int d_pw_shift[MAX_PW_LEN];               // password 文字 i → その語内シフト量
__constant__ uint32_t d_resp_words[MAX_RESP_BLK * 16]; // response メッセージ語(先頭32B=HA1hex 領域=0, pad/bitlen 込み)

// HA1(16バイト) → 32文字の小文字hex → 8語(リトルエンディアン詰め)
__device__ __forceinline__ void ha1_to_words(const uint32_t h1[4], uint32_t hx[8])
{
    unsigned char hb[32];
#pragma unroll
    for (int j = 0; j < 16; j++)
    {
        unsigned char v = (h1[j >> 2] >> ((j & 3) * 8)) & 0xff;
        unsigned char hi = v >> 4, lo = v & 15;
        hb[j * 2 + 0] = hi < 10 ? ('0' + hi) : ('a' + hi - 10);
        hb[j * 2 + 1] = lo < 10 ? ('0' + lo) : ('a' + lo - 10);
    }
#pragma unroll
    for (int w = 0; w < 8; w++)
        hx[w] = (uint32_t)hb[w * 4 + 0] | (uint32_t)hb[w * 4 + 1] << 8 |
                (uint32_t)hb[w * 4 + 2] << 16 | (uint32_t)hb[w * 4 + 3] << 24;
}

// ---- 特殊化カーネル ----------------------------------------------------------
// LEN  : パスワード長(コンパイル時定数)
// RBLK : response メッセージのブロック数(2 or 3, コンパイル時定数)
template <int LEN, int RBLK>
__global__ void __launch_bounds__(TPB) crack(
    uint64_t offset, uint64_t count, int batch, int clen, uint32_t ta, uint32_t tb, uint32_t tc, uint32_t td, int *found, unsigned char *result)
{
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

    // オドメーター初期化: start を base-clen 展開(スレッドあたり1回だけ除算)
    int dig[LEN];
    {
        uint64_t code = start;
#pragma unroll
        for (int i = LEN - 1; i >= 0; i--)
        {
            dig[i] = (int)(code % (uint64_t)clen);
            code /= (uint64_t)clen;
        }
    }

    for (int n = 0; n < work; n++)
    {
        // --- HA1 = MD5(username:realm:password) ---
        // 固定部(prefix+pad+bitlen)を base からコピーし、password バイトを重ねる。
        uint32_t M[16];
#pragma unroll
        for (int i = 0; i < 16; i++)
            M[i] = d_ha1_base[i];
#pragma unroll
        for (int i = 0; i < LEN; i++)
            M[d_pw_word[i]] |= (uint32_t)s_charset[dig[i]] << d_pw_shift[i];
        // 展開後のメッセージ語をレジスタ配列へ移し、圧縮の 64 段をレジスタで回す
        uint32_t Mr[16];
#pragma unroll
        for (int i = 0; i < 16; i++)
            Mr[i] = M[i];
        uint32_t h1[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
        md5_block(h1, Mr);

        // --- HA1 を hex 化して response メッセージの先頭 8 語へ ---
        uint32_t hx[8];
        ha1_to_words(h1, hx);

        // --- response = MD5(HA1hex : nonce : [nc:cnonce:qop:] HA2hex) ---
        uint32_t R[RBLK * 16];
#pragma unroll
        for (int i = 0; i < RBLK * 16; i++)
            R[i] = d_resp_words[i];
#pragma unroll
        for (int i = 0; i < 8; i++)
            R[i] = hx[i];

        uint32_t st2[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
#pragma unroll
        for (int b = 0; b < RBLK; b++)
        {
            uint32_t Mb[16];
#pragma unroll
            for (int i = 0; i < 16; i++)
                Mb[i] = R[b * 16 + i];
            md5_block(st2, Mb);
        }

        if (st2[0] == ta && st2[1] == tb && st2[2] == tc && st2[3] == td)
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

        // オドメーター +1
        bool carry = true;
#pragma unroll
        for (int p = LEN - 1; p >= 0; p--)
        {
            if (carry)
            {
                int d = dig[p] + 1;
                if (d < clen)
                    carry = false;
                else
                    d = 0;
                dig[p] = d;
            }
        }
    }
}

// ---- ディスパッチ(長さ・ブロック数 → 特殊化カーネル) ------------------------
static void launch(int len, int rblk, unsigned int blocks, uint64_t off, uint64_t count, int batch, int clen, uint32_t ta, uint32_t tb, uint32_t tc, uint32_t td, int *d_found, unsigned char *d_result)
{
#define LAUNCH(L)                                                                                     \
    case L:                                                                                           \
        if (rblk == 2)                                                                                \
            crack<L, 2><<<blocks, TPB>>>(off, count, batch, clen, ta, tb, tc, td, d_found, d_result); \
        else                                                                                          \
            crack<L, 3><<<blocks, TPB>>>(off, count, batch, clen, ta, tb, tc, td, d_found, d_result); \
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

// ---- 文字集合の展開ヘルパ(CHAP 版と共通) ------------------------------------
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
                break;
            }
            i += 2;
            continue;
        }
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

int main(int argc, char **argv)
{
#ifdef _WIN32
    UINT prevOutputCP = GetConsoleOutputCP();
    SetConsoleOutputCP(CP_UTF8);
#endif
    // 免責文(UTF-8 バイトを直接埋め込み。cicc は日本語の生リテラルを解釈できないため)
    fprintf(stderr,
            "\xe6\x9c\xac\xe3\x82\xbd\xe3\x83\x95\xe3\x83\x88\xe3\x82\xa6\xe3\x82\xa7\xe3\x82\xa2\xe3\x81\xaf\xe5\xae\x9f\xe9\xa8\x93\xe3\x83\xbb\xe7\xa0\x94\xe7\xa9\xb6\xe3\x83\xbb\xe5\x80\x8b\xe4\xba\xba\xe5\x88\xa9\xe7\x94\xa8\xe3\x81\xae\xe3\x81\xbf\xe3\x82\x92\xe7\x9b\xae\xe7\x9a\x84\xe3\x81\xa8\xe3\x81\x97\xe3\x80\x81\xe7\x84\xa1\xe4\xbf\x9d\xe8\xa8\xbc\xe3\x81\xa7\xe6\x8f\x90\xe4\xbe\x9b\xe3\x81\x95\xe3\x82\x8c\xe3\x82\x8b\xe3\x80\x82"
            "\xe4\xbd\xbf\xe7\x94\xa8\xe3\x81\xaf\xe5\xbf\x85\xe3\x81\x9a\xe8\x87\xaa\xe8\xba\xab\xe3\x81\x8c\xe7\xae\xa1\xe7\x90\x86\xe3\x83\xbb\xe6\x89\x80\xe6\x9c\x89\xe3\x81\x99\xe3\x82\x8b\xe3\x83\x8d\xe3\x83\x83\xe3\x83\x88\xe3\x83\xaf\xe3\x83\xbc\xe3\x82\xaf\xe3\x81\xab\xe9\x99\x90\xe3\x82\x8b\xe3\x81\x93\xe3\x81\xa8\xe3\x80\x82"
            "\xe6\x9c\xac\xe3\x82\xbd\xe3\x83\x95\xe3\x83\x88\xe3\x82\xa6\xe3\x82\xa7\xe3\x82\xa2\xe3\x81\xae\xe4\xbd\xbf\xe7\x94\xa8\xe3\x81\xab\xe3\x82\x88\xe3\x82\x8a\xe7\x94\x9f\xe3\x81\x98\xe3\x81\x9f\xe3\x81\x84\xe3\x81\x8b\xe3\x81\xaa\xe3\x82\x8b\xe6\x90\x8d\xe5\xae\xb3\xe3\x81\xab\xe3\x81\xa4\xe3\x81\x84\xe3\x81\xa6\xe3\x82\x82\xe3\x80\x81\xe4\xbd\x9c\xe8\x80\x85\xe3\x81\xaf\xe4\xb8\x80\xe5\x88\x87\xe3\x81\xae\xe8\xb2\xac\xe4\xbb\xbb\xe3\x82\x92\xe8\xb2\xa0\xe3\x82\x8f\xe3\x81\xaa\xe3\x81\x84\xe3\x80\x82\n");
    fflush(stderr);
#ifdef _WIN32
    SetConsoleOutputCP(prevOutputCP);
#endif

    // ---- パラメータ ----
    std::string charset = "abcdefghijklmnopqrstuvwxyz0123456789";
    int minLen = 1, maxLen = 8;
    std::string username, realm, nonce, uri, method, qop, nc, cnonce, algorithm = "MD5";
    std::string responseHex; // 探索モードで必須: --response
    const char *testPw = nullptr;
    int batch = 512;
    int gpusRequested = 0;
    std::vector<int> deviceList;

    for (int i = 1; i < argc; i++)
    {
        auto need = [&](const char *f)
        { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", f); exit(1); } return argv[++i]; };
        if (!strcmp(argv[i], "--charset"))
            charset = need("--charset");
        else if (!strcmp(argv[i], "--charset-regex"))
        {
            std::string ex;
            if (!expandRegexCharset(need("--charset-regex"), ex) || ex.empty())
            {
                fprintf(stderr, "bad --charset-regex (invalid range or empty result)\n");
                return 1;
            }
            charset = ex;
        }
        else if (!strcmp(argv[i], "--charset-classes"))
        {
            std::string ex = expandClasses(need("--charset-classes"));
            if (ex.empty())
            {
                fprintf(stderr, "bad --charset-classes (empty)\n");
                return 1;
            }
            charset = ex;
        }
        else if (!strcmp(argv[i], "--min"))
            minLen = atoi(need("--min"));
        else if (!strcmp(argv[i], "--max"))
            maxLen = atoi(need("--max"));
        else if (!strcmp(argv[i], "--username"))
            username = need("--username");
        else if (!strcmp(argv[i], "--realm"))
            realm = need("--realm");
        else if (!strcmp(argv[i], "--nonce"))
            nonce = need("--nonce");
        else if (!strcmp(argv[i], "--uri"))
            uri = need("--uri");
        else if (!strcmp(argv[i], "--method"))
            method = need("--method");
        else if (!strcmp(argv[i], "--qop"))
            qop = need("--qop");
        else if (!strcmp(argv[i], "--nc"))
            nc = need("--nc");
        else if (!strcmp(argv[i], "--cnonce"))
            cnonce = need("--cnonce");
        else if (!strcmp(argv[i], "--algorithm"))
            algorithm = need("--algorithm");
        else if (!strcmp(argv[i], "--response"))
            responseHex = need("--response");
        else if (!strcmp(argv[i], "--test"))
            testPw = need("--test");
        else if (!strcmp(argv[i], "--batch"))
            batch = atoi(need("--batch"));
        else if (!strcmp(argv[i], "--gpus"))
            gpusRequested = atoi(need("--gpus"));
        else if (!strcmp(argv[i], "--devices"))
        {
            const char *p = need("--devices");
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
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    // charset 重複除去
    {
        std::string uniq;
        for (char c : charset)
            addUnique(uniq, c);
        charset = uniq;
    }

    // algorithm=MD5 のみ対応(MD5-sess は HA1 の定義が変わるため未対応)
    if (algorithm != "MD5" && algorithm != "md5")
    {
        fprintf(stderr, "only algorithm=MD5 is supported (got '%s'). MD5-sess is not implemented.\n", algorithm.c_str());
        return 1;
    }
    if (username.empty() || realm.empty() || nonce.empty() || uri.empty() || method.empty())
    {
        fprintf(stderr, "provide --username --realm --nonce --uri --method (all required)\n");
        return 1;
    }
    const bool useQop = !qop.empty();
    if (useQop && (nc.empty() || cnonce.empty()))
    {
        fprintf(stderr, "--qop given but --nc/--cnonce missing\n");
        return 1;
    }

    // HA2 = MD5(method:uri) は定数
    const std::string HA2 = md5_hex(method + ":" + uri);
    const std::string prefix = username + ":" + realm + ":"; // HA1 の password 直前まで
    const int prefixLen = (int)prefix.size();

    // response メッセージ(先頭32B=HA1hex は placeholder)を組み立てる
    std::string tail = std::string(":") + nonce + ":";
    if (useQop)
        tail += nc + ":" + cnonce + ":" + qop + ":";
    tail += HA2;
    const int respLen = 32 + (int)tail.size();
    const int rblk = (int)((respLen + 8) / 64 + 1);

    // ---- 検算モード ----
    if (testPw)
    {
        std::string pw = testPw;
        if (prefixLen + (int)pw.size() > 55)
        {
            fprintf(stderr, "HA1 message exceeds single block (prefix %d + pw %d > 55)\n", prefixLen, (int)pw.size());
            return 1;
        }
        std::string HA1 = md5_hex(prefix + pw);
        std::string resp = md5_hex(HA1 + tail);
        printf("password : %s\n", testPw);
        printf("HA1      : %s\n", HA1.c_str());
        printf("HA2      : %s\n", HA2.c_str());
        printf("response : %s\n", resp.c_str());
        if (!responseHex.empty())
        {
            printf("target   : %s\n", responseHex.c_str());
            printf("match    : %s\n", resp == responseHex ? "YES" : "no");
        }
        return 0;
    }

    // ---- 探索モードの検証 ----
    if (charset.empty())
    {
        fprintf(stderr, "empty charset\n");
        return 1;
    }
    if ((int)charset.size() > 128)
    {
        fprintf(stderr, "charset too large (<=128)\n");
        return 1;
    }
    if (responseHex.empty())
    {
        fprintf(stderr, "provide --response <md5hex> (the SIP Digest response to crack)\n");
        return 1;
    }
    unsigned char tgt[16];
    if (hex2bytes(responseHex.c_str(), tgt, 16) != 16)
    {
        fprintf(stderr, "bad --response (need 16 bytes hex)\n");
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
    if (prefixLen + maxLen > 55)
    {
        fprintf(stderr, "HA1 would exceed single block: prefix(%d) + maxLen(%d) > 55.\n"
                        "shorten --max or split HA1 into 2 blocks (not implemented).\n",
                prefixLen, maxLen);
        return 1;
    }
    if (rblk < 2 || rblk > MAX_RESP_BLK)
    {
        fprintf(stderr, "response blocks = %d (len %d) unsupported; only 2..%d.\n"
                        "extend MAX_RESP_BLK and the LAUNCH dispatch.\n",
                rblk, respLen, MAX_RESP_BLK);
        return 1;
    }
    if (batch < 1)
        batch = 1;

    const int clen = (int)charset.size();

    // response 目標語(最終ダイジェストと直接比較する生の語)
    const uint32_t ta = (uint32_t)tgt[0] | (uint32_t)tgt[1] << 8 | (uint32_t)tgt[2] << 16 | (uint32_t)tgt[3] << 24;
    const uint32_t tb = (uint32_t)tgt[4] | (uint32_t)tgt[5] << 8 | (uint32_t)tgt[6] << 16 | (uint32_t)tgt[7] << 24;
    const uint32_t tc = (uint32_t)tgt[8] | (uint32_t)tgt[9] << 8 | (uint32_t)tgt[10] << 16 | (uint32_t)tgt[11] << 24;
    const uint32_t td = (uint32_t)tgt[12] | (uint32_t)tgt[13] << 8 | (uint32_t)tgt[14] << 16 | (uint32_t)tgt[15] << 24;

    // response メッセージ語(パスワード非依存の固定部)を1回だけ構築
    uint32_t respWords[MAX_RESP_BLK * 16] = {0};
    {
        std::vector<unsigned char> rb(respLen, 0);
        for (int j = 0; j < (int)tail.size(); j++)
            rb[32 + j] = (unsigned char)tail[j]; // 先頭32Bは 0 のまま(HA1hex 領域)
        for (int j = 0; j < respLen; j++)
            respWords[j >> 2] |= (uint32_t)rb[j] << ((j & 3) * 8);
        respWords[respLen >> 2] |= (uint32_t)0x80 << ((respLen & 3) * 8);
        uint64_t bits = (uint64_t)respLen * 8;
        respWords[rblk * 16 - 2] = (uint32_t)(bits & 0xffffffffULL);
        respWords[rblk * 16 - 1] = (uint32_t)(bits >> 32);
    }

    // ---- GPU 決定(CHAP 版と同じ) ----
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount < 1)
    {
        fprintf(stderr, "no CUDA device found\n");
        return 1;
    }
    std::vector<int> devices;
    if (!deviceList.empty())
        devices = deviceList;
    else
    {
        int n = (gpusRequested > 0) ? gpusRequested : devCount;
        if (n > devCount)
            n = devCount;
        for (int i = 0; i < n; i++)
            devices.push_back(i);
    }
    for (int d : devices)
        if (d < 0 || d >= devCount)
        {
            fprintf(stderr, "device index %d out of range (0..%d)\n", d, devCount - 1);
            return 1;
        }
    const int numDev = (int)devices.size();

    printf("charset size = %d, length %d..%d, batch=%d\n", clen, minLen, maxLen, batch);
    printf("charset = \"%s\"\n", charset.c_str());
    printf("HA2 = %s, response blocks = %d (msg %d bytes), qop=%s\n", HA2.c_str(), rblk, respLen, useQop ? qop.c_str() : "(none)");
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

    const uint64_t CHUNK = 1ULL << 30;
    auto t_start = std::chrono::high_resolution_clock::now();
    std::atomic<uint64_t> g_tried{0};
    std::atomic<int> g_found{0};
    bool solved = false;
    unsigned char result[MAX_PW_LEN + 1] = {0};

    for (int len = minLen; len <= maxLen && !solved; len++)
    {
        // HA1 メッセージ(password 領域=0)の固定語と、password 文字→(語,シフト)対応表を len ごとに構築
        uint32_t ha1base[16] = {0};
        int pwWord[MAX_PW_LEN] = {0}, pwShift[MAX_PW_LEN] = {0};
        {
            std::vector<unsigned char> hb(prefixLen + len, 0);
            for (int j = 0; j < prefixLen; j++)
                hb[j] = (unsigned char)prefix[j]; // password 領域(prefixLen..)は 0 のまま
            int hlen = prefixLen + len;
            for (int j = 0; j < hlen; j++)
                ha1base[j >> 2] |= (uint32_t)hb[j] << ((j & 3) * 8);
            ha1base[hlen >> 2] |= (uint32_t)0x80 << ((hlen & 3) * 8);
            ha1base[14] = (uint32_t)(hlen * 8); // 単一ブロック(<=55B)前提
            for (int i = 0; i < len; i++)
            {
                int off = prefixLen + i;
                pwWord[i] = off >> 2;
                pwShift[i] = (off & 3) * 8;
            }
        }

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

        std::atomic<uint64_t> g_next{0};
        std::vector<std::thread> pool;
        pool.reserve(numDev);

        for (int ti = 0; ti < numDev; ti++)
        {
            const int dev = devices[ti];
            pool.emplace_back([&, dev, ti]()
                              {
                CUDA_CHECK(cudaSetDevice(dev));
                CUDA_CHECK(cudaMemcpyToSymbol(d_charset, charset.data(), clen));
                CUDA_CHECK(cudaMemcpyToSymbol(d_ha1_base, ha1base, sizeof(ha1base)));
                CUDA_CHECK(cudaMemcpyToSymbol(d_pw_word, pwWord, sizeof(pwWord)));
                CUDA_CHECK(cudaMemcpyToSymbol(d_pw_shift, pwShift, sizeof(pwShift)));
                CUDA_CHECK(cudaMemcpyToSymbol(d_resp_words, respWords, sizeof(respWords)));

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

                    launch(len, rblk, (unsigned int)blocks, off, count, batch, clen, ta, tb, tc, td, d_found, d_result);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());

                    int found = 0;
                    CUDA_CHECK(cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
                    uint64_t tried = g_tried.fetch_add(count) + count;

                    if (found)
                    {
                        if (g_found.exchange(1) == 0)
                            CUDA_CHECK(cudaMemcpy(result, d_result, MAX_PW_LEN + 1, cudaMemcpyDeviceToHost));
                        break;
                    }
                    if (ti == 0)
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
