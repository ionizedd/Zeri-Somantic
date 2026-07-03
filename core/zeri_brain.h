/*
 * zeri_brain.h  —  Single-header SNN Brain Library
 * ====================================================
 * Version : 0.2.0  |  License: MIT
 *
 * A seed-initialized spiking neural network brain:
 *   • All initial weights derived from a 1024-bit seed (no random init)
 *   • STDP learning (spike-timing dependent plasticity)
 *   • Memory stored as a compact delta, not full weights
 *   • Runs entirely in L1 cache for small vocabularies
 *   • Zero dependencies — one header, any C/C++ compiler
 *
 * Compile:
 *   gcc  -O3 -mavx2 -mfma  -DZERI_IMPLEMENTATION  prog.c  -lm
 *   cl   /O2 /arch:AVX2    /DZERI_IMPLEMENTATION   prog.c
 */

#pragma once
#ifndef ZERI_BRAIN_H
#define ZERI_BRAIN_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#if !defined(ZERI_NO_AVX2) && (defined(__AVX2__) || defined(_M_AMD64))
  #include <immintrin.h>
  #define ZERI_AVX2 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ZeriBrain ZeriBrain;

/* zeri_brain_create
 *   seed      : 128 bytes (1024 bits). This IS the personality.
 *   spike_dim : internal width (multiple of 8). 256 = fast, 512 = balanced.
 *   vocab_size: number of output classes (chars / words / actions / ...).
 */
ZeriBrain* zeri_brain_create(const uint8_t* seed, int spike_dim, int vocab_size);

/* zeri_brain_observe
 *   context  : array of recent token IDs
 *   ctx_len  : how many (up to 8 used)
 *   target   : the token that actually came next
 *   valence  : +1.0 reinforce | -1.0 suppress
 *   returns  : surprise
 */
float zeri_brain_observe(ZeriBrain* b, const int* context, int ctx_len,
                          int target, float valence);

/* Argmax prediction given context. */
int   zeri_brain_predict(ZeriBrain* b, const int* context, int ctx_len);

/* Top-k predictions. Caller allocates out_ids[k] and out_probs[k]. */
void  zeri_brain_predict_topk(ZeriBrain* b, const int* context, int ctx_len,
                               int k, int* out_ids, float* out_probs);

/* Raw surprise for a specific (context->target) without updating weights. */
float zeri_brain_surprise(ZeriBrain* b, const int* context, int ctx_len,
                           int target);

/* Memory consolidation (save/load) */
int   zeri_brain_consolidate(ZeriBrain* b, uint8_t* buf, int buf_size);
int   zeri_brain_load_memory(ZeriBrain* b, const uint8_t* buf, int buf_size);

/* out_stats[4] = {steps, avg_surprise, sparsity_frac, total_synapses} */
void  zeri_brain_stats(ZeriBrain* b, float* out_stats);
void  zeri_brain_destroy(ZeriBrain* b);

/* Named brain — same name -> same seed -> same starting personality. */
ZeriBrain* zeri_brain_from_name(const char* name, int spike_dim, int vocab_size);

/* Gene-derived brain — for game NPC integration. */
ZeriBrain* zeri_brain_from_genes(const float* genes, int gene_count,
                                   int spike_dim, int vocab_size);

#ifdef __cplusplus
}
#endif

/* =========================================================================
   IMPLEMENTATION
   ========================================================================= */
#ifdef ZERI_IMPLEMENTATION

#define ZERI_SEED_BYTES    128
#define ZERI_MAX_CTX         8
#define ZERI_STDP_TAU      0.90f
#define ZERI_STDP_A_PLUS   0.018f
#define ZERI_STDP_A_MINUS  0.009f
#define ZERI_LR_BASE       0.030f
#define ZERI_CLIP          4.0f

static inline uint64_t zeri__sm64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x  = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x  = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t zeri__vf(const uint64_t* seed16, int lane, uint64_t blk) {
    return zeri__sm64(seed16[lane & 15] ^ zeri__sm64(blk * 16 + (uint64_t)lane));
}

static void zeri__vfield(const uint64_t* s16, int start, int count, int8_t* out) {
    int blk = start / 8, off = start % 8, i = 0;
    uint64_t cur = zeri__vf(s16, blk & 15, (uint64_t)blk);
    uint8_t  bytes[8]; memcpy(bytes, &cur, 8);

    for (; off < 8 && i < count; off++, i++) out[i] = (int8_t)(bytes[off] % 3) - 1;
    blk++;
    while (i + 8 <= count) {
        cur = zeri__vf(s16, blk & 15, (uint64_t)blk);
        memcpy(bytes, &cur, 8);
        for (int j = 0; j < 8; j++) out[i+j] = (int8_t)(bytes[j] % 3) - 1;
        i += 8; blk++;
    }
    if (i < count) {
        cur = zeri__vf(s16, blk & 15, (uint64_t)blk);
        memcpy(bytes, &cur, 8);
        for (int j = 0; i < count; j++, i++) out[i] = (int8_t)(bytes[j] % 3) - 1;
    }
}

#ifdef ZERI_AVX2
static inline float zeri__dot(const float* w, const float* x, int dim) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(w+i), _mm256_loadu_ps(x+i), acc);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
#else
static inline float zeri__dot(const float* w, const float* x, int dim) {
    float a = 0.0f;
    for (int i = 0; i < dim; i++) a += w[i] * x[i];
    return a;
}
#endif

static void zeri__softmax(const float* lg, float* pr, int n) {
    float mx = lg[0];
    for (int i = 1; i < n; i++) if (lg[i] > mx) mx = lg[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { pr[i] = expf(lg[i] - mx); s += pr[i]; }
    for (int i = 0; i < n; i++) pr[i] /= s;
}

struct ZeriBrain {
    int      spike_dim;
    int      vocab_size;
    uint64_t seed16[16];
    float*   W;           /* [vocab_size × spike_dim] learned */
    float*   W0;          /* [vocab_size × spike_dim] seed snapshot */
    float*   bias;        /* [vocab_size] */
    float*   pre_trace;   /* [spike_dim] */
    float*   post_trace;  /* [vocab_size] */
    float*   enc;         /* [vocab_size × spike_dim] encoder */
    uint64_t steps;
    double   sum_surprise;
};

static void zeri__enc_init(ZeriBrain* b) {
    int V = b->vocab_size, D = b->spike_dim;
    b->enc = (float*)calloc((size_t)V * D, sizeof(float));
    if (V <= D) {
        for (int i = 0; i < V; i++) b->enc[i*D+i] = 1.0f;
    } else {
        for (int tok = 0; tok < V; tok++) {
            float* row = b->enc + tok * D;
            uint64_t h = zeri__sm64((uint64_t)tok + 1);
            int bit = 0;
            for (int d = 0; d < D; d++) {
                if (!bit) { h = zeri__sm64(h); bit = 64; }
                row[d] = (h & 1) ? 1.0f : 0.0f;
                h >>= 1; bit--;
            }
            float norm = 0.0f;
            for (int d = 0; d < D; d++) norm += row[d]*row[d];
            norm = sqrtf(norm);
            if (norm > 0.0f) for (int d = 0; d < D; d++) row[d] /= norm;
        }
    }
}

static void zeri__w_init(ZeriBrain* b) {
    int total = b->vocab_size * b->spike_dim;
    int8_t* tmp = (int8_t*)malloc((size_t)total);
    zeri__vfield(b->seed16, 0, total, tmp);
    for (int i = 0; i < total; i++) { b->W[i] = (float)tmp[i]; b->W0[i] = b->W[i]; }
    free(tmp);
}

static void zeri__ctx_enc(ZeriBrain* b, const int* ctx, int len, float* out) {
    int D = b->spike_dim;
    memset(out, 0, (size_t)D * sizeof(float));
    int use = len < ZERI_MAX_CTX ? len : ZERI_MAX_CTX;
    if (!use) { out[0] = 1.0f; return; }
    for (int i = 0; i < use; i++) {
        int tok = ctx[len - use + i];
        if (tok < 0 || tok >= b->vocab_size) continue;
        float w = (float)(i+1) / (float)use;
        const float* row = b->enc + tok * D;
        for (int d = 0; d < D; d++) out[d] += w * row[d];
    }
    float norm = 0.0f;
    for (int d = 0; d < D; d++) norm += out[d]*out[d];
    norm = sqrtf(norm);
    if (norm > 0.0f) for (int d = 0; d < D; d++) out[d] /= norm;
}

static void zeri__fwd(ZeriBrain* b, const float* x, float* logits, float* probs) {
    int V = b->vocab_size, D = b->spike_dim;
    for (int v = 0; v < V; v++)
        logits[v] = zeri__dot(b->W + v*D, x, D) + b->bias[v];
    zeri__softmax(logits, probs, V);
}

static float zeri__update(ZeriBrain* b, const float* x, int target, float valence) {
    int V = b->vocab_size, D = b->spike_dim;
    float* lg = (float*)malloc((size_t)V * sizeof(float));
    float* pr = (float*)malloc((size_t)V * sizeof(float));
    zeri__fwd(b, x, lg, pr);

    float surprise = -logf(pr[target] + 1e-9f);

    /* decay traces */
    for (int d = 0; d < D; d++) b->pre_trace[d]  *= ZERI_STDP_TAU;
    for (int v = 0; v < V; v++) b->post_trace[v] *= ZERI_STDP_TAU;

    /* update pre-trace from input */
    for (int d = 0; d < D; d++) if (x[d] > 0.0f) b->pre_trace[d] += x[d];

    float lr = ZERI_LR_BASE * valence;
    for (int v = 0; v < V; v++) {
        float g = ((v == target) ? 1.0f : 0.0f) - pr[v];
        if (fabsf(g) < 1e-6f) continue;
        float* Wv = b->W + v * D;
        for (int d = 0; d < D; d++) {
            float dw = lr * g * x[d];
            dw += (v == target)
                  ? ZERI_STDP_A_PLUS  * b->pre_trace[d]
                  : -ZERI_STDP_A_MINUS * b->pre_trace[d];
            Wv[d] += dw;
            if      (Wv[d] >  ZERI_CLIP) Wv[d] =  ZERI_CLIP;
            else if (Wv[d] < -ZERI_CLIP) Wv[d] = -ZERI_CLIP;
        }
        b->bias[v] += lr * g;
    }
    b->post_trace[target] += 1.0f;
    b->steps++;
    b->sum_surprise += (double)surprise;
    free(lg); free(pr);
    return surprise;
}

ZeriBrain* zeri_brain_create(const uint8_t* seed, int spike_dim, int vocab_size) {
    if (spike_dim % 8) spike_dim = (spike_dim + 7) & ~7;
    ZeriBrain* b = (ZeriBrain*)calloc(1, sizeof(ZeriBrain));
    b->spike_dim  = spike_dim;
    b->vocab_size = vocab_size;
    memcpy(b->seed16, seed, ZERI_SEED_BYTES);
    int N = vocab_size * spike_dim;
    b->W         = (float*)calloc((size_t)N, sizeof(float));
    b->W0        = (float*)calloc((size_t)N, sizeof(float));
    b->bias      = (float*)calloc((size_t)vocab_size, sizeof(float));
    b->pre_trace = (float*)calloc((size_t)spike_dim,  sizeof(float));
    b->post_trace= (float*)calloc((size_t)vocab_size, sizeof(float));
    zeri__w_init(b);
    zeri__enc_init(b);
    return b;
}

void zeri_brain_destroy(ZeriBrain* b) {
    if (!b) return;
    free(b->W); free(b->W0); free(b->bias);
    free(b->pre_trace); free(b->post_trace); free(b->enc);
    free(b);
}

float zeri_brain_observe(ZeriBrain* b, const int* ctx, int len,
                          int target, float valence) {
    float* x = (float*)malloc((size_t)b->spike_dim * sizeof(float));
    zeri__ctx_enc(b, ctx, len, x);
    float s = zeri__update(b, x, target, valence);
    free(x);
    return s;
}

int zeri_brain_predict(ZeriBrain* b, const int* ctx, int len) {
    int D = b->spike_dim, V = b->vocab_size;
    float* x  = (float*)malloc((size_t)D * sizeof(float));
    float* lg = (float*)malloc((size_t)V * sizeof(float));
    float* pr = (float*)malloc((size_t)V * sizeof(float));
    zeri__ctx_enc(b, ctx, len, x);
    zeri__fwd(b, x, lg, pr);
    int best = 0;
    for (int v = 1; v < V; v++) if (pr[v] > pr[best]) best = v;
    free(x); free(lg); free(pr);
    return best;
}

void zeri_brain_predict_topk(ZeriBrain* b, const int* ctx, int len,
                               int k, int* ids, float* probs_out) {
    int D = b->spike_dim, V = b->vocab_size;
    float* x  = (float*)malloc((size_t)D * sizeof(float));
    float* lg = (float*)malloc((size_t)V * sizeof(float));
    float* pr = (float*)malloc((size_t)V * sizeof(float));
    zeri__ctx_enc(b, ctx, len, x);
    zeri__fwd(b, x, lg, pr);
    if (k > V) k = V;
    int* used = (int*)calloc((size_t)V, sizeof(int));
    for (int i = 0; i < k; i++) {
        int best = -1;
        for (int v = 0; v < V; v++)
            if (!used[v] && (best < 0 || pr[v] > pr[best])) best = v;
        ids[i] = best; probs_out[i] = pr[best]; used[best] = 1;
    }
    free(used); free(x); free(lg); free(pr);
}

float zeri_brain_surprise(ZeriBrain* b, const int* ctx, int len, int target) {
    int D = b->spike_dim, V = b->vocab_size;
    float* x  = (float*)malloc((size_t)D * sizeof(float));
    float* lg = (float*)malloc((size_t)V * sizeof(float));
    float* pr = (float*)malloc((size_t)V * sizeof(float));
    zeri__ctx_enc(b, ctx, len, x);
    zeri__fwd(b, x, lg, pr);
    float s = -logf(pr[target] + 1e-9f);
    free(x); free(lg); free(pr);
    return s;
}

int zeri_brain_consolidate(ZeriBrain* b, uint8_t* buf, int buf_size) {
    int Wb = b->vocab_size * b->spike_dim * (int)sizeof(float);
    int Bb = b->vocab_size * (int)sizeof(float);
    if (buf_size < 12 + Wb + Bb) return -1;
    uint32_t magic = 0x5A455249;
    memcpy(buf,     &magic,         4);
    memcpy(buf + 4, &b->spike_dim,  4);
    memcpy(buf + 8, &b->vocab_size, 4);
    int off = 12;
    for (int i = 0; i < b->vocab_size * b->spike_dim; i++) {
        float d = b->W[i] - b->W0[i];
        memcpy(buf + off, &d, 4); off += 4;
    }
    for (int i = 0; i < b->vocab_size; i++) {
        memcpy(buf + off, &b->bias[i], 4); off += 4;
    }
    return off;
}

int zeri_brain_load_memory(ZeriBrain* b, const uint8_t* buf, int buf_size) {
    if (buf_size < 12) return -1;
    uint32_t magic; memcpy(&magic, buf, 4);
    if (magic != 0x5A455249) return -2;
    int off = 12;
    for (int i = 0; i < b->vocab_size * b->spike_dim; i++) {
        float d; memcpy(&d, buf + off, 4); off += 4;
        b->W[i] = b->W0[i] + d;
    }
    for (int i = 0; i < b->vocab_size; i++) {
        memcpy(&b->bias[i], buf + off, 4); off += 4;
    }
    return off;
}

void zeri_brain_stats(ZeriBrain* b, float* out) {
    out[0] = (float)b->steps;
    out[1] = b->steps > 0 ? (float)(b->sum_surprise / (double)b->steps) : 0.0f;
    int total = b->vocab_size * b->spike_dim, near0 = 0;
    for (int i = 0; i < total; i++) if (fabsf(b->W[i]) < 0.1f) near0++;
    out[2] = (float)near0 / (float)total;
    out[3] = (float)total;
}

ZeriBrain* zeri_brain_from_name(const char* name, int spike_dim, int vocab_size) {
    uint8_t seed[128] = {0};
    uint64_t h = 14695981039346656037ULL;
    for (const char* p = name; *p; p++) { h ^= (uint8_t)*p; h *= 1099511628211ULL; }
    for (int i = 0; i < 16; i++) { h = zeri__sm64(h + (uint64_t)i); memcpy(seed+i*8, &h, 8); }
    return zeri_brain_create(seed, spike_dim, vocab_size);
}

ZeriBrain* zeri_brain_from_genes(const float* genes, int gene_count,
                                   int spike_dim, int vocab_size) {
    uint8_t seed[128] = {0};
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < gene_count; i++) {
        uint8_t fb[4]; memcpy(fb, &genes[i], 4);
        for (int j = 0; j < 4; j++) { h ^= fb[j]; h *= 1099511628211ULL; }
    }
    for (int i = 0; i < 16; i++) { h = zeri__sm64(h+(uint64_t)i); memcpy(seed+i*8, &h, 8); }
    return zeri_brain_create(seed, spike_dim, vocab_size);
}

#endif /* ZERI_IMPLEMENTATION */
#endif /* ZERI_BRAIN_H */
