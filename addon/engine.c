/* engine.c — native static-site engine for make.fun "Build a static site faster".
 *
 * Written from scratch from the makefun-site README specification and from
 * probing the reference program's behaviour on synthetic inputs (oracle).
 * Single-threaded, arena-allocated, zero-copy where it matters:
 *  - input JSON is parsed schema-first (no generic tree),
 *  - strings without escapes are views into the input buffer,
 *  - markdown/templates render straight into the output buffer,
 *  - answers.json is written with one write() call.
 */
#define _FILE_OFFSET_BITS 64
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#include <immintrin.h>
/* SSE2 mask-draining scan: length of the leading run of bytes that are
 * NOT in the special set {\ ` * [ & < > " '}. Safe to memcpy raw. */
static size_t scan_plain(const char *p, size_t n) {
  __m128i bs = _mm_set1_epi8('\\');
  __m128i bt = _mm_set1_epi8('`');
  __m128i st = _mm_set1_epi8('*');
  __m128i br = _mm_set1_epi8('[');
  __m128i am = _mm_set1_epi8('&');
  __m128i lt = _mm_set1_epi8('<');
  __m128i gt = _mm_set1_epi8('>');
  __m128i dq = _mm_set1_epi8('"');
  __m128i sq = _mm_set1_epi8('\'');
  size_t i = 0;
  while (i + 16 <= n) {
    __m128i w = _mm_loadu_si128((const __m128i *)(p + i));
    int m = _mm_movemask_epi8(_mm_cmpeq_epi8(w, bs))
          | _mm_movemask_epi8(_mm_cmpeq_epi8(w, bt))
          | _mm_movemask_epi8(_mm_cmpeq_epi8(w, st))
          | _mm_movemask_epi8(_mm_cmpeq_epi8(w, br))
          | _mm_movemask_epi8(_mm_cmpeq_epi8(w, am))
          | _mm_movemask_epi8(_mm_cmpeq_epi8(w, lt))
          | _mm_movemask_epi8(_mm_cmpeq_epi8(w, gt))
          | _mm_movemask_epi8(_mm_cmpeq_epi8(w, dq))
          | _mm_movemask_epi8(_mm_cmpeq_epi8(w, sq));
    if (!m) { i += 16; continue; }
    return i + (size_t)__builtin_ctz((unsigned)m);
  }
  while (i < n) {
    char c = p[i];
    if (c=='\\'||c=='`'||c=='*'||c=='['||c=='&'||c=='<'||c=='>'||c=='"'||c=='\'') return i;
    i++;
  }
  return n;
}
/* Length of the leading run with no HTML-special byte (& < > " '). */
static size_t scan_html_plain(const char *p, size_t n) {
  __m128i am = _mm_set1_epi8('&');
  __m128i lt = _mm_set1_epi8('<');
  __m128i gt = _mm_set1_epi8('>');
  __m128i dq = _mm_set1_epi8('"');
  __m128i sq = _mm_set1_epi8('\'');
  size_t i = 0;
  while (i + 16 <= n) {
    __m128i w = _mm_loadu_si128((const __m128i *)(p + i));
    int m = _mm_movemask_epi8(_mm_cmpeq_epi8(w, am))
          | _mm_movemask_epi8(_mm_cmpeq_epi8(w, lt))
          | _mm_movemask_epi8(_mm_cmpeq_epi8(w, gt))
          | _mm_movemask_epi8(_mm_cmpeq_epi8(w, dq))
          | _mm_movemask_epi8(_mm_cmpeq_epi8(w, sq));
    if (!m) { i += 16; continue; }
    return i + (size_t)__builtin_ctz((unsigned)m);
  }
  while (i < n) {
    char c = p[i];
    if (c=='&'||c=='<'||c=='>'||c=='"'||c=='\'') return i;
    i++;
  }
  return n;
}
#include <node_api.h>
#include "slug_tables.h"

/* ---------------- strings ---------------- */
typedef struct { const char *s; size_t n; } Str;
static inline Str str_make(const char *s, size_t n) { Str r; r.s = s; r.n = n; return r; }
static inline int str_eq(Str a, Str b) { return a.n == b.n && (a.n == 0 || memcmp(a.s, b.s, a.n) == 0); }
static inline int str_eq_c(Str a, const char *c) { size_t n = strlen(c); return a.n == n && (n == 0 || memcmp(a.s, c, n) == 0); }
static inline int str_cmp(Str a, Str b) {
  size_t m = a.n < b.n ? a.n : b.n;
  int c = m ? memcmp(a.s, b.s, m) : 0;
  if (c) return c;
  return a.n < b.n ? -1 : a.n > b.n ? 1 : 0;
}
static inline int str_starts(Str a, const char *pre) {
  size_t n = strlen(pre); return a.n >= n && memcmp(a.s, pre, n) == 0;
}
static inline int str_ends(Str a, const char *suf) {
  size_t n = strlen(suf); return a.n >= n && memcmp(a.s + a.n - n, suf, n) == 0;
}

/* ---------------- arena ---------------- */
static char *arena_base, *arena_cur, *arena_end;
static void arena_init(void) {
  if (arena_base) return;
  size_t sz = (size_t)256 << 20;
  arena_base = (char *)malloc(sz);
  if (!arena_base) exit(1);
  arena_cur = arena_base; arena_end = arena_base + sz;
}
static void *arena_alloc(size_t n) {
  n = (n + 15) & ~(size_t)15;
  if (arena_cur + n > arena_end) { fprintf(stderr, "arena overflow\n"); exit(1); }
  void *p = arena_cur; arena_cur += n; return p;
}
static void arena_reset(void) { arena_cur = arena_base; }
/* Second arena: holds the parsed input model (templates/files/decoded
 * strings) so the work arena can be reset per case without wiping it. */
static char *arena2_base, *arena2_cur, *arena2_end;
static void arena2_init(void) {
  if (arena2_base) return;
  size_t sz = (size_t)64 << 20;
  arena2_base = (char *)malloc(sz);
  if (!arena2_base) exit(1);
  arena2_cur = arena2_base; arena2_end = arena2_base + sz;
}
static void *arena2_alloc(size_t n) {
  arena2_init();
  n = (n + 15) & ~(size_t)15;
  if (arena2_cur + n > arena2_end) { fprintf(stderr, "arena2 overflow\n"); exit(1); }
  void *p = arena2_cur; arena2_cur += n; return p;
}
static void arena2_reset(void) { arena2_init(); arena2_cur = arena2_base; }
static Str arena2_str(const char *s, size_t n) {
  char *p = (char *)arena2_alloc(n ? n : 1);
  if (n) memcpy(p, s, n);
  return str_make(p, n);
}
static Str arena_str(const char *s, size_t n) {
  char *p = (char *)arena_alloc(n ? n : 1);
  if (n) memcpy(p, s, n);
  return str_make(p, n);
}
static Str arena_dup(Str a) { return arena_str(a.s, a.n); }
/* sprintf into arena */
static Str arena_fmt(const char *fmt, long a, long b) {
  char tmp[128];
  int n = snprintf(tmp, sizeof tmp, fmt, a, b);
  return arena_str(tmp, (size_t)n);
}

/* ---------------- growable buffer ---------------- */
typedef struct { char *p; size_t n, cap; } Buf;
static void buf_init(Buf *b) { b->cap = 256; b->n = 0; b->p = (char *)malloc(b->cap); if (!b->p) exit(1); }
static void buf_reserve(Buf *b, size_t extra) {
  if (b->n + extra > b->cap) {
    size_t nc = b->cap;
    while (nc < b->n + extra) nc <<= 1;
    b->p = (char *)realloc(b->p, nc); if (!b->p) exit(1);
    b->cap = nc;
  }
}
static inline void buf_add(Buf *b, const void *p, size_t n) { buf_reserve(b, n); memcpy(b->p + b->n, p, n); b->n += n; }
static inline void buf_adds(Buf *b, Str s) { buf_add(b, s.s, s.n); }
static inline void buf_addc(Buf *b, char c) { buf_reserve(b, 1); b->p[b->n++] = c; }
static inline void buf_addlit(Buf *b, const char *s) { buf_add(b, s, strlen(s)); }
static void buf_addll(Buf *b, long long v) {
  char tmp[32]; int n = snprintf(tmp, sizeof tmp, "%lld", v); buf_add(b, tmp, (size_t)n);
}
static void buf_free(Buf *b) { free(b->p); b->p = NULL; b->n = b->cap = 0; }

/* ---------------- UTF-8 ---------------- */
/* Decode one UTF-8 code point (accepts CESU-8 surrogates as 0xD800..0xDFFF).
 * Returns code point, advances *p. Invalid bytes yield the byte value. */
static uint32_t utf8_decode(const char **pp, const char *end) {
  const unsigned char *p = (const unsigned char *)*pp;
  if (p >= (const unsigned char *)end) return 0;
  unsigned c = *p;
  if (c < 0x80) { *pp = (const char *)(p + 1); return c; }
  if (c >= 0xC2 && c < 0xE0 && p + 1 < (const unsigned char *)end && (p[1] & 0xC0) == 0x80) {
    *pp = (const char *)(p + 2); return ((c & 0x1F) << 6) | (p[1] & 0x3F);
  }
  if (c >= 0xE0 && c < 0xF0 && p + 2 < (const unsigned char *)end && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
    *pp = (const char *)(p + 3); return ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
  }
  if (c >= 0xF0 && c < 0xF5 && p + 3 < (const unsigned char *)end && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80 && (p[3] & 0xC0) == 0x80) {
    *pp = (const char *)(p + 4); return ((c & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
  }
  *pp = (const char *)(p + 1); return c;
}
static void utf8_encode(Buf *b, uint32_t cp) {
  if (cp < 0x80) { buf_addc(b, (char)cp); }
  else if (cp < 0x800) { buf_addc(b, (char)(0xC0 | (cp >> 6))); buf_addc(b, (char)(0x80 | (cp & 0x3F))); }
  else if (cp < 0x10000) { buf_addc(b, (char)(0xE0 | (cp >> 12))); buf_addc(b, (char)(0x80 | ((cp >> 6) & 0x3F))); buf_addc(b, (char)(0x80 | (cp & 0x3F))); }
  else { buf_addc(b, (char)(0xF0 | (cp >> 18))); buf_addc(b, (char)(0x80 | ((cp >> 12) & 0x3F))); buf_addc(b, (char)(0x80 | ((cp >> 6) & 0x3F))); buf_addc(b, (char)(0x80 | (cp & 0x3F))); }
}

/* ---------------- slug (NFKD, strip marks, a-z0-9 runs) ---------------- */
static int is_mark(uint32_t cp) {
  int lo = 0, hi = slug_marks_n - 1;
  while (lo <= hi) { int mid = (lo + hi) >> 1;
    if (cp < slug_marks[mid].lo) hi = mid - 1;
    else if (cp > slug_marks[mid].hi) lo = mid + 1;
    else return 1; }
  return 0;
}
static const char *slug_lookup(uint32_t cp) {
  int lo = 0, hi = slug_map_n - 1;
  while (lo <= hi) { int mid = (lo + hi) >> 1;
    if (cp < slug_map[mid].cp) hi = mid - 1;
    else if (cp > slug_map[mid].cp) lo = mid + 1;
    else return slug_map[mid].map; }
  return NULL;
}
/* Append slug of `s` to Buf (already-built slug text, runs collapsed, no
 * leading/trailing '-'). Returns via Buf; caller truncates to 64 later. */
static void slug_append(Buf *b, Str s) {
  const char *p = s.s, *end = s.s + s.n;
  int sep = 1; /* pretend separator before start: drops leading '-' */
  while (p < end) {
    uint32_t cp = utf8_decode(&p, end);
    if (cp < 0x80) {
      if (cp >= 'A' && cp <= 'Z') cp += 32;
      if ((cp >= 'a' && cp <= 'z') || (cp >= '0' && cp <= '9')) { buf_addc(b, (char)cp); sep = 0; }
      else if (!sep) { buf_addc(b, '-'); sep = 1; }
      continue;
    }
    if (is_mark(cp)) continue;
    const char *m = slug_lookup(cp);
    if (!m) { if (!sep) { buf_addc(b, '-'); sep = 1; } continue; }
    for (const char *q = m; *q; q++) {
      if (*q == '-') { if (!sep) { buf_addc(b, '-'); sep = 1; } }
      else { buf_addc(b, *q); sep = 0; }
    }
  }
  if (b->n && b->p[b->n - 1] == '-') b->n--; /* drop trailing '-' */
}
/* slugify into arena Str, cut to 64 chars (trailing '-' at cut dropped),
 * 'untitled' when empty. */
static Str slugify(Str s) {
  Buf b; buf_init(&b);
  slug_append(&b, s);
  if (b.n > 64) { b.n = 64; while (b.n && b.p[b.n - 1] == '-') b.n--; }
  if (b.n == 0) { buf_free(&b); return str_make("untitled", 8); }
  Str r = arena_str(b.p, b.n);
  buf_free(&b);
  return r;
}

/* ---------------- HTML escaping ---------------- */
static const char *html_escapes[256];
static void html_escape_init(void) {
  static int done = 0; if (done) return; done = 1;
  html_escapes['&'] = "&amp;"; html_escapes['<'] = "&lt;"; html_escapes['>'] = "&gt;";
  html_escapes['"'] = "&quot;"; html_escapes['\''] = "&#39;";
}
/* Append Str with HTML escaping. Fast path: SSE2 scan for the 5 specials. */
static void buf_add_escaped(Buf *b, Str s) {
  html_escape_init();
  const char *p = s.s, *end = s.s + s.n;
  while (p < end) {
    size_t run = scan_html_plain(p, (size_t)(end - p));
    if (run) buf_add(b, p, run);
    p += run;
    if (p >= end) break;
    buf_addlit(b, html_escapes[(unsigned char)*p]);
    p++;
  }
}
static inline void buf_add_escaped_char(Buf *b, char c) {
  html_escape_init();
  const char *e = html_escapes[(unsigned char)c];
  if (e) buf_addlit(b, e); else buf_addc(b, c);
}

/* ---------------- JSON string escaping (for answers.json) ---------------- */
/* SWAR helpers: detect stop bytes (", \, <0x20, 0xED) 8 bytes at a time. */
static inline uint64_t _swar_haszero(uint64_t x) { return (x - 0x0101010101010101ULL) & ~x & 0x8080808080808080ULL; }
static inline uint64_t _swar_hasbyte(uint64_t w, unsigned c) { return _swar_haszero(w ^ (0x0101010101010101ULL * c)); }
static inline uint64_t _swar_hasless(uint64_t w, unsigned n) { return (w - 0x0101010101010101ULL * n) & ~w & 0x8080808080808080ULL; }
/* AVX2 JSON escaper: 32-byte chunks. Compiled with target("avx2"); dispatched at runtime. */
__attribute__((target("avx2")))
static void buf_add_json_escaped_avx2(Buf *b, Str s) {
  static const char hexd[] = "0123456789abcdef";
  const unsigned char *p = (const unsigned char *)s.s, *end = p + s.n;
  buf_reserve(b, s.n * 6 + 2);
  char *dst = b->p + b->n;
  *dst++ = '"';
  __m256i qm = _mm256_set1_epi8('"'), bm = _mm256_set1_epi8('\\'), em = _mm256_set1_epi8((char)0xED), lo = _mm256_set1_epi8(0x1F);
  while (end - p >= 32) {
    __m256i v = _mm256_loadu_si256((const __m256i *)p);
    __m256i m = _mm256_or_si256(_mm256_cmpeq_epi8(v, qm), _mm256_cmpeq_epi8(v, bm));
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, em));
    m = _mm256_or_si256(m, _mm256_cmpeq_epi8(_mm256_min_epu8(v, lo), v));
    unsigned mask = (unsigned)_mm256_movemask_epi8(m);
    if (!mask) { memcpy(dst, p, 32); dst += 32; p += 32; continue; }
    unsigned idx = (unsigned)__builtin_ctz(mask);
    if (idx) { memcpy(dst, p, idx); dst += idx; p += idx; }
    unsigned c = *p;
    if (c == '"') { memcpy(dst, "\\\"", 2); dst += 2; p++; }
    else if (c == '\\') { memcpy(dst, "\\\\", 2); dst += 2; p++; }
    else if (c == '\n') { memcpy(dst, "\\n", 2); dst += 2; p++; }
    else if (c == '\r') { memcpy(dst, "\\r", 2); dst += 2; p++; }
    else if (c == '\t') { memcpy(dst, "\\t", 2); dst += 2; p++; }
    else if (c == 0x08) { memcpy(dst, "\\b", 2); dst += 2; p++; }
    else if (c == 0x0C) { memcpy(dst, "\\f", 2); dst += 2; p++; }
    else if (c < 0x20) {
      dst[0] = '\\'; dst[1] = 'u'; dst[2] = '0'; dst[3] = '0';
      dst[4] = hexd[(c >> 4) & 0xF]; dst[5] = hexd[c & 0xF]; dst += 6; p++;
    } else if (c == 0xED && p + 2 < end && (p[1] & 0xE0) == 0xA0 && (p[2] & 0xC0) == 0x80) {
      unsigned cp = ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
      dst[0] = '\\'; dst[1] = 'u'; dst[2] = hexd[(cp >> 12) & 0xF];
      dst[3] = hexd[(cp >> 8) & 0xF]; dst[4] = hexd[(cp >> 4) & 0xF]; dst[5] = hexd[cp & 0xF];
      dst += 6; p += 3;
    } else { *dst++ = (char)c; p++; }
  }
  while (p < end) {
    unsigned c = *p;
    if (c == '"') { memcpy(dst, "\\\"", 2); dst += 2; p++; }
    else if (c == '\\') { memcpy(dst, "\\\\", 2); dst += 2; p++; }
    else if (c == '\n') { memcpy(dst, "\\n", 2); dst += 2; p++; }
    else if (c == '\r') { memcpy(dst, "\\r", 2); dst += 2; p++; }
    else if (c == '\t') { memcpy(dst, "\\t", 2); dst += 2; p++; }
    else if (c == 0x08) { memcpy(dst, "\\b", 2); dst += 2; p++; }
    else if (c == 0x0C) { memcpy(dst, "\\f", 2); dst += 2; p++; }
    else if (c < 0x20) {
      dst[0] = '\\'; dst[1] = 'u'; dst[2] = '0'; dst[3] = '0';
      dst[4] = hexd[(c >> 4) & 0xF]; dst[5] = hexd[c & 0xF]; dst += 6; p++;
    } else if (c == 0xED && p + 2 < end && (p[1] & 0xE0) == 0xA0 && (p[2] & 0xC0) == 0x80) {
      unsigned cp = ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
      dst[0] = '\\'; dst[1] = 'u'; dst[2] = hexd[(cp >> 12) & 0xF];
      dst[3] = hexd[(cp >> 8) & 0xF]; dst[4] = hexd[(cp >> 4) & 0xF]; dst[5] = hexd[cp & 0xF];
      dst += 6; p += 3;
    } else { *dst++ = (char)c; p++; }
  }
  *dst++ = '"';
  b->n = (size_t)(dst - b->p);
}

static int _have_avx2 = -1;
static void buf_add_json_escaped(Buf *b, Str s) {
  if (_have_avx2 < 0) { __builtin_cpu_init(); _have_avx2 = __builtin_cpu_supports("avx2") ? 1 : 0; }
  if (_have_avx2) { buf_add_json_escaped_avx2(b, s); return; }
  static const char hexd[] = "0123456789abcdef";
  const unsigned char *p = (const unsigned char *)s.s, *end = p + s.n;
  /* reserve worst-case space once (6x: every byte could become \u00xx),
   * then write via a raw cursor */
  buf_reserve(b, s.n * 6 + 2);
  char *dst = b->p + b->n;
  *dst++ = '"';
  /* SSE2: stop bytes are ", \, <0x20, 0xED (CESU-8 surrogate lead).
   * Scan 16 bytes at a time; copy the clean run up to the first stop. */
  __m128i qm = _mm_set1_epi8('"'), bm = _mm_set1_epi8('\\'), em = _mm_set1_epi8((char)0xED), lo = _mm_set1_epi8(0x1F);
  while (end - p >= 16) {
    __m128i v = _mm_loadu_si128((const __m128i *)p);
    __m128i m = _mm_or_si128(_mm_cmpeq_epi8(v, qm), _mm_cmpeq_epi8(v, bm));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(v, em));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(_mm_min_epu8(v, lo), v));
    unsigned mask = (unsigned)_mm_movemask_epi8(m);
    if (!mask) { memcpy(dst, p, 16); dst += 16; p += 16; continue; }
    unsigned idx = (unsigned)__builtin_ctz(mask);
    if (idx) { memcpy(dst, p, idx); dst += idx; p += idx; }
    unsigned c = *p; /* stop byte: handle one byte */
    if (c == '"') { memcpy(dst, "\\\"", 2); dst += 2; p++; }
    else if (c == '\\') { memcpy(dst, "\\\\", 2); dst += 2; p++; }
    else if (c == '\n') { memcpy(dst, "\\n", 2); dst += 2; p++; }
    else if (c == '\r') { memcpy(dst, "\\r", 2); dst += 2; p++; }
    else if (c == '\t') { memcpy(dst, "\\t", 2); dst += 2; p++; }
    else if (c == 0x08) { memcpy(dst, "\\b", 2); dst += 2; p++; }
    else if (c == 0x0C) { memcpy(dst, "\\f", 2); dst += 2; p++; }
    else if (c < 0x20) {
      dst[0] = '\\'; dst[1] = 'u'; dst[2] = '0'; dst[3] = '0';
      dst[4] = hexd[(c >> 4) & 0xF]; dst[5] = hexd[c & 0xF]; dst += 6; p++;
    } else if (c == 0xED && p + 2 < end && (p[1] & 0xE0) == 0xA0 && (p[2] & 0xC0) == 0x80) {
      unsigned cp = ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
      dst[0] = '\\'; dst[1] = 'u'; dst[2] = hexd[(cp >> 12) & 0xF];
      dst[3] = hexd[(cp >> 8) & 0xF]; dst[4] = hexd[(cp >> 4) & 0xF]; dst[5] = hexd[cp & 0xF];
      dst += 6; p += 3;
    } else { *dst++ = (char)c; p++; }
  }
  /* tail (<16 bytes): scalar */
  while (p < end) {
    unsigned c = *p;
    if (c == '"') { memcpy(dst, "\\\"", 2); dst += 2; p++; }
    else if (c == '\\') { memcpy(dst, "\\\\", 2); dst += 2; p++; }
    else if (c == '\n') { memcpy(dst, "\\n", 2); dst += 2; p++; }
    else if (c == '\r') { memcpy(dst, "\\r", 2); dst += 2; p++; }
    else if (c == '\t') { memcpy(dst, "\\t", 2); dst += 2; p++; }
    else if (c == 0x08) { memcpy(dst, "\\b", 2); dst += 2; p++; }
    else if (c == 0x0C) { memcpy(dst, "\\f", 2); dst += 2; p++; }
    else if (c < 0x20) {
      dst[0] = '\\'; dst[1] = 'u'; dst[2] = '0'; dst[3] = '0';
      dst[4] = hexd[(c >> 4) & 0xF]; dst[5] = hexd[c & 0xF]; dst += 6; p++;
    } else if (c == 0xED && p + 2 < end && (p[1] & 0xE0) == 0xA0 && (p[2] & 0xC0) == 0x80) {
      unsigned cp = ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
      dst[0] = '\\'; dst[1] = 'u'; dst[2] = hexd[(cp >> 12) & 0xF];
      dst[3] = hexd[(cp >> 8) & 0xF]; dst[4] = hexd[(cp >> 4) & 0xF]; dst[5] = hexd[cp & 0xF];
      dst += 6; p += 3;
    } else { *dst++ = (char)c; p++; }
  }
  *dst++ = '"';
  b->n = (size_t)(dst - b->p);
}

/* ---------------- dates ---------------- */
static const char *month_names[12] = {"January","February","March","April","May","June",
  "July","August","September","October","November","December"};
static int days_in_month(int y, int m) {
  static const int dm[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
  if (m == 2) { int leap = (y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)); return dm[m-1] + leap; }
  return dm[m-1];
}
static int valid_date_str(Str s) {
  if (s.n != 10 || s.s[4] != '-' || s.s[7] != '-') return 0;
  for (int i = 0; i < 10; i++) { if (i == 4 || i == 7) continue; if (s.s[i] < '0' || s.s[i] > '9') return 0; }
  int y = (s.s[0]-'0')*1000 + (s.s[1]-'0')*100 + (s.s[2]-'0')*10 + (s.s[3]-'0');
  int m = (s.s[5]-'0')*10 + (s.s[6]-'0');
  int d = (s.s[8]-'0')*10 + (s.s[9]-'0');
  if (y < 1970 || y > 2099 || m < 1 || m > 12 || d < 1 || d > days_in_month(y, m)) return 0;
  return 1;
}
/* "2026-11-09" -> "9 November 2026" into arena */
static Str date_long(Str s) {
  int y = (s.s[0]-'0')*1000 + (s.s[1]-'0')*100 + (s.s[2]-'0')*10 + (s.s[3]-'0');
  int m = (s.s[5]-'0')*10 + (s.s[6]-'0');
  int d = (s.s[8]-'0')*10 + (s.s[9]-'0');
  char tmp[64];
  int n = snprintf(tmp, sizeof tmp, "%d %s %d", d, month_names[m-1], y);
  return arena_str(tmp, (size_t)n);
}

/* ---------------- JSON parsing (schema-first, zero-copy strings) ---------------- */
typedef struct { const char *p, *end; } JCur;
static inline void j_ws(JCur *j) {
  while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) j->p++;
}
static int j_expect(JCur *j, char c) { j_ws(j); if (j->p < j->end && *j->p == c) { j->p++; return 1; } return 0; }

/* Decode a JSON string starting at the opening quote. Returns Str:
 * view into input when no escapes, arena copy when decoded. */
static Str j_string(JCur *j) {
  /* j->p points at '"' */
  const char *start = j->p + 1;
  const char *p = start;
  const char *bs = NULL, *quote = NULL;
  /* fast scan: find closing quote and first backslash with memchr.
   * The backslash search is bounded by the quote position (never scans
   * past the end of this string). */
  while (p < j->end) {
    const char *q = (const char *)memchr(p, '"', (size_t)(j->end - p));
    if (!q) break;
    const char *b = (const char *)memchr(p, '\\', (size_t)(q - p));
    if (b) { bs = b; p = b + 2; continue; } /* skip the escaped char too */
    quote = q; break;
  }
  if (!quote) { j->p = j->end; return str_make("", 0); }
  j->p = quote + 1;
  if (!bs) return str_make(start, (size_t)(quote - start));
  /* decode with escapes into arena */
  Buf b; buf_init(&b);
  p = start;
  while (p < quote) {
    char c = *p;
    if (c != '\\') { buf_addc(&b, c); p++; continue; }
    p++;
    if (p >= quote) break;
    char e = *p++;
    switch (e) {
      case '"': buf_addc(&b, '"'); break;
      case '\\': buf_addc(&b, '\\'); break;
      case '/': buf_addc(&b, '/'); break;
      case 'b': buf_addc(&b, '\b'); break;
      case 'f': buf_addc(&b, '\f'); break;
      case 'n': buf_addc(&b, '\n'); break;
      case 'r': buf_addc(&b, '\r'); break;
      case 't': buf_addc(&b, '\t'); break;
      case 'u': {
        unsigned cp = 0;
        for (int i = 0; i < 4 && p < quote; i++) {
          char h = *p++; cp <<= 4;
          if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
          else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
          else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
        }
        if (cp >= 0xD800 && cp <= 0xDBFF && p + 6 <= quote && p[0] == '\\' && p[1] == 'u') {
          unsigned lo = 0;
          const char *q2 = p + 2;
          for (int i = 0; i < 4; i++) {
            char h = *q2++; lo <<= 4;
            if (h >= '0' && h <= '9') lo |= (unsigned)(h - '0');
            else if (h >= 'a' && h <= 'f') lo |= (unsigned)(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') lo |= (unsigned)(h - 'A' + 10);
          }
          if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); p += 6; }
        }
        if (cp >= 0xD800 && cp <= 0xDFFF) {
          /* lone surrogate: store CESU-8 so it round-trips as \udXXX */
          buf_addc(&b, (char)(0xE0 | (cp >> 12)));
          buf_addc(&b, (char)(0x80 | ((cp >> 6) & 0x3F)));
          buf_addc(&b, (char)(0x80 | (cp & 0x3F)));
        } else utf8_encode(&b, cp);
        break;
      }
      default: buf_addc(&b, e); break;
    }
  }
  Str r = arena2_str(b.p, b.n);
  buf_free(&b);
  return r;
}
typedef struct { int is_double; long long i; double d; } JNum;
static JNum j_number(JCur *j) {
  JNum r; r.is_double = 0; r.i = 0; r.d = 0;
  const char *s = j->p;
  int neg = 0;
  if (j->p < j->end && *j->p == '-') { neg = 1; j->p++; }
  long long v = 0; int isdbl = 0;
  while (j->p < j->end && *j->p >= '0' && *j->p <= '9') { v = v * 10 + (*j->p - '0'); j->p++; }
  if (j->p < j->end && (*j->p == '.' || *j->p == 'e' || *j->p == 'E')) isdbl = 1;
  if (isdbl) {
    char tmp[64]; size_t n = (size_t)(j->p - s);
    /* consume rest of number */
    while (j->p < j->end && (isdigit((unsigned char)*j->p) || *j->p == '.' || *j->p == 'e' || *j->p == 'E' || *j->p == '+' || *j->p == '-')) j->p++;
    n = (size_t)(j->p - s); if (n >= sizeof tmp) n = sizeof tmp - 1;
    memcpy(tmp, s, n); tmp[n] = 0;
    r.is_double = 1; r.d = strtod(tmp, NULL);
  } else {
    r.i = neg ? -v : v;
  }
  return r;
}
static void j_skip(JCur *j); /* fwd */
static void j_skip_value(JCur *j) { j_skip(j); }
static void j_skip(JCur *j) {
  j_ws(j);
  if (j->p >= j->end) return;
  char c = *j->p;
  if (c == '"') { (void)j_string(j); return; }
  if (c == '{') {
    j->p++;
    if (j_expect(j, '}')) return;
    for (;;) { j_ws(j); (void)j_string(j); j_expect(j, ':'); j_skip(j); if (!j_expect(j, ',')) break; }
    j_expect(j, '}'); return;
  }
  if (c == '[') {
    j->p++;
    if (j_expect(j, ']')) return;
    for (;;) { j_skip(j); if (!j_expect(j, ',')) break; }
    j_expect(j, ']'); return;
  }
  if (c == 't') { j->p += 4; return; }
  if (c == 'f') { j->p += 5; return; }
  if (c == 'n') { j->p += 4; return; }
  (void)j_number(j);
}

/* ---------------- case input model ---------------- */
typedef struct { Str name, src; } Template;
typedef struct { Str path, content; } InFile;
typedef struct {
  Str title, base_url; long long per_page, recent;
  Template *templates; int ntemplates;
  InFile *files; int nfiles;
} CaseInput;

static void parse_site(JCur *j, CaseInput *ci) {
  j_expect(j, '{');
  if (j_expect(j, '}')) return;
  for (;;) {
    j_ws(j); Str k = j_string(j); j_expect(j, ':'); j_ws(j);
    if (str_eq_c(k, "title")) ci->title = j_string(j);
    else if (str_eq_c(k, "base_url")) ci->base_url = j_string(j);
    else if (str_eq_c(k, "per_page")) ci->per_page = j_number(j).i;
    else if (str_eq_c(k, "recent")) ci->recent = j_number(j).i;
    else j_skip_value(j);
    if (!j_expect(j, ',')) break;
  }
  j_expect(j, '}');
}
static int cmp_template(const void *a, const void *b) {
  return str_cmp(((const Template *)a)->name, ((const Template *)b)->name);
}
static void parse_templates(JCur *j, CaseInput *ci) {
  int cap = 64, n = 0;
  Template *arr = (Template *)arena2_alloc(sizeof(Template) * cap);
  j_expect(j, '{');
  if (j_expect(j, '}')) { ci->templates = arr; ci->ntemplates = 0; return; }
  for (;;) {
    j_ws(j); Str k = j_string(j); j_expect(j, ':'); j_ws(j);
    Str v = j_string(j);
    if (n == cap) { /* templates are few; arena is big, just use a bigger static bump */ }
    arr[n].name = k; arr[n].src = v; n++;
    if (!j_expect(j, ',')) break;
  }
  j_expect(j, '}');
  qsort(arr, (size_t)n, sizeof(Template), cmp_template);
  ci->templates = arr; ci->ntemplates = n;
}
static void parse_files(JCur *j, CaseInput *ci) {
  int cap = 1024, n = 0;
  InFile *arr = (InFile *)arena2_alloc(sizeof(InFile) * cap);
  j_expect(j, '[');
  if (j_expect(j, ']')) { ci->files = arr; ci->nfiles = 0; return; }
  for (;;) {
    j_ws(j);
    InFile f; f.path = str_make("", 0); f.content = str_make("", 0);
    j_expect(j, '{');
    if (!j_expect(j, '}')) {
      for (;;) {
        j_ws(j); Str k = j_string(j); j_expect(j, ':'); j_ws(j);
        if (str_eq_c(k, "path")) f.path = j_string(j);
        else if (str_eq_c(k, "content")) f.content = j_string(j);
        else j_skip_value(j);
        if (!j_expect(j, ',')) break;
      }
      j_expect(j, '}');
    }
    arr[n++] = f;
    if (!j_expect(j, ',')) break;
  }
  j_expect(j, ']');
  ci->files = arr; ci->nfiles = n;
}
static void parse_case_input(JCur *j, CaseInput *ci) {
  memset(ci, 0, sizeof *ci);
  ci->per_page = 0; ci->recent = 0;
  j_expect(j, '{');
  if (j_expect(j, '}')) return;
  for (;;) {
    j_ws(j); Str k = j_string(j); j_expect(j, ':'); j_ws(j);
    if (str_eq_c(k, "site")) parse_site(j, ci);
    else if (str_eq_c(k, "templates")) parse_templates(j, ci);
    else if (str_eq_c(k, "files")) parse_files(j, ci);
    else j_skip_value(j);
    if (!j_expect(j, ',')) break;
  }
  j_expect(j, '}');
}

/* ---------------- template values ---------------- */
typedef enum { V_NIL, V_STR, V_NUM, V_BOOL, V_ARR, V_OBJ } VType;
typedef struct Value Value;
typedef struct { Str name; Value *val; } Field;
struct Value {
  VType t;
  Str s;            /* V_STR */
  long long i; double d; int is_double; /* V_NUM */
  int b;            /* V_BOOL */
  Value **items; int n; /* V_ARR */
  Field *fields; int nf; /* V_OBJ */
};
static Value *v_new(VType t) { Value *v = (Value *)arena_alloc(sizeof(Value)); memset(v, 0, sizeof *v); v->t = t; return v; }
static Value *v_str(Str s) { Value *v = v_new(V_STR); v->s = s; return v; }
static Value *v_num(long long i) { Value *v = v_new(V_NUM); v->i = i; v->is_double = 0; return v; }
static Value *v_bool(int b) { Value *v = v_new(V_BOOL); v->b = !!b; return v; }
static Value *v_nil(void) { static Value z; z.t = V_NIL; return &z; }
static Value *v_arr(Value **items, int n) { Value *v = v_new(V_ARR); v->items = items; v->n = n; return v; }
static Value *v_obj(Field *f, int n) { Value *v = v_new(V_OBJ); v->fields = f; v->nf = n; return v; }
static Value *obj_field(Value *o, Str name) {
  if (!o || o->t != V_OBJ) return NULL;
  for (int i = 0; i < o->nf; i++) if (str_eq(o->fields[i].name, name)) return o->fields[i].val;
  return NULL;
}
static int v_truthy(Value *v) {
  if (!v) return 0;
  switch (v->t) {
    case V_NIL: return 0;
    case V_BOOL: return v->b;
    case V_NUM: return v->is_double ? v->d != 0 : v->i != 0;
    case V_STR: return v->s.n != 0;
    case V_ARR: return v->n != 0;
    default: return 1;
  }
}
static void format_num(Buf *b, Value *v) {
  if (!v->is_double) { buf_addll(b, v->i); return; }
  /* shortest round-trip among precisions 15..17 */
  char tmp[64];
  for (int prec = 15; prec <= 17; prec++) {
    snprintf(tmp, sizeof tmp, "%.*g", prec, v->d);
    if (strtod(tmp, NULL) == v->d) { buf_addlit(b, tmp); return; }
  }
  snprintf(tmp, sizeof tmp, "%.17g", v->d); buf_addlit(b, tmp);
}
static void v_write(Value *v, Buf *b, int escape) {
  if (!v) return;
  switch (v->t) {
    case V_NIL: break;
    case V_STR: if (escape) buf_add_escaped(b, v->s); else buf_adds(b, v->s); break;
    case V_NUM: format_num(b, v); break;
    case V_BOOL: buf_addlit(b, v->b ? "true" : "false"); break;
    case V_ARR:
      for (int i = 0; i < v->n; i++) { if (i) buf_addlit(b, ", "); v_write(v->items[i], b, escape); }
      break;
    default: break;
  }
}

/* ---------------- template parsing ---------------- */
typedef enum { N_TEXT, N_VAR, N_IF, N_FOR, N_INCLUDE } NKind;
typedef struct Node Node;
struct Node {
  NKind kind;
  Str text;          /* N_TEXT */
  Str *segs; int nsegs; /* N_VAR / N_IF / N_FOR path */
  int raw;           /* N_VAR */
  int neg;           /* N_IF */
  Str varname;       /* N_FOR */
  Str inc_name;      /* N_INCLUDE */
  Node *then_branch, *else_branch; /* N_IF */
  Node *body;        /* N_FOR */
  Node *next;
};
typedef struct {
  const char *s; size_t len, pos;
  char err[256];
} TParse;
/* Validate a path string (already trimmed). Returns segment count or -1. */
static int path_segs(Str p, Str **out) {
  if (p.n == 0) return -1;
  int n = 1;
  for (size_t i = 0; i < p.n; i++) if (p.s[i] == '.') n++;
  Str *segs = (Str *)arena_alloc(sizeof(Str) * (size_t)n);
  int k = 0; size_t start = 0;
  for (size_t i = 0; i <= p.n; i++) {
    if (i == p.n || p.s[i] == '.') {
      size_t ln = i - start;
      if (ln == 0) return -1;
      char c0 = p.s[start];
      if (!((c0 >= 'a' && c0 <= 'z') || (c0 >= 'A' && c0 <= 'Z') || c0 == '_')) return -1;
      for (size_t j = start + 1; j < i; j++) {
        char c = p.s[j];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) return -1;
      }
      segs[k++] = str_make(p.s + start, ln);
      start = i + 1;
    }
  }
  *out = segs;
  return k;
}
static Str trim_str(Str s) {
  while (s.n && (s.s[0] == ' ' || s.s[0] == '\t')) { s.s++; s.n--; }
  while (s.n && (s.s[s.n-1] == ' ' || s.s[s.n-1] == '\t')) s.n--;
  return s;
}
static Node *node_new(NKind k) { Node *nd = (Node *)arena_alloc(sizeof(Node)); memset(nd, 0, sizeof *nd); nd->kind = k; return nd; }

enum { PS_NONE, PS_IF, PS_IFELSE, PS_FOR };
enum { PE_OK, PE_ELSE, PE_ENDIF, PE_ENDFOR };
/* Parse a node sequence until EOF or a closing tag for `stop`.
 * On return *endkind is PE_OK (EOF) or the closing kind. */
static Node *parse_seq(TParse *tp, int stop, int *endkind) {
  Node *head = NULL, **tail = &head;
  *endkind = PE_OK;
  while (tp->pos < tp->len) {
    const char *s = tp->s; size_t n = tp->len, pos = tp->pos;
    /* find next {{ or {% */
    size_t nxt = n; int is_tag = 0, is_raw = 0;
    for (size_t i = pos; i + 1 < n; i++) {
      if (s[i] == '{' && (s[i+1] == '{' || s[i+1] == '%')) { nxt = i; is_tag = (s[i+1] == '%'); is_raw = (s[i+1] == '{' && i + 2 < n && s[i+2] == '{'); break; }
    }
    if (nxt > pos) {
      Node *nd = node_new(N_TEXT);
      nd->text = str_make(s + pos, nxt - pos);
      *tail = nd; tail = &nd->next;
      tp->pos = nxt;
      if (nxt >= n) break;
    }
    if (tp->pos >= n) break;
    if (is_tag) {
      const char *close = NULL;
      for (size_t i = tp->pos + 2; i + 1 < n; i++) if (s[i] == '%' && s[i+1] == '}') { close = s + i; break; }
      if (!close) { snprintf(tp->err, sizeof tp->err, "a {%% is not closed"); return head; }
      Str inner = trim_str(str_make(s + tp->pos + 2, (size_t)(close - (s + tp->pos + 2))));
      tp->pos = (size_t)(close - s) + 2;
      /* classify tag */
      if (str_eq_c(inner, "endif")) {
        if (stop == PS_IF || stop == PS_IFELSE) { *endkind = PE_ENDIF; return head; }
        snprintf(tp->err, sizeof tp->err, "an endif with no open if"); return head;
      }
      if (str_eq_c(inner, "endfor")) {
        if (stop == PS_FOR) { *endkind = PE_ENDFOR; return head; }
        snprintf(tp->err, sizeof tp->err, "an endfor with no open for"); return head;
      }
      if (str_eq_c(inner, "else")) {
        if (stop == PS_IF) { *endkind = PE_ELSE; return head; }
        snprintf(tp->err, sizeof tp->err, "an else outside an if"); return head;
      }
      /* tokenize inner by whitespace */
      /* if / for / include */
      if (str_starts(inner, "if ") || str_eq_c(inner, "if")) {
        if (str_eq_c(inner, "if")) goto bad_tag;
        Str rest = str_make(inner.s + 3, inner.n - 3);
        int neg = 0;
        if (str_starts(rest, "not ")) { neg = 1; rest.s += 4; rest.n -= 4; }
        rest = trim_str(rest);
        /* single-token path: a bad path is reported as not-a-path;
         * empty or multi-token rest is not a tag */
        {
          int hasws = 0;
          for (size_t k2 = 0; k2 < rest.n; k2++) if (rest.s[k2] == ' ' || rest.s[k2] == '\t') { hasws = 1; break; }
          if (rest.n == 0 || hasws) goto bad_tag;
        }
        Str *segs; int ns = path_segs(rest, &segs);
        if (ns < 0) {
          char tmp[200]; size_t cn = rest.n < 180 ? rest.n : 180;
          memcpy(tmp, rest.s, cn); tmp[cn] = 0;
          snprintf(tp->err, sizeof tp->err, "\"%s\" is not a path", tmp);
          return head;
        }
        Node *nd = node_new(N_IF); nd->segs = segs; nd->nsegs = ns; nd->neg = neg;
        int ek1; nd->then_branch = parse_seq(tp, PS_IF, &ek1);
        if (tp->err[0]) return head;
        if (ek1 == PE_OK) { snprintf(tp->err, sizeof tp->err, "a if is not closed"); return head; }
        if (ek1 == PE_ELSE) {
          int ek2; nd->else_branch = parse_seq(tp, PS_IFELSE, &ek2);
          if (tp->err[0]) return head;
          if (ek2 != PE_ENDIF) { snprintf(tp->err, sizeof tp->err, "a if is not closed"); return head; }
        }
        *tail = nd; tail = &nd->next;
        continue;
      }
      if (str_starts(inner, "for ") || str_eq_c(inner, "for")) {
        if (str_eq_c(inner, "for")) goto bad_tag;
        /* tokens: var in path */
        Str rest = trim_str(str_make(inner.s + 4, inner.n - 4));
        /* split into 3 whitespace-separated tokens */
        Str tok[3]; int nt = 0;
        size_t i = 0;
        while (i < rest.n && nt < 3) {
          while (i < rest.n && (rest.s[i] == ' ' || rest.s[i] == '\t')) i++;
          if (i >= rest.n) break;
          size_t st = i;
          while (i < rest.n && rest.s[i] != ' ' && rest.s[i] != '\t') i++;
          tok[nt++] = str_make(rest.s + st, i - st);
        }
        while (i < rest.n && (rest.s[i] == ' ' || rest.s[i] == '\t')) i++;
        if (nt != 3 || i != rest.n || !str_eq_c(tok[1], "in")) goto bad_tag;
        /* var name valid? */
        {
          Str vn = tok[0]; int ok = vn.n > 0 && ((vn.s[0] >= 'a' && vn.s[0] <= 'z') || (vn.s[0] >= 'A' && vn.s[0] <= 'Z') || vn.s[0] == '_');
          for (size_t k2 = 1; ok && k2 < vn.n; k2++) { char c = vn.s[k2]; ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }
          if (!ok) goto bad_tag;
        }
        Str *segs; int ns = path_segs(tok[2], &segs);
        if (ns < 0) {
          char tmp[200]; size_t cn = tok[2].n < 180 ? tok[2].n : 180;
          memcpy(tmp, tok[2].s, cn); tmp[cn] = 0;
          snprintf(tp->err, sizeof tp->err, "\"%s\" is not a path", tmp);
          return head;
        }
        Node *nd = node_new(N_FOR); nd->varname = tok[0]; nd->segs = segs; nd->nsegs = ns;
        int ek; nd->body = parse_seq(tp, PS_FOR, &ek);
        if (tp->err[0]) return head;
        if (ek != PE_ENDFOR) { snprintf(tp->err, sizeof tp->err, "a for is not closed"); return head; }
        *tail = nd; tail = &nd->next;
        continue;
      }
      if (str_starts(inner, "include ") || str_eq_c(inner, "include")) {
        if (str_eq_c(inner, "include")) goto bad_tag;
        Str rest = trim_str(str_make(inner.s + 8, inner.n - 8));
        if (rest.n < 2 || rest.s[0] != '"' || rest.s[rest.n-1] != '"') goto bad_tag;
        for (size_t k2 = 1; k2 + 1 < rest.n; k2++) if (rest.s[k2] == '"') goto bad_tag;
        Node *nd = node_new(N_INCLUDE);
        nd->inc_name = str_make(rest.s + 1, rest.n - 2);
        *tail = nd; tail = &nd->next;
        continue;
      }
bad_tag:;
      {
        /* {% <inner> %} is not a tag — inner as trimmed */
        char tmp[200];
        size_t cn = inner.n < 180 ? inner.n : 180;
        memcpy(tmp, inner.s, cn); tmp[cn] = 0;
        snprintf(tp->err, sizeof tp->err, "{%% %s %%} is not a tag", tmp);
        return head;
      }
    } else {
      /* {{ or {{{ */
      int raw = is_raw;
      const char *pat = raw ? "}}}" : "}}";
      size_t plen = raw ? 3 : 2;
      const char *close = NULL;
      for (size_t i = tp->pos + plen; i + plen <= n; i++) {
        if (memcmp(s + i, pat, plen) == 0) { close = s + i; break; }
      }
      if (!close) { snprintf(tp->err, sizeof tp->err, "a %s is not closed", raw ? "{{{" : "{{"); return head; }
      Str inner = trim_str(str_make(s + tp->pos + plen, (size_t)(close - (s + tp->pos + plen))));
      tp->pos = (size_t)(close - s) + plen;
      Str *segs; int ns = path_segs(inner, &segs);
      if (ns < 0) {
        char tmp[200]; size_t cn = inner.n < 180 ? inner.n : 180;
        memcpy(tmp, inner.s, cn); tmp[cn] = 0;
        snprintf(tp->err, sizeof tp->err, "\"%s\" is not a path", tmp);
        return head;
      }
      Node *nd = node_new(N_VAR); nd->segs = segs; nd->nsegs = ns; nd->raw = raw;
      *tail = nd; tail = &nd->next;
    }
  }
  if (stop != PS_NONE) {
    snprintf(tp->err, sizeof tp->err, "a %s is not closed", stop == PS_FOR ? "for" : "if");
  }
  return head;
}

/* Remove standalone {% %} lines (only spaces/tabs beside the tag on its line).
 * The tag itself is kept; the line's surrounding spaces and its \n are not.
 * Decided line by line on the original source. */
static Str tpl_clean(Str src) {
  Buf b; buf_init(&b);
  const char *s = src.s; size_t n = src.n, i = 0;
  while (i < n) {
    size_t ls = i, le = i;
    while (le < n && s[le] != '\n') le++;
    /* line is s[ls..le), newline at le if le<n */
    size_t tstart = ls, tend = le;
    while (tstart < tend && (s[tstart] == ' ' || s[tstart] == '\t')) tstart++;
    while (tend > tstart && (s[tend-1] == ' ' || s[tend-1] == '\t')) tend--;
    int standalone = 0; size_t tag_b = 0, tag_e = 0;
    if (tend - tstart >= 4 && s[tstart] == '{' && s[tstart+1] == '%') {
      /* find %} within the trimmed span */
      for (size_t k = tstart + 2; k + 1 < tend; k++) {
        if (s[k] == '%' && s[k+1] == '}') {
          if (k + 2 == tend) { standalone = 1; tag_b = tstart; tag_e = k + 2; }
          break;
        }
      }
    }
    if (standalone) {
      buf_add(&b, s + tag_b, tag_e - tag_b);
    } else {
      buf_add(&b, s + ls, le - ls);
      if (le < n) buf_addc(&b, '\n');
    }
    i = le < n ? le + 1 : n;
  }
  Str r = arena_str(b.p, b.n);
  buf_free(&b);
  return r;
}

typedef struct { Str name; Node *nodes; int valid; Str err; int indep; int maxinc; Str cached; int has_cached; } ParsedTpl;
static ParsedTpl *parse_template(Str name, Str src) {
  ParsedTpl *pt = (ParsedTpl *)arena_alloc(sizeof(ParsedTpl));
  pt->name = name; pt->valid = 1; pt->err = str_make("", 0); pt->nodes = NULL;
  pt->indep = 0; pt->maxinc = 99; pt->cached = str_make("", 0); pt->has_cached = 0;
  Str clean = tpl_clean(src);
  TParse tp; tp.s = clean.s; tp.len = clean.n; tp.pos = 0; tp.err[0] = 0;
  int ek; pt->nodes = parse_seq(&tp, PS_NONE, &ek);
  if (tp.err[0]) { pt->valid = 0; pt->err = arena_str(tp.err, strlen(tp.err)); }
  return pt;
}

/* ---------------- template rendering ---------------- */
typedef struct {
  ParsedTpl **tpls; int ntpls;
} TplSet;
static ParsedTpl *tpl_find(TplSet *ts, Str name) {
  for (int i = 0; i < ts->ntpls; i++) if (str_eq(ts->tpls[i]->name, name)) return ts->tpls[i];
  return NULL;
}
typedef struct { Str varname; Value *elem; Value *loop; } Frame;
typedef struct { Frame *frames; int n, cap; Value *root; TplSet *ts; } Scope;
static Value *resolve_path(Scope *sc, Str *segs, int nsegs) {
  Value *v = NULL;
  Str first = segs[0];
  for (int i = sc->n - 1; i >= 0; i--) {
    Frame *f = &sc->frames[i];
    if (f->loop && str_eq_c(first, "loop")) { v = f->loop; break; }
    if (f->varname.n && str_eq(f->varname, first)) { v = f->elem; break; }
  }
  if (!v) {
    v = obj_field(sc->root, first);
    if (!v) return v_nil();
  }
  for (int k = 1; k < nsegs; k++) {
    if (v->t == V_OBJ) { Value *f2 = obj_field(v, segs[k]); if (!f2) return v_nil(); v = f2; }
    else if (v->t == V_ARR && str_eq_c(segs[k], "length")) { v = v_num(v->n); }
    else return v_nil();
  }
  return v;
}
static void render_nodes(Node *nd, Scope *sc, Buf *out, int depth);
static void render_nodes(Node *nd, Scope *sc, Buf *out, int depth) {
  for (; nd; nd = nd->next) {
    switch (nd->kind) {
      case N_TEXT: buf_adds(out, nd->text); break;
      case N_VAR: { Value *v = resolve_path(sc, nd->segs, nd->nsegs); v_write(v, out, !nd->raw); break; }
      case N_IF: {
        Value *v = resolve_path(sc, nd->segs, nd->nsegs);
        int t = v_truthy(v); if (nd->neg) t = !t;
        if (t) render_nodes(nd->then_branch, sc, out, depth);
        else if (nd->else_branch) render_nodes(nd->else_branch, sc, out, depth);
        break;
      }
      case N_FOR: {
        Value *v = resolve_path(sc, nd->segs, nd->nsegs);
        if (v->t != V_ARR) break;
        for (int i = 0; i < v->n; i++) {
          Field lf[4];
          lf[0].name = str_make("index", 5); lf[0].val = v_num(i + 1);
          lf[1].name = str_make("first", 5); lf[1].val = v_bool(i == 0);
          lf[2].name = str_make("last", 4); lf[2].val = v_bool(i == v->n - 1);
          lf[3].name = str_make("length", 6); lf[3].val = v_num(v->n);
          Value *loop = v_obj(lf, 4);
          Frame fr; fr.varname = nd->varname; fr.elem = v->items[i]; fr.loop = loop;
          sc->frames[sc->n++] = fr;
          render_nodes(nd->body, sc, out, depth);
          sc->n--;
        }
        break;
      }
      case N_INCLUDE: {
        if (depth >= 8) break;
        ParsedTpl *pt = tpl_find(sc->ts, nd->inc_name);
        if (!pt || !pt->valid) break;
        if (pt->indep && depth + pt->maxinc < 8) {
          if (pt->has_cached) { buf_adds(out, pt->cached); break; }
          size_t st = out->n;
          render_nodes(pt->nodes, sc, out, depth + 1);
          pt->cached = arena_str(out->p + st, out->n - st);
          pt->has_cached = 1;
          break;
        }
        render_nodes(pt->nodes, sc, out, depth + 1);
        break;
      }
    }
  }
}

/* ---- page-independence analysis for include caching ----
 * A template is page-independent (indep) if its rendered output depends only
 * on case-level data: root names site/recent/tags, or loop variables bound to
 * arrays under those names (incl. the built-in `loop` inside such a for).
 * Such a template renders identically on every page of a case, so its output
 * is rendered once and reused. maxinc = max include-nesting depth (99 if a
 * cycle exists); the cache is only used when depth + maxinc < 8 so the
 * depth-8 include cutoff cannot change the output. */
static int bound_has(Str *bound, int nbound, Str first) {
  for (int i = 0; i < nbound; i++) if (str_eq(bound[i], first)) return 1;
  return 0;
}
static int nodes_indep(Node *nd, TplSet *ts, Str *bound, int nbound) {
  for (; nd; nd = nd->next) {
    switch (nd->kind) {
      case N_TEXT: break;
      case N_VAR:
      case N_IF:
        if (nd->nsegs < 1 || !bound_has(bound, nbound, nd->segs[0])) return 0;
        if (nd->kind == N_IF) {
          if (!nodes_indep(nd->then_branch, ts, bound, nbound)) return 0;
          if (!nodes_indep(nd->else_branch, ts, bound, nbound)) return 0;
        }
        break;
      case N_FOR: {
        if (nd->nsegs < 1 || !bound_has(bound, nbound, nd->segs[0])) return 0;
        Str nb[8]; int nn = nbound;
        for (int i = 0; i < nbound && i < 8; i++) nb[i] = bound[i];
        if (nn < 8) nb[nn++] = nd->varname;
        if (nn < 8) nb[nn++] = str_make("loop", 4);
        if (!nodes_indep(nd->body, ts, nb, nn)) return 0;
        break;
      }
      case N_INCLUDE: {
        ParsedTpl *inc = tpl_find(ts, nd->inc_name);
        if (!inc || !inc->valid || !inc->indep) return 0;
        break;
      }
    }
  }
  return 1;
}
static int nodes_maxinc(Node *nd, TplSet *ts, int *state);
static int tpl_index(TplSet *ts, ParsedTpl *pt) {
  for (int i = 0; i < ts->ntpls; i++) if (ts->tpls[i] == pt) return i;
  return -1;
}
static int tpl_maxinc_at(TplSet *ts, int idx, int *state) {
  if (state[idx] == 2) return ts->tpls[idx]->maxinc;
  if (state[idx] == 1) return 99; /* include cycle */
  state[idx] = 1;
  int m = nodes_maxinc(ts->tpls[idx]->nodes, ts, state);
  ts->tpls[idx]->maxinc = m;
  state[idx] = 2;
  return m;
}
static int nodes_maxinc(Node *nd, TplSet *ts, int *state) {
  int m = 0;
  for (; nd; nd = nd->next) {
    switch (nd->kind) {
      case N_TEXT: case N_VAR: break;
      case N_IF: {
        int a = nodes_maxinc(nd->then_branch, ts, state);
        int b = nodes_maxinc(nd->else_branch, ts, state);
        if (a > m) m = a;
        if (b > m) m = b;
        break;
      }
      case N_FOR: {
        int a = nodes_maxinc(nd->body, ts, state);
        if (a > m) m = a;
        break;
      }
      case N_INCLUDE: {
        ParsedTpl *inc = tpl_find(ts, nd->inc_name);
        if (inc && inc->valid) {
          int j = tpl_index(ts, inc);
          int d = j < 0 ? 99 : tpl_maxinc_at(ts, j, state);
          if (d >= 99) return 99;
          if (1 + d > m) m = 1 + d;
        }
        break;
      }
    }
  }
  return m;
}
static void tpl_analyze(TplSet *ts) {
  Str base[3]; base[0] = str_make("site", 4); base[1] = str_make("recent", 6); base[2] = str_make("tags", 4);
  for (int i = 0; i < ts->ntpls; i++) { ts->tpls[i]->indep = ts->tpls[i]->valid ? 1 : 0; ts->tpls[i]->maxinc = 0; }
  int changed = 1;
  while (changed) {
    changed = 0;
    for (int i = 0; i < ts->ntpls; i++) {
      if (!ts->tpls[i]->indep) continue;
      if (!nodes_indep(ts->tpls[i]->nodes, ts, base, 3)) { ts->tpls[i]->indep = 0; changed = 1; }
    }
  }
  int *state = (int *)arena_alloc(sizeof(int) * (size_t)(ts->ntpls ? ts->ntpls : 1));
  for (int i = 0; i < ts->ntpls; i++) state[i] = 0;
  for (int i = 0; i < ts->ntpls; i++) if (ts->tpls[i]->valid) tpl_maxinc_at(ts, i, state);
}

/* ---------------- markdown + front matter ---------------- */
typedef struct Page Page;
typedef struct {
  const CaseInput *ci;
  Str base;               /* base_url with trailing slashes dropped */
  Page *pages; int npages; /* built pages in source order */
  /* broken-link dedup: array of (source,target) */
  Str *bl_src, *bl_tgt; int nbl, capbl;
  /* report lines */
  Str *rep; int nrep, caprep;
  /* built-page lookup for links: sorted by source path */
  Page **by_src; int nby_src;
  Str *static_paths; int nstatic;
} Build;
struct Page {
  Str src_path;      /* source file path */
  Str srcdir;        /* source dir (src_path minus last component), "" at root */
  Str dir;           /* output dir: "posts", "docs/a", or "" */
  Str slug; Str out_path; Str url;
  int is_post;
  Str title, summary, date; /* date may be "" */
  int has_date;
  Str *tags; int ntags; /* tag names as written */
  Str *tag_slugs; /* slug of each tag, computed once */
  Str body;          /* markdown body */
  Str layout;        /* effective layout template name */
  Str content;       /* rendered HTML */
  Str excerpt;       /* first paragraph inner HTML or "" */
  long long reading_minutes;
  struct Value *card; /* cached card value (built once, reused) */
  /* toc */
  struct { int level; Str id, html; } *toc; int ntoc;
  Page *newer, *older; /* in collection order (posts only) */
  int index_in_pages;
};

static void rep_add(Build *bd, Str line) {
  if (bd->nrep == bd->caprep) { bd->caprep = bd->caprep ? bd->caprep * 2 : 64; bd->rep = (Str *)realloc(bd->rep, sizeof(Str) * (size_t)bd->caprep); }
  bd->rep[bd->nrep++] = line;
}
static void rep_fmt(Build *bd, const char *fmt, Str a, Str b) {
  /* fmt uses %s for a and b (arena) — build line in a Buf then copy to arena */
  Buf lb; buf_init(&lb);
  const char *p = fmt;
  int which = 0;
  while (*p) {
    if (p[0] == '%' && p[1] == 's') { Str v = which++ == 0 ? a : b; buf_adds(&lb, v); p += 2; }
    else buf_addc(&lb, *p++);
  }
  Str line = arena_str(lb.p, lb.n);
  buf_free(&lb);
  rep_add(bd, line);
}
static void bl_add(Build *bd, Str src, Str tgt) {
  for (int i = 0; i < bd->nbl; i++) if (str_eq(bd->bl_src[i], src) && str_eq(bd->bl_tgt[i], tgt)) return;
  /* copy into arena: targets are views into transient render buffers */
  src = arena_dup(src); tgt = arena_dup(tgt);
  if (bd->nbl == bd->capbl) {
    bd->capbl = bd->capbl ? bd->capbl * 2 : 128;
    bd->bl_src = (Str *)realloc(bd->bl_src, sizeof(Str) * (size_t)bd->capbl);
    bd->bl_tgt = (Str *)realloc(bd->bl_tgt, sizeof(Str) * (size_t)bd->capbl);
  }
  bd->bl_src[bd->nbl] = src; bd->bl_tgt[bd->nbl] = tgt; bd->nbl++;
}

/* --- inline markdown --- */
static int is_escapable(char c) {
  switch (c) {
    case '`': case '\\': case '*': case '_': case '[': case ']': case '(': case ')':
    case '#': case '+': case '-': case '.': case '!': case '>': return 1;
    default: return 0;
  }
}
static void md_inline(Str s, Buf *out, Build *bd, Page *pg);
/* find next unescaped '`' at or after `from` (j-i>=1 char between caller) */
/* Count consecutive backslashes immediately before k, restricted to [from, k).
 * A special char at k is escaped iff this count is odd (pair-skipping). */
static size_t bs_before(Str s, size_t from, size_t k) {
  size_t b = k;
  while (b > from && s.s[b-1] == '\\') b--;
  return k - b;
}
static long find_code_close(Str s, size_t from) {
  size_t j = from;
  while (j < s.n) {
    const char *q = (const char *)memchr(s.s + j, '`', s.n - j);
    if (!q) return -1;
    size_t k = (size_t)(q - s.s);
    if ((bs_before(s, from, k) & 1) == 0) return (long)k;
    j = k + 1;
  }
  return -1;
}
static long find_strong_close(Str s, size_t from) {
  size_t j = from;
  while (j + 1 < s.n) {
    const char *q = (const char *)memchr(s.s + j, '*', s.n - j);
    if (!q) return -1;
    size_t k = (size_t)(q - s.s);
    if (k + 1 >= s.n) return -1;
    if ((bs_before(s, from, k) & 1) == 0 && s.s[k+1] == '*') return (long)k;
    j = k + 1;
  }
  return -1;
}
/* Find the closing * for emphasis opened at from-1 (from = opener+1).
 * While scanning, an escaped char is skipped as a pair, and a ** pair is
 * skipped as a pair; a lone * (not followed by *, not escaped) closes.
 * The char right after the opener cannot close (content must be >=1). */
static long find_em_close(Str s, size_t from) {
  size_t j = from;
  while (j < s.n) {
    const char *q = (const char *)memchr(s.s + j, '*', s.n - j);
    if (!q) return -1;
    size_t k = (size_t)(q - s.s);
    if ((bs_before(s, from, k) & 1) == 1) { j = k + 1; continue; }
    if (k + 1 < s.n && s.s[k+1] == '*') { j = k + 2; continue; }
    if (k == from) { j = k + 1; continue; }
    return (long)k;
  }
  return -1;
}
static long find_bracket_close(Str s, size_t from) {
  size_t j = from;
  while (j < s.n) {
    const char *q = (const char *)memchr(s.s + j, ']', s.n - j);
    if (!q) return -1;
    size_t k = (size_t)(q - s.s);
    if ((bs_before(s, from, k) & 1) == 0) return (long)k;
    j = k + 1;
  }
  return -1;
}
static int is_ws_char(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

/* Resolve a link target. Returns href Str (arena) and records broken links.
 * `raw` is the target as written. */
static Str resolve_link(Build *bd, Page *pg, Str raw) {
  /* split at first '#' */
  size_t hash = raw.n; int has_hash = 0;
  for (size_t i = 0; i < raw.n; i++) if (raw.s[i] == '#') { hash = i; has_hash = 1; break; }
  Str path = str_make(raw.s, hash);
  Str frag = has_hash ? str_make(raw.s + hash + 1, raw.n - hash - 1) : str_make("", 0);
  if (path.n == 0) return raw; /* fragment only: kept as written */
  if (path.s[0] == '/') return raw;
  /* scheme: ^[A-Za-z][A-Za-z0-9+.-]*: */
  {
    size_t i = 0;
    if (path.n > 0 && ((path.s[0] >= 'a' && path.s[0] <= 'z') || (path.s[0] >= 'A' && path.s[0] <= 'Z'))) {
      i = 1;
      while (i < path.n && ((path.s[i] >= 'a' && path.s[i] <= 'z') || (path.s[i] >= 'A' && path.s[i] <= 'Z') || (path.s[i] >= '0' && path.s[i] <= '9') || path.s[i] == '+' || path.s[i] == '-' || path.s[i] == '.')) i++;
      if (i < path.n && path.s[i] == ':') return raw;
    }
  }
  /* normalize: source dir + path (no malloc: build norm in the arena) */
  Str srcdir = pg->srcdir;
  /* segments stack: array of Str */
  Str segs[64]; int nseg = 0; int failed = 0;
  /* seed with srcdir segments */
  {
    size_t i = 0;
    while (i < srcdir.n) {
      size_t st = i;
      while (i < srcdir.n && srcdir.s[i] != '/') i++;
      if (i > st && nseg < 64) segs[nseg++] = str_make(srcdir.s + st, i - st);
      if (i < srcdir.n) i++;
    }
  }
  {
    size_t i = 0;
    while (i < path.n) {
      size_t st = i;
      while (i < path.n && path.s[i] != '/') i++;
      Str seg = str_make(path.s + st, i - st);
      if (seg.n == 0 || str_eq_c(seg, ".")) { /* skip */ }
      else if (str_eq_c(seg, "..")) { if (nseg == 0) failed = 1; else nseg--; }
      else { if (nseg < 64) segs[nseg++] = seg; else failed = 1; }
      if (i < path.n) i++;
    }
  }
  if (failed) { bl_add(bd, pg->src_path, raw); return raw; }
  size_t norm_n = 0;
  for (int i = 0; i < nseg; i++) norm_n += segs[i].n + (i ? 1 : 0);
  char *norm_p = (char *)arena_alloc(norm_n ? norm_n : 1);
  {
    size_t o = 0;
    for (int i = 0; i < nseg; i++) { if (i) norm_p[o++] = '/'; memcpy(norm_p + o, segs[i].s, segs[i].n); o += segs[i].n; }
  }
  Str norm = str_make(norm_p, norm_n);
  /* look up built page by source path */
  {
    int lo = 0, hi = bd->nby_src - 1;
    while (lo <= hi) {
      int mid = (lo + hi) >> 1;
      int c = str_cmp(norm, bd->by_src[mid]->src_path);
      if (c == 0) {
        Str u = bd->by_src[mid]->url;
        size_t rn = u.n + (has_hash ? 1 + frag.n : 0);
        char *rp = (char *)arena_alloc(rn ? rn : 1);
        memcpy(rp, u.s, u.n);
        if (has_hash) { rp[u.n] = '#'; memcpy(rp + u.n + 1, frag.s, frag.n); }
        return str_make(rp, rn);
      }
      if (c < 0) hi = mid - 1; else lo = mid + 1;
    }
  }
  /* static file? */
  for (int i = 0; i < bd->nstatic; i++) if (str_eq(norm, bd->static_paths[i])) {
    size_t rn = bd->base.n + 1 + norm.n + (has_hash ? 1 + frag.n : 0);
    char *rp = (char *)arena_alloc(rn ? rn : 1);
    memcpy(rp, bd->base.s, bd->base.n);
    rp[bd->base.n] = '/';
    memcpy(rp + bd->base.n + 1, norm.s, norm.n);
    if (has_hash) { rp[bd->base.n + 1 + norm.n] = '#'; memcpy(rp + bd->base.n + 1 + norm.n + 1, frag.s, frag.n); }
    return str_make(rp, rn);
  }
  bl_add(bd, pg->src_path, raw);
  return raw;
}

static void md_inline(Str s, Buf *out, Build *bd, Page *pg) {
  size_t i = 0;
  while (i < s.n) {
    /* fast path: bulk-copy a run of plain bytes (no markdown/HTML special) */
    size_t run = scan_plain(s.s + i, s.n - i);
    if (run) { buf_add(out, s.s + i, run); i += run; }
    if (i >= s.n) break;
    char c = s.s[i];
    if (c == '\\' && i + 1 < s.n && is_escapable(s.s[i+1])) {
      buf_add_escaped_char(out, s.s[i+1]); i += 2; continue;
    }
    if (c == '\\') { buf_addc(out, '\\'); i++; continue; }
    if (c == '`') {
      if (i + 1 < s.n && s.s[i+1] == '`') { buf_addc(out, '`'); i++; continue; }
      long j = find_code_close(s, i + 2);
      if (j < 0) { buf_addc(out, '`'); i++; continue; }
      buf_addlit(out, "<code>");
      buf_add_escaped(out, str_make(s.s + i + 1, (size_t)(j - (long)i - 1)));
      buf_addlit(out, "</code>");
      i = (size_t)j + 1; continue;
    }
    if (c == '*' && i + 1 < s.n && s.s[i+1] == '*') {
      long j = find_strong_close(s, i + 2);
      if (j < 0 || j == (long)i + 2) { buf_addlit(out, "**"); i += 2; continue; }
      buf_addlit(out, "<strong>");
      md_inline(str_make(s.s + i + 2, (size_t)(j - (long)i - 2)), out, bd, pg);
      buf_addlit(out, "</strong>");
      i = (size_t)j + 2; continue;
    }
    if (c == '*') {
      long j = find_em_close(s, i + 1);
      if (j < 0) { buf_addc(out, '*'); i++; continue; }
      buf_addlit(out, "<em>");
      md_inline(str_make(s.s + i + 1, (size_t)(j - (long)i - 1)), out, bd, pg);
      buf_addlit(out, "</em>");
      i = (size_t)j + 1; continue;
    }
    if (c == '[') {
      long j = find_bracket_close(s, i + 1);
      int ok = 0; long k = -1; Str target = str_make("", 0);
      if (j >= 0 && (size_t)j + 1 < s.n && s.s[j+1] == '(') {
        for (size_t t = (size_t)j + 2; t < s.n; t++) if (s.s[t] == ')') { k = (long)t; break; }
        if (k >= 0) {
          target = str_make(s.s + j + 2, (size_t)(k - j - 2));
          int hasws = 0;
          for (size_t t = 0; t < target.n; t++) if (is_ws_char(target.s[t])) { hasws = 1; break; }
          if (target.n > 0 && !hasws) ok = 1;
        }
      }
      if (!ok) { buf_addc(out, '['); i++; continue; }
      Str href = resolve_link(bd, pg, target);
      buf_addlit(out, "<a href=\"");
      buf_add_escaped(out, href);
      buf_addlit(out, "\">");
      md_inline(str_make(s.s + i + 1, (size_t)(j - (long)i - 1)), out, bd, pg);
      buf_addlit(out, "</a>");
      i = (size_t)k + 1; continue;
    }
    buf_add_escaped_char(out, c);
    i++;
  }
}

/* --- block markdown --- */
static int is_blank_line(Str l) {
  for (size_t i = 0; i < l.n; i++) if (l.s[i] != ' ' && l.s[i] != '\t') return 0;
  return 1;
}
static Str trim_line(Str l) {
  while (l.n && (l.s[0] == ' ' || l.s[0] == '\t')) { l.s++; l.n--; }
  while (l.n && (l.s[l.n-1] == ' ' || l.s[l.n-1] == '\t')) l.n--;
  return l;
}
/* fence opener? returns 1 and sets *lang (view, may be empty) */
static int fence_open(Str l, Str *lang) {
  if (l.n < 3 || l.s[0] != '`' || l.s[1] != '`' || l.s[2] != '`') return 0;
  size_t j = 3;
  while (j < l.n) {
    char c = l.s[j];
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '+' || c == '-') j++;
    else break;
  }
  for (size_t k = j; k < l.n; k++) if (l.s[k] != ' ' && l.s[k] != '\t') return 0;
  *lang = str_make(l.s + 3, j - 3);
  return 1;
}
static int fence_close(Str l) {
  if (l.n < 3 || l.s[0] != '`' || l.s[1] != '`' || l.s[2] != '`') return 0;
  for (size_t k = 3; k < l.n; k++) if (l.s[k] != ' ' && l.s[k] != '\t') return 0;
  return 1;
}
/* heading? returns level 0 if not; sets *text (trimmed raw text) */
static int heading_line(Str l, Str *text) {
  if (l.n == 0 || l.s[0] != '#') return 0;
  size_t i = 0; while (i < l.n && l.s[i] == '#') i++;
  if (i > 6) return 0;
  if (i >= l.n || (l.s[i] != ' ' && l.s[i] != '\t')) return 0;
  Str t = trim_line(str_make(l.s + i, l.n - i));
  if (t.n == 0) return 0;
  *text = t;
  return (int)i;
}
static int hr_line(Str l) {
  if (l.n < 3) return 0;
  if (!((l.s[0] == '-' && l.s[1] == '-' && l.s[2] == '-') || (l.s[0] == '*' && l.s[1] == '*' && l.s[2] == '*'))) return 0;
  for (size_t k = 3; k < l.n; k++) if (l.s[k] != ' ' && l.s[k] != '\t') return 0;
  return 1;
}
/* list item start? kind: 1 bullet, 2 ordered; sets *text and *num (ordered start) */
static int list_item(Str l, int *kind, Str *text, long long *num) {
  if (l.n >= 2 && (l.s[0] == '-' || l.s[0] == '*') && l.s[1] == ' ') {
    *kind = 1; *text = trim_line(str_make(l.s + 2, l.n - 2)); return 1;
  }
  size_t i = 0;
  while (i < l.n && l.s[i] >= '0' && l.s[i] <= '9') i++;
  if (i > 0 && i + 1 < l.n && l.s[i] == '.' && l.s[i+1] == ' ') {
    /* safe-integer check */
    long double v = 0; int overflow = 0;
    for (size_t k = 0; k < i; k++) { v = v * 10 + (l.s[k] - '0'); if (v > 9007199254740991.0L) overflow = 1; }
    if (overflow) return 0;
    long long n2 = 0; for (size_t k = 0; k < i; k++) n2 = n2 * 10 + (l.s[k] - '0');
    *kind = 2; *num = n2; *text = trim_line(str_make(l.s + i + 2, l.n - (i + 2))); return 1;
  }
  return 0;
}
static int is_ws_cp(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f' ||
    cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
    cp == 0x202F || cp == 0x205F || cp == 0x3000 || cp == 0xFEFF;
}
/* Vectorized word count: SSE2 16-byte chunks for ASCII, scalar fallback for
 * any chunk containing a byte >= 0x80 (exact Unicode-space semantics kept). */
static long long count_words(Str body) {
  static const unsigned char ws_tab[256] = {
    [' '] = 1, ['\t'] = 1, ['\n'] = 1, ['\r'] = 1, ['\v'] = 1, ['\f'] = 1,
  };
  long long words = 0;
  int inword = 0;
  const unsigned char *p = (const unsigned char *)body.s, *end = p + body.n;
  __m128i sp = _mm_set1_epi8(' '), tb = _mm_set1_epi8('\t'), nl = _mm_set1_epi8('\n'),
          cr = _mm_set1_epi8('\r'), vt = _mm_set1_epi8('\v'), ff = _mm_set1_epi8('\f'),
          hi = _mm_set1_epi8(0x7F);
  while (end - p >= 16) {
    __m128i v = _mm_loadu_si128((const __m128i *)p);
    __m128i w = _mm_or_si128(_mm_cmpeq_epi8(v, sp), _mm_cmpeq_epi8(v, tb));
    w = _mm_or_si128(w, _mm_cmpeq_epi8(v, nl));
    w = _mm_or_si128(w, _mm_cmpeq_epi8(v, cr));
    w = _mm_or_si128(w, _mm_cmpeq_epi8(v, vt));
    w = _mm_or_si128(w, _mm_cmpeq_epi8(v, ff));
    unsigned ws_mask = (unsigned)_mm_movemask_epi8(w);
    unsigned ascii_mask = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(v, hi), v));
    if (ascii_mask == 0xFFFF) {
      /* pure ASCII chunk: count word starts via transition mask */
      unsigned nonws = (~ws_mask) & 0xFFFF;
      unsigned prev_ws = (ws_mask << 1) | (inword ? 0u : 1u);
      words += __builtin_popcount(nonws & prev_ws);
      inword = (ws_mask >> 15) & 1 ? 0 : 1;
      p += 16;
      continue;
    }
    /* chunk has non-ASCII: scalar, byte by byte (exact semantics).
     * A multi-byte char may straddle the chunk end; advance p to where
     * the scalar scan actually stopped. */
    const char *q = (const char *)p, *qe = q + 16;
    while (q < qe) {
      unsigned char c = (unsigned char)*q;
      if (c < 0x80) { if (ws_tab[c]) inword = 0; else if (!inword) { words++; inword = 1; } q++; continue; }
      const char *save = q;
      uint32_t cp = utf8_decode(&q, body.s + body.n);
      if (cp == (uint32_t)(unsigned char)*save && c >= 0x80) { if (!inword) { words++; inword = 1; } continue; }
      if (is_ws_cp(cp)) inword = 0; else if (!inword) { words++; inword = 1; }
    }
    p = (const unsigned char *)q;
  }
  while (p < end) {
    unsigned char c = *p;
    if (c < 0x80) { if (ws_tab[c]) inword = 0; else if (!inword) { words++; inword = 1; } p++; continue; }
    const char *save = (const char *)p;
    uint32_t cp = utf8_decode((const char **)&p, body.s + body.n);
    if (cp == (uint32_t)(unsigned char)*save && c >= 0x80) { if (!inword) { words++; inword = 1; } continue; }
    if (is_ws_cp(cp)) inword = 0; else if (!inword) { words++; inword = 1; }
  }
  return words;
}

static void md_render(Build *bd, Page *pg) {
  Str body = pg->body;
  /* split body into lines: #lines = #newlines + 1 (trailing "" counts) */
  int nlines = 1;
  {
    const char *q = body.s, *e = body.s + body.n;
    while ((q = (const char *)memchr(q, '\n', (size_t)(e - q)))) { nlines++; q++; }
  }
  Str *lines = (Str *)arena_alloc(sizeof(Str) * (size_t)(nlines ? nlines : 1));
  {
    size_t i = 0; int nl = 0;
    while (1) {
      size_t st = i;
      const char *q = (const char *)memchr(body.s + i, '\n', body.n - i);
      size_t e = q ? (size_t)(q - body.s) : body.n;
      lines[nl++] = str_make(body.s + st, e - st);
      if (!q) break;
      i = e + 1;
      if (i == body.n) { lines[nl++] = str_make(body.s + body.n, 0); break; }
    }
    nlines = nl;
  }
  pg->reading_minutes = 1;
  {
    long long w = count_words(body);
    long long m = (w + 199) / 200;
    if (m < 1) m = 1;
    pg->reading_minutes = m;
  }
  /* classify each line ONCE (0 para, 1 blank, 2 fence-open, 3 fence-close,
   * 4 heading, 5 hr, 6 quote, 7 list) — same order as the block loop, so the
   * block/paragraph/list loops below just read the kind instead of
   * re-running every classifier on every line. */
  uint8_t *lkl = (uint8_t *)arena_alloc((size_t)(nlines ? nlines : 1));
  for (int li = 0; li < nlines; li++) {
    Str l = lines[li];
    uint8_t k = 0;
    Str dummy; int dk; long long dn;
    if (is_blank_line(l)) k = 1;
    else if (fence_open(l, &dummy)) k = fence_close(l) ? 3 : 2;
    else if (heading_line(l, &dummy)) k = 4;
    else if (hr_line(l)) k = 5;
    else if (l.n > 0 && l.s[0] == '>') k = 6;
    else if (list_item(l, &dk, &dummy, &dn)) k = 7;
    lkl[li] = k;
  }
  Buf out; buf_init(&out);
  pg->excerpt = str_make("", 0);
  int have_excerpt = 0;
  Str *taken = (Str *)arena_alloc(sizeof(Str) * 1024); int ntaken = 0;
  int i = 0;
  while (i < nlines) {
    Str l = lines[i];
    uint8_t lk = lkl[i];
    if (lk == 1) { i++; continue; }
    Str lang;
    if (lk == 2 || lk == 3) {
      fence_open(l, &lang);
      buf_addlit(&out, "<pre><code");
      if (lang.n) { buf_addlit(&out, " class=\"language-"); buf_adds(&out, lang); buf_addc(&out, '"'); }
      buf_addc(&out, '>');
      i++;
      while (i < nlines && lkl[i] != 3) {
        buf_add_escaped(&out, lines[i]); buf_addc(&out, '\n'); i++;
      }
      if (i < nlines) i++; /* closing fence */
      buf_addlit(&out, "</code></pre>\n");
      continue;
    }
    Str htext; int lvl = 0;
    if (lk == 4) lvl = heading_line(l, &htext);
    if (lvl) {
      Str id = slugify(htext);
      /* uniqueness in page */
      {
        int found = 1, k = 1;
        Str base = id;
        while (found) {
          found = 0;
          for (int t = 0; t < ntaken; t++) if (str_eq(taken[t], id)) { found = 1; break; }
          if (found) { k++; char tmp[32]; int nn = snprintf(tmp, sizeof tmp, "-%d", k); Buf tb; buf_init(&tb); buf_adds(&tb, base); buf_add(&tb, tmp, (size_t)nn); id = arena_str(tb.p, tb.n); buf_free(&tb); }
        }
        taken[ntaken++] = id;
      }
      buf_addlit(&out, "<h"); buf_addc(&out, (char)('0' + lvl)); buf_addlit(&out, " id=\"");
      buf_adds(&out, id); buf_addlit(&out, "\">");
      size_t hstart = out.n;
      md_inline(htext, &out, bd, pg);
      size_t hend = out.n;
      buf_addlit(&out, "</h"); buf_addc(&out, (char)('0' + lvl)); buf_addlit(&out, ">\n");
      if (lvl == 2 || lvl == 3) {
        /* append toc entry (grow by copy in arena); html = rendered span */
        typedef struct { int level; Str id, html; } TocE;
        TocE *na = (TocE *)arena_alloc(sizeof(TocE) * (size_t)(pg->ntoc + 1));
        for (int t = 0; t < pg->ntoc; t++) { na[t].level = pg->toc[t].level; na[t].id = pg->toc[t].id; na[t].html = pg->toc[t].html; }
        na[pg->ntoc].level = lvl; na[pg->ntoc].id = id; na[pg->ntoc].html = arena_str(out.p + hstart, hend - hstart);
        pg->toc = (void *)na;
        pg->ntoc++;
      }
      i++;
      continue;
    }
    if (lk == 5) { buf_addlit(&out, "<hr>\n"); i++; continue; }
    if (lk == 6) {
      Buf qb; buf_init(&qb);
      int first = 1;
      while (i < nlines && lkl[i] == 6) {
        Str t = str_make(lines[i].s + 1, lines[i].n - 1);
        if (t.n > 0 && t.s[0] == ' ') { t.s++; t.n--; }
        t = trim_line(t);
        if (t.n) { if (!first) buf_addc(&qb, '\n'); buf_adds(&qb, t); first = 0; }
        i++;
      }
      if (qb.n == 0) { buf_addlit(&out, "<blockquote>\n</blockquote>\n"); }
      else {
        buf_addlit(&out, "<blockquote>\n<p>");
        md_inline(str_make(qb.p, qb.n), &out, bd, pg);
        buf_addlit(&out, "</p>\n</blockquote>\n");
      }
      buf_free(&qb);
      continue;
    }
    {
      int kind; Str itext; long long num = 0;
      if (lk == 7 && list_item(l, &kind, &itext, &num)) {
        int lkind = kind; long long firstnum = num;
        Buf items; buf_init(&items);
        /* collect items: each item is a Buf of joined inline text */
        typedef struct { Buf b; } Item;
        Item *its = (Item *)arena_alloc(sizeof(Item) * 512); int nits = 0;
        while (1) {
          Item *it = &its[nits++];
          buf_init(&it->b);
          buf_adds(&it->b, itext);
          i++;
          /* continuation + next items */
          while (i < nlines) {
            Str nl = lines[i];
            if (lkl[i] == 1) break;
            int k2; Str tx2; long long n2 = 0;
            if (lkl[i] == 7 && list_item(nl, &k2, &tx2, &n2) && k2 == lkind) { itext = tx2; num = n2; goto next_item; }
            /* continuation: >=2 spaces then non-space */
            if (nl.n >= 2 && nl.s[0] == ' ' && nl.s[1] == ' ') {
              Str ct = trim_line(str_make(nl.s + 2, nl.n - 2));
              if (ct.n) { buf_addc(&it->b, '\n'); buf_adds(&it->b, ct); i++; continue; }
            }
            break;
          }
          break;
        next_item:;
          continue;
        }
        if (lkind == 1) buf_addlit(&out, "<ul>\n");
        else {
          if (firstnum == 1) buf_addlit(&out, "<ol>\n");
          else { buf_addlit(&out, "<ol start=\""); buf_addll(&out, firstnum); buf_addlit(&out, "\">\n"); }
        }
        for (int k = 0; k < nits; k++) {
          buf_addlit(&out, "<li>");
          md_inline(str_make(its[k].b.p, its[k].b.n), &out, bd, pg);
          buf_addlit(&out, "</li>\n");
          buf_free(&its[k].b);
        }
        if (lkind == 1) buf_addlit(&out, "</ul>\n"); else buf_addlit(&out, "</ol>\n");
        buf_free(&items);
        continue;
      }
    }
    /* paragraph: until blank / fence / heading / hr */
    {
      Buf pb; buf_init(&pb);
      int first = 1;
      while (i < nlines) {
        Str nl = lines[i];
        uint8_t pk = lkl[i];
        if (pk == 1 || pk == 2 || pk == 3 || pk == 4 || pk == 5) break;
        Str t = trim_line(nl);
        if (!first) buf_addc(&pb, '\n');
        buf_adds(&pb, t);
        first = 0;
        i++;
      }
      buf_addlit(&out, "<p>");
      size_t pstart = out.n;
      md_inline(str_make(pb.p, pb.n), &out, bd, pg);
      size_t pend = out.n;
      if (!have_excerpt) { pg->excerpt = arena_str(out.p + pstart, pend - pstart); have_excerpt = 1; }
      buf_addlit(&out, "</p>\n");
      buf_free(&pb);
      continue;
    }
  }
  pg->content = arena_str(out.p, out.n);
  buf_free(&out);
}

/* ---------------- front matter ---------------- */
/* Normalize CRLF / lone CR to LF for .md sources (arena copy). */
static Str normalize_crlf(Str c) {
  int need = 0;
  for (size_t i = 0; i < c.n; i++) if (c.s[i] == '\r') { need = 1; break; }
  if (!need) return c;
  Buf b; buf_init(&b);
  for (size_t i = 0; i < c.n; i++) {
    if (c.s[i] == '\r') { buf_addc(&b, '\n'); if (i + 1 < c.n && c.s[i+1] == '\n') i++; }
    else buf_addc(&b, c.s[i]);
  }
  Str r = arena_str(b.p, b.n);
  buf_free(&b);
  return r;
}
typedef struct {
  int ok;              /* 0 = refused, reason in err */
  char err[256];
  Str title, summary, date, layout, slug;
  int has_title, has_summary, has_date, has_layout, has_slug;
  int draft;
  Str *tags; int ntags;
  Str body;
} FM;
static int fm_key_known(Str k) {
  return str_eq_c(k, "title") || str_eq_c(k, "summary") || str_eq_c(k, "date") ||
    str_eq_c(k, "tags") || str_eq_c(k, "draft") || str_eq_c(k, "layout") || str_eq_c(k, "slug");
}
/* Parse a quoted string value (value starts with '"'). On success returns
 * the unquoted value; on failure writes reason into err and returns 0. */
static int parse_quoted(Str v, Str *out, char *err) {
  size_t i = 1;
  Buf b; buf_init(&b);
  while (i < v.n) {
    char c = v.s[i];
    if (c == '\\') {
      if (i + 1 >= v.n) { snprintf(err, 256, "a quoted string is not closed"); buf_free(&b); return 0; }
      char e = v.s[i+1];
      if (e == '"' || e == '\\') { buf_addc(&b, e); i += 2; continue; }
      snprintf(err, 256, "a quoted string holds an escape other than \\\" or \\\\");
      buf_free(&b); return 0;
    }
    if (c == '"') {
      if (i + 1 != v.n) { snprintf(err, 256, "a quoted string is not closed"); buf_free(&b); return 0; }
      *out = arena_str(b.p, b.n); buf_free(&b); return 1;
    }
    buf_addc(&b, c); i++;
  }
  snprintf(err, 256, "a quoted string is not closed");
  buf_free(&b); return 0;
}
static int valid_slug_value(Str v) {
  if (v.n == 0 || v.n > 64) return 0;
  /* ^[a-z0-9]+(-[a-z0-9]+)*$ */
  size_t i = 0;
  while (i < v.n) {
    if (!((v.s[i] >= 'a' && v.s[i] <= 'z') || (v.s[i] >= '0' && v.s[i] <= '9'))) return 0;
    while (i < v.n && ((v.s[i] >= 'a' && v.s[i] <= 'z') || (v.s[i] >= '0' && v.s[i] <= '9'))) i++;
    if (i < v.n) {
      if (v.s[i] != '-') return 0;
      i++;
      if (i >= v.n) return 0;
    }
  }
  return 1;
}
static int valid_layout_name(Str v) {
  if (v.n == 0) return 0;
  for (size_t i = 0; i < v.n; i++) {
    char c = v.s[i];
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) return 0;
  }
  return 1;
}
/* Parse front matter of a normalized .md content. Fills FM; on refusal
 * fm.ok=0 and fm.err holds the reason (without "refused <path>: "). */
static void parse_fm(Str content, FM *fm) {
  memset(fm, 0, sizeof *fm);
  fm->ok = 1; fm->title = str_make("", 0); fm->summary = str_make("", 0);
  fm->date = str_make("", 0); fm->layout = str_make("", 0); fm->slug = str_make("", 0);
  /* first line must be --- */
  size_t nl = 0; while (nl < content.n && content.s[nl] != '\n') nl++;
  if (!(nl == 3 && content.s[0] == '-' && content.s[1] == '-' && content.s[2] == '-')) {
    fm->ok = 0; snprintf(fm->err, sizeof fm->err, "no front matter"); return;
  }
  /* find closing line: a line exactly --- */
  size_t pos = nl + 1; /* start of line 2 */
  int closed = 0; size_t body_start = content.n; long lineno = 2;
  size_t fm_end = content.n;
  {
    size_t i = pos;
    while (i <= content.n) {
      size_t ls = i;
      while (i < content.n && content.s[i] != '\n') i++;
      size_t llen = i - ls;
      if (llen == 3 && content.s[ls] == '-' && content.s[ls+1] == '-' && content.s[ls+2] == '-') {
        closed = 1; fm_end = ls; body_start = (i < content.n) ? i + 1 : content.n; break;
      }
      if (i >= content.n) break;
      i++; lineno++;
    }
  }
  if (!closed) { fm->ok = 0; snprintf(fm->err, sizeof fm->err, "the front matter is not closed"); return; }
  /* parse FM lines between pos and fm_end */
  {
    size_t i = pos; long ln = 2;
    int seen_title = 0, seen_summary = 0, seen_date = 0, seen_tags = 0, seen_draft = 0, seen_layout = 0, seen_slug = 0;
    while (i < fm_end) {
      size_t ls = i;
      while (i < fm_end && content.s[i] != '\n') i++;
      Str line = str_make(content.s + ls, i - ls);
      if (i < fm_end) i++;
      /* blank or comment */
      int blank = 1;
      for (size_t k = 0; k < line.n; k++) if (line.s[k] != ' ' && line.s[k] != '\t') { blank = 0; break; }
      if (blank) { ln++; continue; }
      if (line.s[0] == '#') { ln++; continue; }
      /* key: value — key [a-z_]+, ": ", value */
      size_t ki = 0;
      while (ki < line.n && ((line.s[ki] >= 'a' && line.s[ki] <= 'z') || line.s[ki] == '_')) ki++;
      if (ki == 0 || ki >= line.n || line.s[ki] != ':') {
        fm->ok = 0; snprintf(fm->err, sizeof fm->err, "line %ld is not \"key: value\"", ln); return;
      }
      Str key = str_make(line.s, ki);
      Str val;
      if (ki + 1 == line.n) val = str_make("", 0); /* "key:" at EOL = empty value */
      else if (line.s[ki+1] == ' ') val = str_make(line.s + ki + 2, line.n - (ki + 2));
      else {
        fm->ok = 0; snprintf(fm->err, sizeof fm->err, "line %ld is not \"key: value\"", ln); return;
      }
      /* drop trailing spaces/tabs */
      while (val.n && (val.s[val.n-1] == ' ' || val.s[val.n-1] == '\t')) val.n--;
      if (!fm_key_known(key)) {
        fm->ok = 0;
        char tmp[64]; size_t cn = key.n < 60 ? key.n : 60; memcpy(tmp, key.s, cn); tmp[cn] = 0;
        snprintf(fm->err, sizeof fm->err, "unknown key \"%s\"", tmp); return;
      }
      int *seen = NULL;
      if (str_eq_c(key, "title")) seen = &seen_title;
      else if (str_eq_c(key, "summary")) seen = &seen_summary;
      else if (str_eq_c(key, "date")) seen = &seen_date;
      else if (str_eq_c(key, "tags")) seen = &seen_tags;
      else if (str_eq_c(key, "draft")) seen = &seen_draft;
      else if (str_eq_c(key, "layout")) seen = &seen_layout;
      else seen = &seen_slug;
      if (*seen) {
        fm->ok = 0;
        char tmp[64]; size_t cn = key.n < 60 ? key.n : 60; memcpy(tmp, key.s, cn); tmp[cn] = 0;
        snprintf(fm->err, sizeof fm->err, "key \"%s\" is given twice", tmp); return;
      }
      *seen = 1;
      if (val.n == 0) {
        fm->ok = 0;
        char tmp[64]; size_t cn = key.n < 60 ? key.n : 60; memcpy(tmp, key.s, cn); tmp[cn] = 0;
        snprintf(fm->err, sizeof fm->err, "key \"%s\" has no value", tmp); return;
      }
      char kerr[256];
      if (str_eq_c(key, "title") || str_eq_c(key, "summary")) {
        Str v;
        if (val.s[0] == '"') {
          if (!parse_quoted(val, &v, kerr)) {
            fm->ok = 0;
            char tmp[64]; size_t cn = key.n < 60 ? key.n : 60; memcpy(tmp, key.s, cn); tmp[cn] = 0;
            snprintf(fm->err, sizeof fm->err, "key \"%s\": %s", tmp, kerr); return;
          }
        } else if (val.s[0] == '[') {
          fm->ok = 0;
          char tmp[64]; size_t cn = key.n < 60 ? key.n : 60; memcpy(tmp, key.s, cn); tmp[cn] = 0;
          snprintf(fm->err, sizeof fm->err, "key \"%s\": a bare string opens on [", tmp); return;
        } else v = val;
        if (str_eq_c(key, "title")) { fm->title = v; fm->has_title = 1; }
        else { fm->summary = v; fm->has_summary = 1; }
      } else if (str_eq_c(key, "date")) {
        if (!valid_date_str(val)) {
          fm->ok = 0; snprintf(fm->err, sizeof fm->err, "key \"date\": not a date from 1970-01-01 to 2099-12-31 (YYYY-MM-DD)"); return;
        }
        fm->date = val; fm->has_date = 1;
      } else if (str_eq_c(key, "tags")) {
        if (val.n < 2 || val.s[0] != '[' || val.s[val.n-1] != ']') {
          fm->ok = 0; snprintf(fm->err, sizeof fm->err, "key \"tags\": not a list [a, b]"); return;
        }
        Str inner = str_make(val.s + 1, val.n - 2);
        Str ti = trim_str(inner);
        fm->tags = NULL; fm->ntags = 0;
        if (ti.n > 0) {
          /* split by ',' */
          int cap = 8, nt = 0;
          Str *arr = (Str *)arena_alloc(sizeof(Str) * 64);
          size_t st = 0;
          for (size_t k = 0; k <= ti.n; k++) {
            if (k == ti.n || ti.s[k] == ',') {
              Str el = trim_str(str_make(ti.s + st, k - st));
              int bad = el.n == 0;
              for (size_t q = 0; !bad && q < el.n; q++) if (el.s[q] == '"' || el.s[q] == '[' || el.s[q] == ']') bad = 1;
              if (bad) {
                fm->ok = 0; snprintf(fm->err, sizeof fm->err, "key \"tags\": an empty tag, or one holding \" [ or ]"); return;
              }
              arr[nt++] = el;
              st = k + 1;
            }
          }
          (void)cap;
          fm->tags = arr; fm->ntags = nt;
        }
      } else if (str_eq_c(key, "draft")) {
        if (str_eq_c(val, "true")) fm->draft = 1;
        else if (str_eq_c(val, "false")) fm->draft = 0;
        else { fm->ok = 0; snprintf(fm->err, sizeof fm->err, "key \"draft\": true or false"); return; }
      } else if (str_eq_c(key, "layout")) {
        if (!valid_layout_name(val)) {
          fm->ok = 0; snprintf(fm->err, sizeof fm->err, "key \"layout\": not a template name"); return;
        }
        fm->layout = val; fm->has_layout = 1;
      } else if (str_eq_c(key, "slug")) {
        if (!valid_slug_value(val)) {
          fm->ok = 0; snprintf(fm->err, sizeof fm->err, "key \"slug\": not a slug"); return;
        }
        fm->slug = val; fm->has_slug = 1;
      }
      ln++;
    }
  }
  if (!fm->has_title) { fm->ok = 0; snprintf(fm->err, sizeof fm->err, "no title"); return; }
  fm->body = str_make(content.s + body_start, content.n - body_start);
}

/* ---------------- card / page values ---------------- */
typedef struct { Str slug; Str name; Str url; Page **posts; int nposts, capposts; } Tag;
static Str tag_url_for(Build *bd, Tag *tags, int ntags, Str slug) {
  (void)bd;
  for (int i = 0; i < ntags; i++) if (str_eq(tags[i].slug, slug)) return tags[i].url;
  return str_make("", 0);
}
static Value *page_tag_vals(Build *bd, Page *pg, Tag *tags, int ntags) {
  /* page tags are deduplicated by slug, first occurrence wins, FM order */
  Str *seen = (Str *)arena_alloc(sizeof(Str) * (size_t)(pg->ntags ? pg->ntags : 1));
  int nseen = 0;
  Value **items = (Value **)arena_alloc(sizeof(Value *) * (size_t)(pg->ntags ? pg->ntags : 1));
  int nitems = 0;
  for (int i = 0; i < pg->ntags; i++) {
    Str slug = pg->tag_slugs[i];
    int dup = 0;
    for (int k = 0; k < nseen; k++) if (str_eq(seen[k], slug)) { dup = 1; break; }
    if (dup) continue;
    seen[nseen++] = slug;
    /* page tag name is the GLOBAL tag name (first spelling in collection
     * order); if no post carries this slug, keep the FM spelling */
    Str name = pg->tags[i];
    for (int k = 0; k < ntags; k++) if (str_eq(tags[k].slug, slug)) { name = tags[k].name; break; }
    Field *f = (Field *)arena_alloc(sizeof(Field) * 3);
    f[0].name = str_make("name", 4); f[0].val = v_str(name);
    f[1].name = str_make("slug", 4); f[1].val = v_str(slug);
    f[2].name = str_make("url", 3); f[2].val = v_str(tag_url_for(bd, tags, ntags, slug));
    items[nitems++] = v_obj(f, 3);
  }
  return v_arr(items, nitems);
}
static Value *build_card(Build *bd, Page *pg, Tag *tags, int ntags) {
  Field *f = (Field *)arena_alloc(sizeof(Field) * 8);
  f[0].name = str_make("title", 5); f[0].val = v_str(pg->title);
  f[1].name = str_make("url", 3); f[1].val = v_str(pg->url);
  f[2].name = str_make("date", 4); f[2].val = v_str(pg->date);
  f[3].name = str_make("date_long", 9); f[3].val = v_str(pg->has_date ? date_long(pg->date) : str_make("", 0));
  f[4].name = str_make("summary", 7); f[4].val = v_str(pg->summary);
  f[5].name = str_make("excerpt", 7); f[5].val = v_str(pg->excerpt);
  f[6].name = str_make("reading_minutes", 15); f[6].val = v_num(pg->reading_minutes);
  f[7].name = str_make("tags", 4); f[7].val = page_tag_vals(bd, pg, tags, ntags);
  return v_obj(f, 8);
}
static Value *neighbour_val(Page *p) {
  if (!p) return v_nil();
  Field *f = (Field *)arena_alloc(sizeof(Field) * 4);
  f[0].name = str_make("title", 5); f[0].val = v_str(p->title);
  f[1].name = str_make("url", 3); f[1].val = v_str(p->url);
  f[2].name = str_make("date", 4); f[2].val = v_str(p->date);
  f[3].name = str_make("date_long", 9); f[3].val = v_str(p->has_date ? date_long(p->date) : str_make("", 0));
  return v_obj(f, 4);
}
static Value *build_page_obj(Build *bd, Page *pg, Tag *tags, int ntags) {
  (void)bd; (void)tags; (void)ntags;
  Value *card = pg->card;
  Field *f = (Field *)arena_alloc(sizeof(Field) * 14);
  for (int i = 0; i < 8; i++) f[i] = card->fields[i];
  f[8].name = str_make("slug", 4); f[8].val = v_str(pg->slug);
  f[9].name = str_make("content", 7); f[9].val = v_str(pg->content);
  /* toc */
  Value **toc_items = (Value **)arena_alloc(sizeof(Value *) * (size_t)(pg->ntoc ? pg->ntoc : 1));
  for (int i = 0; i < pg->ntoc; i++) {
    Field *tf = (Field *)arena_alloc(sizeof(Field) * 3);
    tf[0].name = str_make("level", 5); tf[0].val = v_num(pg->toc[i].level);
    tf[1].name = str_make("id", 2); tf[1].val = v_str(pg->toc[i].id);
    tf[2].name = str_make("html", 4); tf[2].val = v_str(pg->toc[i].html);
    toc_items[i] = v_obj(tf, 3);
  }
  f[10].name = str_make("toc", 3); f[10].val = v_arr(toc_items, pg->ntoc);
  f[11].name = str_make("newer", 5); f[11].val = neighbour_val(pg->is_post ? pg->newer : NULL);
  f[12].name = str_make("older", 5); f[12].val = neighbour_val(pg->is_post ? pg->older : NULL);
  f[13].name = str_make("is_post", 7); f[13].val = v_bool(pg->is_post);
  return v_obj(f, 14);
}

/* ---------------- build ---------------- */
static int cmp_infile(const void *a, const void *b) { return str_cmp(((const InFile *)a)->path, ((const InFile *)b)->path); }
static int cmp_str(const void *a, const void *b) { return str_cmp(*(const Str *)a, *(const Str *)b); }
static int cmp_page_src(const void *a, const void *b) {
  Page *pa = *(Page **)a, *pb = *(Page **)b;
  return str_cmp(pa->src_path, pb->src_path);
}
static int cmp_page_coll(const void *a, const void *b) {
  Page *pa = *(Page **)a, *pb = *(Page **)b;
  int c = str_cmp(pb->date, pa->date); /* date desc */
  if (c) return c;
  return str_cmp(pa->slug, pb->slug);
}
static int cmp_tag(const void *a, const void *b) {
  const Tag *ta = (const Tag *)a, *tb = (const Tag *)b;
  if (ta->nposts != tb->nposts) return tb->nposts - ta->nposts;
  return str_cmp(ta->slug, tb->slug);
}
/* slug uniqueness per directory */
typedef struct { Str dir; Str *slugs; int n, cap; } DirTaken;
static Str unique_slug(DirTaken *dirs, int *ndirs, Str dir, Str base) {
  DirTaken *dt = NULL;
  for (int i = 0; i < *ndirs; i++) if (str_eq(dirs[i].dir, dir)) { dt = &dirs[i]; break; }
  if (!dt) {
    dt = &dirs[(*ndirs)++];
    dt->dir = dir; dt->slugs = NULL; dt->n = 0; dt->cap = 0;
    if (dir.n == 0) {
      /* root: posts, tags, page taken from the start */
      dt->slugs = (Str *)arena_alloc(sizeof(Str) * 16);
      dt->slugs[0] = str_make("posts", 5); dt->slugs[1] = str_make("tags", 4); dt->slugs[2] = str_make("page", 4);
      dt->n = 3; dt->cap = 16;
    }
  }
  Str cand = base;
  int k = 1;
  for (;;) {
    int taken = 0;
    for (int i = 0; i < dt->n; i++) if (str_eq(dt->slugs[i], cand)) { taken = 1; break; }
    if (!taken) break;
    k++;
    char tmp[32]; int nn = snprintf(tmp, sizeof tmp, "-%d", k);
    Buf b; buf_init(&b); buf_adds(&b, base); buf_add(&b, tmp, (size_t)nn);
    cand = arena_str(b.p, b.n); buf_free(&b);
  }
  if (dt->n == dt->cap) {
    Str *na = (Str *)arena_alloc(sizeof(Str) * (size_t)(dt->cap ? dt->cap * 2 : 16));
    for (int i = 0; i < dt->n; i++) na[i] = dt->slugs[i];
    dt->slugs = na; dt->cap = dt->cap ? dt->cap * 2 : 16;
  }
  dt->slugs[dt->n++] = cand;
  return cand;
}

/* output tree: ordered list of (path, text) */
typedef struct { Str path, text; } OutEnt;
typedef struct { OutEnt *e; int n, cap; } OutTree;
static void out_add(OutTree *t, Str path, Str text) {
  if (t->n == t->cap) { t->cap = t->cap ? t->cap * 2 : 64; t->e = (OutEnt *)realloc(t->e, sizeof(OutEnt) * (size_t)t->cap); }
  t->e[t->n].path = path; t->e[t->n].text = text; t->n++;
}
static int out_has(OutTree *t, Str path) {
  for (int i = 0; i < t->n; i++) if (str_eq(t->e[i].path, path)) return 1;
  return 0;
}

/* Build one case into an OutTree (arena-allocated strings). */
static void build_case(const CaseInput *ci, OutTree *tree) {
  arena_init();
  Build bd; memset(&bd, 0, sizeof bd);
  bd.ci = ci;
  /* base: base_url with trailing slashes dropped */
  {
    Str b = ci->base_url;
    while (b.n && b.s[b.n-1] == '/') b.n--;
    bd.base = b;
  }
  /* sort files by path */
  InFile *files = (InFile *)arena_alloc(sizeof(InFile) * (size_t)(ci->nfiles ? ci->nfiles : 1));
  for (int i = 0; i < ci->nfiles; i++) files[i] = ci->files[i];
  qsort(files, (size_t)ci->nfiles, sizeof(InFile), cmp_infile);
  /* templates: parse in name order */
  TplSet ts; ts.ntpls = ci->ntemplates;
  ts.tpls = (ParsedTpl **)arena_alloc(sizeof(ParsedTpl *) * (size_t)(ci->ntemplates ? ci->ntemplates : 1));
  for (int i = 0; i < ci->ntemplates; i++) {
    ts.tpls[i] = parse_template(ci->templates[i].name, ci->templates[i].src);
    if (!ts.tpls[i]->valid) {
      Buf lb; buf_init(&lb);
      buf_addlit(&lb, "template \""); buf_adds(&lb, ci->templates[i].name);
      buf_addlit(&lb, "\": "); buf_adds(&lb, ts.tpls[i]->err);
      rep_add(&bd, arena_str(lb.p, lb.n)); buf_free(&lb);
    }
  }
  tpl_analyze(&ts);
  /* static paths (sorted) */
  int nstatic = 0;
  for (int i = 0; i < ci->nfiles; i++) if (!str_ends(files[i].path, ".md")) nstatic++;
  bd.static_paths = (Str *)arena_alloc(sizeof(Str) * (size_t)(nstatic ? nstatic : 1));
  bd.nstatic = 0;
  for (int i = 0; i < ci->nfiles; i++) if (!str_ends(files[i].path, ".md")) bd.static_paths[bd.nstatic++] = files[i].path;
  qsort(bd.static_paths, (size_t)bd.nstatic, sizeof(Str), cmp_str);
  /* pages */
  Page *pages = (Page *)arena_alloc(sizeof(Page) * (size_t)(ci->nfiles ? ci->nfiles : 1));
  int npages = 0;
  DirTaken *dirs = (DirTaken *)arena_alloc(sizeof(DirTaken) * 64); int ndirs = 0;
  int nposts_written = 0, npages_written = 0;
  for (int fi = 0; fi < ci->nfiles; fi++) {
    InFile *f = &files[fi];
    if (!str_ends(f->path, ".md")) continue;
    Str content = normalize_crlf(f->content);
    FM fm; parse_fm(content, &fm);
    int is_post = str_starts(f->path, "posts/");
    Str layout_name;
    if (!fm.ok) {
      Buf lb; buf_init(&lb);
      buf_addlit(&lb, "refused "); buf_adds(&lb, f->path); buf_addlit(&lb, ": "); buf_addlit(&lb, fm.err);
      rep_add(&bd, arena_str(lb.p, lb.n)); buf_free(&lb);
      continue;
    }
    if (str_starts(f->path, "tags/")) {
      Buf lb; buf_init(&lb);
      buf_addlit(&lb, "refused "); buf_adds(&lb, f->path); buf_addlit(&lb, ": tags/ is the generator's own");
      rep_add(&bd, arena_str(lb.p, lb.n)); buf_free(&lb);
      continue;
    }
    if (str_starts(f->path, "page/")) {
      Buf lb; buf_init(&lb);
      buf_addlit(&lb, "refused "); buf_adds(&lb, f->path); buf_addlit(&lb, ": page/ is the generator's own");
      rep_add(&bd, arena_str(lb.p, lb.n)); buf_free(&lb);
      continue;
    }
    if (is_post && !fm.has_date) {
      Buf lb; buf_init(&lb);
      buf_addlit(&lb, "refused "); buf_adds(&lb, f->path); buf_addlit(&lb, ": a post needs a date");
      rep_add(&bd, arena_str(lb.p, lb.n)); buf_free(&lb);
      continue;
    }
    layout_name = fm.has_layout ? fm.layout : (is_post ? str_make("post", 4) : str_make("page", 4));
    {
      ParsedTpl *lt = tpl_find(&ts, layout_name);
      if (!lt || !lt->valid) {
        Buf lb; buf_init(&lb);
        buf_addlit(&lb, "refused "); buf_adds(&lb, f->path);
        buf_addlit(&lb, ": layout \""); buf_adds(&lb, layout_name);
        buf_addlit(&lb, "\" is no template that parses");
        rep_add(&bd, arena_str(lb.p, lb.n)); buf_free(&lb);
        continue;
      }
    }
    if (fm.draft) {
      Buf lb; buf_init(&lb);
      buf_addlit(&lb, "draft "); buf_adds(&lb, f->path);
      rep_add(&bd, arena_str(lb.p, lb.n)); buf_free(&lb);
      continue;
    }
    Page *pg = &pages[npages];
    memset(pg, 0, sizeof *pg);
    pg->src_path = f->path;
    /* srcdir: src_path minus last component ("" at root) */
    {
      size_t slash = 0; int has_slash = 0;
      for (size_t i = 0; i < f->path.n; i++) if (f->path.s[i] == '/') { slash = i; has_slash = 1; }
      pg->srcdir = has_slash ? str_make(f->path.s, slash) : str_make("", 0);
    }
    pg->is_post = is_post;
    pg->title = fm.title; pg->summary = fm.summary;
    pg->date = fm.has_date ? fm.date : str_make("", 0);
    pg->has_date = fm.has_date;
    pg->tags = fm.tags; pg->ntags = fm.ntags;
    if (fm.ntags) {
      pg->tag_slugs = (Str *)arena_alloc(sizeof(Str) * (size_t)fm.ntags);
      for (int ti = 0; ti < fm.ntags; ti++) pg->tag_slugs[ti] = slugify(fm.tags[ti]);
    }
    pg->body = fm.body;
    pg->layout = layout_name;
    pg->index_in_pages = npages;
    /* dir + slug */
    Str dir;
    if (is_post) dir = str_make("posts", 5);
    else {
      size_t slash = 0; int hs = 0;
      for (size_t i = 0; i < f->path.n; i++) if (f->path.s[i] == '/') { slash = i; hs = 1; }
      dir = hs ? str_make(f->path.s, slash) : str_make("", 0);
    }
    pg->dir = dir;
    Str base_slug = fm.has_slug ? fm.slug : slugify(fm.title);
    pg->slug = unique_slug(dirs, &ndirs, dir, base_slug);
    {
      Buf pb; buf_init(&pb);
      if (dir.n) { buf_adds(&pb, dir); buf_addc(&pb, '/'); }
      buf_adds(&pb, pg->slug); buf_addlit(&pb, "/index.html");
      pg->out_path = arena_str(pb.p, pb.n); buf_free(&pb);
      Buf ub; buf_init(&ub);
      buf_adds(&ub, bd.base); buf_addc(&ub, '/');
      if (dir.n) { buf_adds(&ub, dir); buf_addc(&ub, '/'); }
      buf_adds(&ub, pg->slug); buf_addc(&ub, '/');
      pg->url = arena_str(ub.p, ub.n); buf_free(&ub);
    }
    npages++;
    if (is_post) nposts_written++; else npages_written++;
  }
  bd.pages = pages; bd.npages = npages;
  /* by_src sorted for link lookup */
  bd.by_src = (Page **)arena_alloc(sizeof(Page *) * (size_t)(npages ? npages : 1));
  for (int i = 0; i < npages; i++) bd.by_src[i] = &pages[i];
  qsort(bd.by_src, (size_t)npages, sizeof(Page *), cmp_page_src);
  bd.nby_src = npages;
  /* markdown render for every page */
  for (int i = 0; i < npages; i++) md_render(&bd, &pages[i]);
  /* collections: posts sorted newest first, equal dates by slug */
  Page **posts = (Page **)arena_alloc(sizeof(Page *) * (size_t)(nposts_written ? nposts_written : 1));
  int nposts = 0;
  for (int i = 0; i < npages; i++) if (pages[i].is_post) posts[nposts++] = &pages[i];
  qsort(posts, (size_t)nposts, sizeof(Page *), cmp_page_coll);
  for (int i = 0; i < nposts; i++) {
    posts[i]->newer = i > 0 ? posts[i-1] : NULL;
    posts[i]->older = i + 1 < nposts ? posts[i+1] : NULL;
  }
  /* tags */
  Tag *tags = (Tag *)arena_alloc(sizeof(Tag) * 1024); int ntags = 0;
  for (int i = 0; i < nposts; i++) {
    Page *p = posts[i];
    for (int t = 0; t < p->ntags; t++) {
      Str tslug = p->tag_slugs[t];
      Tag *tg = NULL;
      for (int k = 0; k < ntags; k++) if (str_eq(tags[k].slug, tslug)) { tg = &tags[k]; break; }
      if (!tg) {
        tg = &tags[ntags++];
        memset(tg, 0, sizeof *tg);
        tg->slug = tslug; tg->name = p->tags[t];
        Buf ub; buf_init(&ub); buf_adds(&ub, bd.base); buf_addlit(&ub, "/tags/"); buf_adds(&ub, tslug); buf_addc(&ub, '/');
        tg->url = arena_str(ub.p, ub.n); buf_free(&ub);
        tg->posts = (Page **)arena_alloc(sizeof(Page *) * 64); tg->capposts = 64;
      }
      if (tg->nposts > 0 && tg->posts[tg->nposts-1] == p) continue; /* same post twice */
      if (tg->nposts == tg->capposts) {
        Page **na = (Page **)arena_alloc(sizeof(Page *) * (size_t)(tg->capposts * 2));
        for (int q = 0; q < tg->nposts; q++) na[q] = tg->posts[q];
        tg->posts = na; tg->capposts *= 2;
      }
      tg->posts[tg->nposts++] = p;
    }
  }
  qsort(tags, (size_t)ntags, sizeof(Tag), cmp_tag);
  /* tag lookup helper: url for a slug */
  /* ---- value builders ---- */
  /* site object */
  Field sitef[2];
  sitef[0].name = str_make("title", 5); sitef[0].val = v_str(ci->title);
  sitef[1].name = str_make("base_url", 8); sitef[1].val = v_str(bd.base);
  Value *site_v = v_obj(sitef, 2);
  /* tags array (objects) */
  Value **tag_vals = (Value **)arena_alloc(sizeof(Value *) * (size_t)(ntags ? ntags : 1));
  for (int i = 0; i < ntags; i++) {
    Field *tf = (Field *)arena_alloc(sizeof(Field) * 4);
    tf[0].name = str_make("name", 4); tf[0].val = v_str(tags[i].name);
    tf[1].name = str_make("slug", 4); tf[1].val = v_str(tags[i].slug);
    tf[2].name = str_make("url", 3); tf[2].val = v_str(tags[i].url);
    tf[3].name = str_make("count", 5); tf[3].val = v_num(tags[i].nposts);
    tag_vals[i] = v_obj(tf, 4);
  }
  Value *tags_v = v_arr(tag_vals, ntags);
  /* build each page's card once; index/tag/recent pages reuse it */
  for (int i = 0; i < npages; i++) pages[i].card = build_card(&bd, &pages[i], tags, ntags);
  /* recent cards */
  int nrecent = (int)(ci->recent < nposts ? ci->recent : nposts);
  if (ci->recent < 0) nrecent = 0;
  Value **recent_vals = (Value **)arena_alloc(sizeof(Value *) * (size_t)(nrecent ? nrecent : 1));
  /* card builder (needs tags) */
  /* forward: build card for a page */
  /* helper lambdas via local functions are not available in C89 style; inline below */
  /* We build cards on demand with a function-like macro through a helper. */
  /* ---- render pages ---- */
  for (int i = 0; i < nrecent; i++) recent_vals[i] = posts[i]->card;
  Value *recent_v = v_arr(recent_vals, nrecent);
  Frame *frames = (Frame *)arena_alloc(sizeof(Frame) * 64);
  for (int pi = 0; pi < npages; pi++) {
    Page *pg = &pages[pi];
    ParsedTpl *lt = tpl_find(&ts, pg->layout);
    Value *page_v = build_page_obj(&bd, pg, tags, ntags);
    Field rf[4];
    rf[0].name = str_make("site", 4); rf[0].val = site_v;
    rf[1].name = str_make("tags", 4); rf[1].val = tags_v;
    rf[2].name = str_make("recent", 6); rf[2].val = recent_v;
    rf[3].name = str_make("page", 4); rf[3].val = page_v;
    Value *root = v_obj(rf, 4);
    Scope sc; sc.frames = frames; sc.n = 0; sc.cap = 64; sc.root = root; sc.ts = &ts;
    Buf ob; buf_init(&ob);
    render_nodes(lt->nodes, &sc, &ob, 0);
    out_add(tree, pg->out_path, arena_str(ob.p, ob.n));
    buf_free(&ob);
  }
  /* ---- index pages ---- */
  {
    ParsedTpl *idx = tpl_find(&ts, str_make("index", 5));
    long long pp = ci->per_page;
    long long total = 1;
    if (pp > 0) { total = (nposts + pp - 1) / pp; if (total < 1) total = 1; }
    if (idx && idx->valid) {
      for (long long k = 1; k <= total; k++) {
        long long start = pp > 0 ? (k - 1) * pp : 0;
        long long cnt = 0;
        if (pp > 0 && start < nposts) { cnt = nposts - start; if (cnt > pp) cnt = pp; }
        Value **pv = (Value **)arena_alloc(sizeof(Value *) * (size_t)(cnt ? cnt : 1));
        for (long long q = 0; q < cnt; q++) pv[q] = posts[start + q]->card;
        Value *posts_v = v_arr(pv, (int)cnt);
        /* pager */
        Buf ub; buf_init(&ub);
        buf_adds(&ub, bd.base); buf_addc(&ub, '/');
        Str purl = arena_str(ub.p, ub.n); buf_free(&ub);
        Str purl_k, newer_u, older_u;
        if (k == 1) { purl_k = purl; newer_u = str_make("", 0); }
        else {
          Buf b2; buf_init(&b2); buf_adds(&b2, bd.base);
          if (k == 2) buf_addc(&b2, '/');
          else { buf_addlit(&b2, "/page/"); buf_addll(&b2, k - 1); buf_addc(&b2, '/'); }
          newer_u = arena_str(b2.p, b2.n); buf_free(&b2);
          Buf b3; buf_init(&b3); buf_adds(&b3, bd.base); buf_addlit(&b3, "/page/"); buf_addll(&b3, k); buf_addc(&b3, '/');
          purl_k = arena_str(b3.p, b3.n); buf_free(&b3);
        }
        if (k < total) {
          Buf b4; buf_init(&b4); buf_adds(&b4, bd.base); buf_addlit(&b4, "/page/"); buf_addll(&b4, k + 1); buf_addc(&b4, '/');
          older_u = arena_str(b4.p, b4.n); buf_free(&b4);
        } else older_u = str_make("", 0);
        Field pf[5];
        pf[0].name = str_make("number", 6); pf[0].val = v_num(k);
        pf[1].name = str_make("total", 5); pf[1].val = v_num(total);
        pf[2].name = str_make("url", 3); pf[2].val = v_str(purl_k);
        pf[3].name = str_make("newer_url", 9); pf[3].val = v_str(newer_u);
        pf[4].name = str_make("older_url", 9); pf[4].val = v_str(older_u);
        Value *pager_v = v_obj(pf, 5);
        Field rf[5];
        rf[0].name = str_make("site", 4); rf[0].val = site_v;
        rf[1].name = str_make("tags", 4); rf[1].val = tags_v;
        rf[2].name = str_make("recent", 6); rf[2].val = recent_v;
        rf[3].name = str_make("posts", 5); rf[3].val = posts_v;
        rf[4].name = str_make("pager", 5); rf[4].val = pager_v;
        Value *root = v_obj(rf, 5);
        Scope sc; sc.frames = frames; sc.n = 0; sc.cap = 64; sc.root = root; sc.ts = &ts;
        Buf ob; buf_init(&ob);
        render_nodes(idx->nodes, &sc, &ob, 0);
        Str opath;
        if (k == 1) opath = str_make("index.html", 10);
        else {
          Buf b5; buf_init(&b5); buf_addlit(&b5, "page/"); buf_addll(&b5, k); buf_addlit(&b5, "/index.html");
          opath = arena_str(b5.p, b5.n); buf_free(&b5);
        }
        out_add(tree, opath, arena_str(ob.p, ob.n));
        buf_free(&ob);
      }
    }
  }
  /* ---- tag pages + tags index ---- */
  {
    ParsedTpl *tagt = tpl_find(&ts, str_make("tag", 3));
    if (tagt && tagt->valid) {
      for (int ti = 0; ti < ntags; ti++) {
        Tag *tg = &tags[ti];
        Value **pv = (Value **)arena_alloc(sizeof(Value *) * (size_t)(tg->nposts ? tg->nposts : 1));
        for (int q = 0; q < tg->nposts; q++) pv[q] = tg->posts[q]->card;
        Value *posts_v = v_arr(pv, tg->nposts);
        Field tf[5];
        tf[0].name = str_make("name", 4); tf[0].val = v_str(tg->name);
        tf[1].name = str_make("slug", 4); tf[1].val = v_str(tg->slug);
        tf[2].name = str_make("url", 3); tf[2].val = v_str(tag_url_for(&bd, tags, ntags, tg->slug));
        tf[3].name = str_make("count", 5); tf[3].val = v_num(tg->nposts);
        tf[4].name = str_make("posts", 5); tf[4].val = posts_v;
        Value *tag_v = v_obj(tf, 5);
        Field rf[4];
        rf[0].name = str_make("site", 4); rf[0].val = site_v;
        rf[1].name = str_make("tags", 4); rf[1].val = tags_v;
        rf[2].name = str_make("recent", 6); rf[2].val = recent_v;
        rf[3].name = str_make("tag", 3); rf[3].val = tag_v;
        Value *root = v_obj(rf, 4);
        Scope sc; sc.frames = frames; sc.n = 0; sc.cap = 64; sc.root = root; sc.ts = &ts;
        Buf ob; buf_init(&ob);
        render_nodes(tagt->nodes, &sc, &ob, 0);
        Buf pb; buf_init(&pb); buf_addlit(&pb, "tags/"); buf_adds(&pb, tg->slug); buf_addlit(&pb, "/index.html");
        out_add(tree, arena_str(pb.p, pb.n), arena_str(ob.p, ob.n));
        buf_free(&pb); buf_free(&ob);
      }
    }
    ParsedTpl *tagst = tpl_find(&ts, str_make("tags", 4));
    if (tagst && tagst->valid) {
      Field rf[3];
      rf[0].name = str_make("site", 4); rf[0].val = site_v;
      rf[1].name = str_make("tags", 4); rf[1].val = tags_v;
      rf[2].name = str_make("recent", 6); rf[2].val = recent_v;
      Value *root = v_obj(rf, 3);
      Scope sc; sc.frames = frames; sc.n = 0; sc.cap = 64; sc.root = root; sc.ts = &ts;
      Buf ob; buf_init(&ob);
      render_nodes(tagst->nodes, &sc, &ob, 0);
      out_add(tree, str_make("tags/index.html", 15), arena_str(ob.p, ob.n));
      buf_free(&ob);
    }
  }
  /* ---- static files ---- */
  for (int i = 0; i < ci->nfiles; i++) {
    if (str_ends(files[i].path, ".md")) continue;
    if (out_has(tree, files[i].path)) {
      Buf lb; buf_init(&lb);
      buf_addlit(&lb, "shadowed "); buf_adds(&lb, files[i].path);
      buf_addlit(&lb, ": a built page has its path");
      rep_add(&bd, arena_str(lb.p, lb.n)); buf_free(&lb);
    } else {
      out_add(tree, files[i].path, files[i].content);
    }
  }
  /* ---- report ---- */
  for (int i = 0; i < bd.nbl; i++) {
    Buf lb; buf_init(&lb);
    buf_addlit(&lb, "broken-link "); buf_adds(&lb, bd.bl_src[i]);
    buf_addlit(&lb, " -> "); buf_adds(&lb, bd.bl_tgt[i]);
    rep_add(&bd, arena_str(lb.p, lb.n)); buf_free(&lb);
  }
  qsort(bd.rep, (size_t)bd.nrep, sizeof(Str), cmp_str);
  {
    Buf rb; buf_init(&rb);
    for (int i = 0; i < bd.nrep; i++) { buf_adds(&rb, bd.rep[i]); buf_addc(&rb, '\n'); }
    buf_addlit(&rb, "built ");
    buf_addll(&rb, npages_written); buf_addlit(&rb, " pages, ");
    buf_addll(&rb, nposts_written); buf_addlit(&rb, " posts, ");
    buf_addll(&rb, ntags); buf_addlit(&rb, " tags and ");
    buf_addll(&rb, bd.nstatic); buf_addlit(&rb, " static files\n");
    out_add(tree, str_make("build-report.txt", 16), arena_str(rb.p, rb.n));
    buf_free(&rb);
  }
  free(bd.rep); free(bd.bl_src); free(bd.bl_tgt);
}

/* ---------------- driver: one case -> JSON object text ---------------- */
static void render_case_json(const CaseInput *ci, Buf *out) {
  arena_reset();
  OutTree tree; memset(&tree, 0, sizeof tree);
  build_case(ci, &tree);
  buf_addc(out, '{');
  for (int i = 0; i < tree.n; i++) {
    if (i) buf_addc(out, ',');
    buf_add_json_escaped(out, tree.e[i].path);
    buf_addc(out, ':');
    buf_add_json_escaped(out, tree.e[i].text);
  }
  buf_addc(out, '}');
  free(tree.e);
}

/* ---------------- file IO ---------------- */
static char *read_all(const char *path, size_t *out_n) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return NULL;
  struct stat st;
  if (fstat(fd, &st) != 0) { close(fd); return NULL; }
  size_t n = st.st_size > 0 ? (size_t)st.st_size : 0;
  char *buf = (char *)malloc(n + 1);
  if (!buf) { close(fd); return NULL; }
  size_t got = 0;
  while (got < n) {
    ssize_t r = read(fd, buf + got, n - got);
    if (r <= 0) break;
    got += (size_t)r;
  }
  close(fd);
  buf[got] = 0;
  *out_n = got;
  return buf;
}
/* mkdir -p equivalent for a directory path (tolerates existing dirs). */
static void mkdir_p(const char *dir) {
  char tmp[4096];
  size_t n = strlen(dir);
  if (n == 0 || n >= sizeof tmp) return;
  memcpy(tmp, dir, n + 1);
  for (size_t i = 1; i <= n; i++) {
    if (tmp[i] == '/' || tmp[i] == '\0') {
      char save = tmp[i]; tmp[i] = '\0';
      mkdir(tmp, 0755);
      tmp[i] = save;
    }
  }
}
static int write_all(const char *path, const char *data, size_t n) {
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return -1;
  size_t off = 0;
  while (off < n) {
    ssize_t w = write(fd, data + off, n - off);
    if (w <= 0) { close(fd); return -1; }
    off += (size_t)w;
  }
  if (close(fd) != 0) return -1;
  return 0;
}
static char *join_path(const char *dir, const char *name) {
  size_t a = strlen(dir), b = strlen(name);
  char *p = (char *)malloc(a + b + 2);
  memcpy(p, dir, a);
  if (a && p[a-1] != '/') p[a++] = '/';
  memcpy(p + a, name, b + 1);
  return p;
}

/* Run all cases in <inputDir>/cases.json, write <outputDir>/answers.json.
 * Returns 0 on success. */
static int run_all(const char *input_dir, const char *output_dir) {
  arena_init();
  char *in_path = join_path(input_dir, "cases.json");
  size_t in_n = 0;
  char *in_data = read_all(in_path, &in_n);
  free(in_path);
  if (!in_data) return -1;
  Buf answers; buf_init(&answers); buf_reserve(&answers, in_n);
  buf_addc(&answers, '[');
  JCur j; j.p = in_data; j.end = in_data + in_n;
  j_ws(&j);
  int first = 1;
  if (j_expect(&j, '[')) {
    if (!j_expect(&j, ']')) {
      for (;;) {
        j_ws(&j);
        /* one case object: {"id": ..., "input": {...}} */
        if (!j_expect(&j, '{')) break;
        if (j_expect(&j, '}')) { /* empty case object: empty answer */ }
        else {
          for (;;) {
            j_ws(&j);
            Str k = j_string(&j);
            j_expect(&j, ':'); j_ws(&j);
            if (str_eq_c(k, "input")) {
              arena2_reset();
              CaseInput ci;
              parse_case_input(&j, &ci);
              if (!first) buf_addc(&answers, ',');
              first = 0;
              render_case_json(&ci, &answers);
            } else {
              j_skip_value(&j);
            }
            if (!j_expect(&j, ',')) break;
          }
          j_expect(&j, '}');
        }
        if (!j_expect(&j, ',')) break;
      }
      j_expect(&j, ']');
    }
  }
  buf_addc(&answers, ']');
  mkdir_p(output_dir);
  char *out_path = join_path(output_dir, "answers.json");
  int rc = write_all(out_path, answers.p, answers.n);
  free(out_path);
  buf_free(&answers);
  free(in_data);
  return rc;
}

/* ---------------- N-API glue ---------------- */
static napi_value napi_run_all(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value argv[2];
  napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  char in_dir[1024], out_dir[1024];
  size_t in_n = 0, out_n = 0;
  napi_get_value_string_utf8(env, argv[0], in_dir, sizeof in_dir, &in_n);
  napi_get_value_string_utf8(env, argv[1], out_dir, sizeof out_dir, &out_n);
  int rc = run_all(in_dir, out_dir);
  if (rc != 0) napi_throw_error(env, NULL, "runAll failed");
  napi_value undef;
  napi_get_undefined(env, &undef);
  return undef;
}
static napi_value napi_run_one(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  size_t len = 0;
  napi_get_value_string_utf8(env, argv[0], NULL, 0, &len);
  char *s = (char *)malloc(len + 1);
  napi_get_value_string_utf8(env, argv[0], s, len + 1, &len);
  arena_init();
  arena2_reset();
  JCur j; j.p = s; j.end = s + len;
  CaseInput ci;
  parse_case_input(&j, &ci);
  Buf out; buf_init(&out);
  render_case_json(&ci, &out);
  napi_value res;
  napi_create_string_utf8(env, out.p, out.n, &res);
  buf_free(&out);
  free(s);
  return res;
}
static napi_value Init(napi_env env, napi_value exports) {
  napi_property_descriptor desc[2];
  desc[0].utf8name = "runAll"; desc[0].name = NULL; desc[0].method = napi_run_all;
  desc[0].getter = NULL; desc[0].setter = NULL; desc[0].value = NULL;
  desc[0].attributes = napi_default; desc[0].data = NULL;
  desc[1].utf8name = "runOne"; desc[1].name = NULL; desc[1].method = napi_run_one;
  desc[1].getter = NULL; desc[1].setter = NULL; desc[1].value = NULL;
  desc[1].attributes = napi_default; desc[1].data = NULL;
  napi_define_properties(env, exports, 2, desc);
  return exports;
}
NAPI_MODULE(engine, Init)
