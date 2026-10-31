// 策略 int8 推理的实现。逐步的数学与 training/int8.py 的 Int8Policy 一一对应。
#include "policy_int8.h"

#include <math.h>
#include <string.h>

#if defined(ARDUINO) || defined(ESP_PLATFORM)
// 板上：本库 src/espnn/ 里带的 esp-nn，这个头把通用函数名映射到 S3 SIMD 实现。
#include <esp_nn_esp32s3.h>
#else
// 主机：上游 esp-nn，没定义 CONFIG_NN_OPTIMIZED 时是 ANSI 实现（与 S3 实现逐位一致）。
#include <esp_nn.h>
#endif

// 两块轮换的中间缓冲，宽度取最宽的一层。16 字节对齐：esp-nn S3 快路径要求输入或权重之一对齐。
// 静态缓冲 ⇒ 同一时刻只能跑一份策略（控制环是单线程的）。
static int8_t buf_a[POLICY_INT8_MAX_WIDTH] __attribute__((aligned(16)));
static int8_t buf_b[POLICY_INT8_MAX_WIDTH] __attribute__((aligned(16)));

// 原始观测 → 第一层的 int8 输入（补零到对齐长度）。全程 float32，顺序与 Python 参考相同。
static void quantize_input(const policy_int8_t *p, const float *obs, int8_t *q) {
  memset(q, 0, (size_t)p->layer_dims[0][0]);
  for (int i = 0; i < p->obs_dim; ++i) {
    float x = (obs[i] - p->obs_mean[i]) * p->obs_inv_std[i];
    float t = x * p->in_inv_scale + p->in_zero_plus_half;
    float f = floorf(t);
    q[i] = (int8_t)(f < -128.0f ? -128 : (f > 127.0f ? 127 : (int)f));
  }
}

void policy_int8_run_q(const policy_int8_t *p, const float *obs, int8_t *out_q) {
  int8_t *in = buf_a;
  int8_t *out = buf_b;
  quantize_input(p, obs, in);
  for (int l = 0; l < p->num_layers; ++l) {
    const int32_t *d = p->layer_dims[l];
    esp_nn_fully_connected_per_ch_s8(in, d[2], (uint16_t)d[0], p->weights[l], 0, p->bias[l], out,
                                     (uint16_t)d[1], d[3], p->shift[l], p->mult[l], -128, 127);
    if (p->lut[l] != 0) {
      for (int c = 0; c < d[1]; ++c) out[c] = p->lut[l][(int)out[c] + 128];
    }
    int8_t *t = in;
    in = out;
    out = t;
  }
  memcpy(out_q, in, (size_t)p->act_dim);
}

void policy_int8_run(const policy_int8_t *p, const float *obs, float *act) {
  int8_t q[POLICY_INT8_MAX_WIDTH];
  policy_int8_run_q(p, obs, q);
  for (int i = 0; i < p->act_dim; ++i) act[i] = p->out_scale * (float)((int)q[i] - p->out_zero);
}

int policy_int8_selftest(const policy_int8_t *p) {
  if (p->num_layers < 1 || p->num_layers > POLICY_INT8_MAX_LAYERS) return -1;
  for (int l = 0; l < p->num_layers; ++l) {
    if (p->layer_dims[l][0] > POLICY_INT8_MAX_WIDTH || p->layer_dims[l][1] > POLICY_INT8_MAX_WIDTH) {
      return -1;
    }
  }
  int mismatches = 0;
  int8_t q[POLICY_INT8_MAX_WIDTH];
  for (int k = 0; k < p->num_tests; ++k) {
    policy_int8_run_q(p, &p->test_obs[k * p->obs_dim], q);
    for (int i = 0; i < p->act_dim; ++i) mismatches += (q[i] != p->test_out_q[k * p->act_dim + i]);
  }
  return mismatches;
}

// arena 里每一块都按 16 字节对齐：权重是 esp-nn S3 快路径的输入之一，要求对齐。
static size_t align16(size_t n) { return (n + 15u) & ~(size_t)15u; }

static void *take(uint8_t **cursor, size_t bytes) {
  void *out = *cursor;
  *cursor += align16(bytes);
  return out;
}

size_t policy_int8_arena_size(const policy_int8_t *src) {
  size_t L = (size_t)src->num_layers;
  size_t n = align16(sizeof(int32_t) * 4 * L)      // layer_dims
             + 5 * align16(sizeof(void *) * L)     // weights / bias / mult / shift / lut 指针表
             + 2 * align16(sizeof(float) * (size_t)src->obs_dim);
  for (size_t l = 0; l < L; ++l) {
    const int32_t *d = src->layer_dims[l];
    n += align16((size_t)d[0] * (size_t)d[1]) + 3 * align16(sizeof(int32_t) * (size_t)d[1]);
    if (src->lut[l] != 0) n += align16(256);
  }
  return n;
}

int policy_int8_copy_to(const policy_int8_t *src, void *arena, size_t arena_size,
                        policy_int8_t *dst) {
  if (src->num_layers < 1 || src->num_layers > POLICY_INT8_MAX_LAYERS) return -1;
  if (((uintptr_t)arena & 15u) != 0 || arena_size < policy_int8_arena_size(src)) return -1;
  int L = src->num_layers;
  uint8_t *cursor = (uint8_t *)arena;

  int32_t(*dims)[4] = take(&cursor, sizeof(int32_t) * 4 * (size_t)L);
  memcpy(dims, src->layer_dims, sizeof(int32_t) * 4 * (size_t)L);
  const int8_t **weights = take(&cursor, sizeof(void *) * (size_t)L);
  const int32_t **bias = take(&cursor, sizeof(void *) * (size_t)L);
  const int32_t **mult = take(&cursor, sizeof(void *) * (size_t)L);
  const int32_t **shift = take(&cursor, sizeof(void *) * (size_t)L);
  const int8_t **lut = take(&cursor, sizeof(void *) * (size_t)L);
  float *mean = take(&cursor, sizeof(float) * (size_t)src->obs_dim);
  memcpy(mean, src->obs_mean, sizeof(float) * (size_t)src->obs_dim);
  float *inv_std = take(&cursor, sizeof(float) * (size_t)src->obs_dim);
  memcpy(inv_std, src->obs_inv_std, sizeof(float) * (size_t)src->obs_dim);

  for (int l = 0; l < L; ++l) {
    size_t out = (size_t)dims[l][1];
    size_t w_bytes = (size_t)dims[l][0] * out;
    int8_t *w = take(&cursor, w_bytes);
    memcpy(w, src->weights[l], w_bytes);
    weights[l] = w;
    int32_t *b = take(&cursor, sizeof(int32_t) * out);
    memcpy(b, src->bias[l], sizeof(int32_t) * out);
    bias[l] = b;
    int32_t *m = take(&cursor, sizeof(int32_t) * out);
    memcpy(m, src->mult[l], sizeof(int32_t) * out);
    mult[l] = m;
    int32_t *s = take(&cursor, sizeof(int32_t) * out);
    memcpy(s, src->shift[l], sizeof(int32_t) * out);
    shift[l] = s;
    lut[l] = 0;
    if (src->lut[l] != 0) {
      int8_t *t = take(&cursor, 256);
      memcpy(t, src->lut[l], 256);
      lut[l] = t;
    }
  }

  *dst = *src;
  dst->layer_dims = (const int32_t(*)[4])dims;
  dst->weights = weights;
  dst->bias = bias;
  dst->mult = mult;
  dst->shift = shift;
  dst->lut = lut;
  dst->obs_mean = mean;
  dst->obs_inv_std = inv_std;
  return 0;
}
