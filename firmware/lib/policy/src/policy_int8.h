// 策略的 int8 推理（全 int8 + 层间 ELU 查表）：原始观测 → 舵机动作。各个动作共用这一份代码。
//
// 固件与训练侧共用这一份：固件用它跑训练出来的策略；training/hostcheck/ 在主机上编译同一份代码，验证导出的 int8 参数与
// Python 整数参考逐位一致。
//
// 每个动作的参数由 training/export_policy.py 导出成 policies/<硬件版本>/<动作>/<名>_int8.h（定义一个
// `static const policy_int8_t <名>_policy`，主机自检直接编它），再由 firmware/tools/assets.py 转成同目录的
// <模型>.mdp；固件从板上文件系统读 .mdp，经 policy_int8_blob.h 解析成同一个结构，数值逐位相同。
// 本库只做计算：浮点归一化 → int8 输入量化 → 逐层 esp-nn 逐通道全连接，层间 ELU 查表。
//
// 输入输出的含义与该动作的 ONNX 的 obs / actions 完全相同（观测的顺序与单位见
// policies/<硬件版本>/<动作>/MANIFEST.json 的 observation_names）。
//
// 编译要求：
//   · 板上：esp-nn 的 S3 实现（本库 src/espnn/ 里带着，头文件 esp_nn_esp32s3.h）；
//     主机：上游 esp-nn 的 ANSI 实现（头文件 esp_nn.h）
//   · 主机自检：policies/ 在 include 路径里（按 <硬件版本>/<动作>/<名>_int8.h 引用）
//   · -ffp-contract=off：输入量化那两步浮点乘加不许合成一条指令，否则舍入与参考实现不同
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 中间层最宽多少通道（缓冲按它静态分配）。导出脚本按同一个数检查，超了就不导出。
#define POLICY_INT8_MAX_WIDTH 512
#define POLICY_INT8_MAX_LAYERS 8

typedef struct {
  const char *name;
  int obs_dim;
  int act_dim;
  int num_layers;
  // 每层：{补齐后的输入长度, 输出通道数, input_offset, output_offset}
  const int32_t (*layer_dims)[4];
  const int8_t *const *weights;
  const int32_t *const *bias;
  const int32_t *const *mult;
  const int32_t *const *shift;
  // 隐藏层的 ELU 查表（下标 q + 128）；最后一层为 NULL。
  const int8_t *const *lut;
  const float *obs_mean;
  const float *obs_inv_std;
  float in_inv_scale;
  float in_zero_plus_half;
  float out_scale;
  int out_zero;
  // 测试向量：原始观测与整数参考前向的 int8 输出。
  int num_tests;
  const float *test_obs;
  const int8_t *test_out_q;
} policy_int8_t;

// 跑一次策略，输出浮点动作（与 ONNX 的 actions 同义）。
void policy_int8_run(const policy_int8_t *p, const float *obs, float *act);

// 同上，但返回最后一层的 int8 原始输出 —— 自检按它逐位比对。
void policy_int8_run_q(const policy_int8_t *p, const float *obs, int8_t *out_q);

// 用策略自带的测试向量自检，返回不一致的输出个数（0 = 与 Python 参考逐位相同）；
// 参数本身不合本库的约束（层数、宽度）时返回 -1。
int policy_int8_selftest(const policy_int8_t *p);

// 把一份策略的推理参数整体拷进调用方给的一块连续内存需要多少字节。
// training/hostcheck 用它核对搬运本身；固件不用：.mdp 整份读进 PSRAM 后原地解析（policy_int8_blob.h）。
size_t policy_int8_arena_size(const policy_int8_t *src);

// 把 src 的推理参数拷进 arena（须 16 字节对齐、至少 policy_int8_arena_size 字节），
// 在 dst 里填一份指向 arena 的策略。名字与测试向量不拷，仍指向 src 的。
// 成功返回 0；arena 未对齐、不够大，或层数超出本库上限时返回 -1。
int policy_int8_copy_to(const policy_int8_t *src, void *arena, size_t arena_size,
                        policy_int8_t *dst);

#ifdef __cplusplus
}
#endif
