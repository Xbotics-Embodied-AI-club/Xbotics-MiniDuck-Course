// 在主机上跑某个动作的 int8 策略自检：固件里那份 C 推理代码（firmware/lib/policy）
// + esp-nn 的 ANSI 内核，输出必须与 Python 整数参考逐位相同。退出码 0 才算过。
//
// 两遍：一遍直接用头文件里的参数；一遍先 policy_int8_copy_to 拷进一块 16 字节对齐的内存
// 再跑 —— 板上切策略时就是这样把 flash 里的权重搬进 SRAM 的，搬运本身也要核对：
// 拷出来的那份逐位一致，且它的每个参数指针都落在新内存里（没有偷偷还指着原件）。
//
// 编译时给两个宏（run.sh 按动作名拼好）：
//   POLICY_HEADER  生成的头文件名，如 "standup_int8.h"
//   POLICY         头文件里定义的策略实例，如 standup_policy
#include <stdio.h>
#include <stdlib.h>

#include "policy_int8.h"
#include POLICY_HEADER

static int outside(const void *p, const void *base, size_t size) {
  return (const char *)p < (const char *)base || (const char *)p >= (const char *)base + size;
}

// 拷贝里有几个参数指针不在 arena 里（测试向量按约定仍指着原件，不算）。
static int pointers_outside(const policy_int8_t *c, const void *arena, size_t size) {
  int n = outside(c->layer_dims, arena, size) + outside(c->obs_mean, arena, size) +
          outside(c->obs_inv_std, arena, size);
  for (int l = 0; l < c->num_layers; ++l) {
    n += outside(c->weights[l], arena, size) + outside(c->bias[l], arena, size) +
         outside(c->mult[l], arena, size) + outside(c->shift[l], arena, size);
    if (c->lut[l] != 0) n += outside(c->lut[l], arena, size);
  }
  return n;
}

int main(void) {
  int direct = policy_int8_selftest(&POLICY);

  size_t size = policy_int8_arena_size(&POLICY);
  void *arena = aligned_alloc(16, (size + 15) / 16 * 16);
  policy_int8_t copy;
  int copied = arena != NULL ? policy_int8_copy_to(&POLICY, arena, size, &copy) : -1;
  int via_copy = copied == 0 ? policy_int8_selftest(&copy) : -1;
  int stray = copied == 0 ? pointers_outside(&copy, arena, size) : -1;
  printf("%s: policy_int8_selftest %d mismatches over %d test vectors; "
         "copy (%zu bytes): %d mismatches, %d pointers outside\n",
         POLICY.name, direct, POLICY.num_tests, size, via_copy, stray);
  free(arena);
  return (direct == 0 && via_copy == 0 && stray == 0) ? 0 : 1;
}
