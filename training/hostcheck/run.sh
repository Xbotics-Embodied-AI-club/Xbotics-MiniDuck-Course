#!/usr/bin/env bash
# 主机上编译并运行某个动作的 int8 自检：firmware/lib/policy 的 C 代码（与固件同一份） + esp-nn 的 ANSI 内核 +
# <头文件目录>/<名>_int8.h 里的测试向量。与 Python 整数参考逐位一致才退 0。
#
# esp-nn 钉在与固件相同的 commit；源码拉到 build/（gitignore），不进仓。
# 用法（仓根目录）：bash training/hostcheck/run.sh <头文件目录> <名>
#   例：bash training/hostcheck/run.sh policies/2026-10-24-hd1910/standup standup
set -euo pipefail

HEADER_DIR="$(cd "$1" && pwd)"
NAME="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/training/hostcheck/build"
ESP_NN_COMMIT=d116020
mkdir -p "$BUILD"

if [ ! -f "$HEADER_DIR/${NAME}_int8.h" ]; then
  echo "没有 $HEADER_DIR/${NAME}_int8.h" >&2
  exit 2
fi
if [ ! -d "$BUILD/esp-nn/.git" ]; then
  git clone -q https://github.com/espressif/esp-nn.git "$BUILD/esp-nn"
fi
git -C "$BUILD/esp-nn" checkout -q "$ESP_NN_COMMIT"

INC=(-I "$BUILD/esp-nn/include" -I "$BUILD/esp-nn/src/common"
     -I "$ROOT/firmware/lib/policy/src" -I "$HEADER_DIR")
# esp-nn 是上游代码，它的公共头里有未使用的 static 函数 —— 告警归它，不拿 -Werror 卡它。
gcc -std=gnu11 -O2 -w -c "${INC[@]}" \
  "$BUILD/esp-nn/src/fully_connected/esp_nn_fully_connected_ansi.c" -o "$BUILD/esp_nn_fc.o"
# 我们自己的代码：告警即错误；浮点不许合成乘加（见 policy_int8.h）。
# 每次编到一个独立的输出名，并发跑几个动作的自检时互不覆盖。
OUT="$(mktemp -p "$BUILD" "${NAME}_selftest.XXXXXX")"
trap 'rm -f "$OUT"' EXIT
gcc -std=gnu11 -O2 -Wall -Werror -ffp-contract=off "${INC[@]}" \
  -DPOLICY_HEADER="\"${NAME}_int8.h\"" -DPOLICY="${NAME}_policy" \
  "$ROOT/training/hostcheck/main.c" "$ROOT/firmware/lib/policy/src/policy_int8.c" \
  "$BUILD/esp_nn_fc.o" -lm -o "$OUT"

"$OUT"
