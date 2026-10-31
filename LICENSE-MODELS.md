# 3D 模型、视频与教程的许可

本仓代码使用 Apache-2.0（见 `LICENSE`）。下面这些例外：

## 3D 模型

`sim_model/structure/<日期>/microduck/assets/*.stl` 里，除 `battery_box.stl`、`battery_bracket.stl` 之外的网格来自
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)，版权归 **Pollen Robotics**，许可为
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。

我们换了电池（新增上面两个网格，来自我们自己的 SolidWorks 设计），并按 SolidWorks 重算了躯干的质量属性。
改编后的网格同样以 CC BY-NC-SA 4.0 授权，不得用于商业目的。

`sim_model/structure/<日期>/microduck/*.xml`（MJCF 机器人描述）来自同一上游，许可为 Apache-2.0。

## 舵机参数

`sim_model/servos/feetech/hd1910/params/hd1910_m6.json` 来自 [LuwuDynamics/xgoduck_rl](https://github.com/LuwuDynamics/xgoduck_rl)，
许可为 Apache-2.0，原文在同目录 `LICENSE-hd1910_m6`。

## 手册

手册站的文字与图片以 CC BY-NC-SA 4.0 授权。
