# 飞特 HD-1910：仿真参数的来源

- `params/hd1910_m6.json` 原样取自 [LuwuDynamics/xgoduck_rl](https://github.com/LuwuDynamics/xgoduck_rl) 提交
  `326d77a1122870bdefa2c36403937502c958e69c` 的 `src/mjlab_microduck/robot/xgoduck/params/1910_m6.json`，
  未改动；SHA-256 `ef2d51adfb1cc0831b9b02ca19aa9176fceecdf725148d084378d7d9a64afceb`。
- 许可证 Apache-2.0，原文见 `params/LICENSE-hd1910_m6`。
- 上游把舵机叫 HLS1910；JSON 里的 `"actuator": "sts3215"` 是选 BAM 里飞特控制器的实现，
  不是换成 STS3215 的出厂参数。这份参数不是我们自己辨识的。
- 同一份参数也被 fanhao375/microduck-replica、zhuoligetu123/microduck-hd1910m 使用；
  BAM 的飞特控制器在 mjlab 下少初始化一个状态量，修法照 microduck-replica（见 `training/servos/feetech.py`）。
- `kp_fw=5` 与公开训练基线一致；固件把舵机运行 P 写成同一个值、D 写 0（仿真模型里没有 D）。
