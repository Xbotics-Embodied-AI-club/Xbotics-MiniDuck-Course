# 更新日志

版本号是发布日期加当天序号。每个版本在 [Releases](https://github.com/Xbotics-Embodied-AI-club/Xbotics-MiniDuck-Course/releases) 有同名发布，发布后不再改动。

## v20261031.01

初始版本。

| 内容 | 版本 |
| --- | --- |
| 硬件版本 | `2026-10-24-hd1910`（飞特 HD-1910，V0.2 头板 + 身体板） |
| 固件 | 0.8.1 |
| 出厂策略 | standup · velstand-flat · sitstand |

**你要做什么**

- 鸭子：固件是 0.7.6 到 0.8.0 时，按手册「固件与策略更新」只烧 `0x10000_firmware.bin`；更早的固件六个文件全烧；已经是 0.8.1 的不用动。
- 电脑：第一次 `git clone` 后运行 `uv sync`。
