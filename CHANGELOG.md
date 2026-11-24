# 更新日志

版本号是发布日期加当天序号。每个版本在 [Releases](https://github.com/Xbotics-Embodied-AI-club/Xbotics-MiniDuck-Course/releases) 有同名发布，发布后不再改动。

## v20261124.01

当前版本。代码、仿真模型、固件和策略与 v20261120.01 相同。

- 手册新增「装配」：部件、装配顺序和八段装配动画；七个装配章节逐步列出做什么、要注意什么、什么时候接线、用几颗螺丝。
- 手册新增「课程」：课程表、01–03 讲的视频，以及第 02、03 讲的 notebook。
- README 加课程视频一节。

**你要做什么**

- 不用做什么。已经 clone 的同学 `git pull` 即可。

## v20261120.01

硬件版本 `2026-11-07-hd1910`（飞特 HD-1910 舵机，整机 792.7 g），固件 0.9.1，六个出厂策略：standup（起身）· velstand-flat（走路）· sitstand（坐站）· ground-pick（捡东西）· ball-kick（踢球）· roulade（翻滚）。

- 起身：舵机出力弱时也翻得过身。仿真里把舵机出力降到标称的七成，倒地后第一次就站起的比例为 98%。
- 走路：下坡稳。仿真里 10° 下坡摔倒的比例为 8%。
- 模型文件在 `sim_model/structure/2026-11-07/miniduck/`，授权说明见 `LICENSE-MODELS.md`。

**你要做什么**

- 之前 clone 过这个仓库的同学：把旧目录删掉重新 clone，再运行 `uv sync`。旧目录里自己导出过、想留的文件先移到别处。
