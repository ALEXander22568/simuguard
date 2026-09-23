# 环境搭建：在 RoboTwin 上跑 SimuGuard + LingBot-VA

仓库里只有 SimuGuard 本身（核心库、RoboTwin 适配器、脚本、`compat/` 里的一个导入垫片）。评测还需要四样**仓库外**的东西。node1 / node2 上都已经装好，直接用；换机器要按下面重建。

## 一、工作区布局

```
$WS/                                  # SIMUGUARD_WORKSPACE
├── SimuGuard/                        # 本仓库（git clone）
├── RoboTwin/                         # RoboTwin 2.0 官方仓库，未修改（含 assets/、XPolicyLab/）
├── .venv-lingbot/                    # 覆盖环境：LingBot 的 .venv-server 站点包 + h5py（跑策略服务器）
├── compat/flash_attn/                # 可选，仓库 compat/ 里有同一份
└── runs/                             # 所有评测输出

$RUNTIME/                             # LINGBOT_RUNTIME，LingBot-VA 的隔离复现目录
├── .venv-server/                     # 策略服务器环境（torch + lingbot-va）
├── .venv-client/                     # 仿真客户端环境（torch 2.4.1+cu121, sapien 3, curobo 0.7.8, pytorch3d）
├── checkpoints/lingbot-va-posttrain-robotwin-modelscope/   # 45 GB 权重
├── lingbot-va/                       # LingBot-VA 源码 7c6ffa9
├── install_server_env.sh / install_client_env.sh / install_curobo.sh / install_pytorch3d.sh
└── README_REPRO.md                   # 复现说明（三处配置改动，其余与上游一致）
```

现有位置：

| | node1 | node2 |
|---|---|---|
| `$WS` | `/mnt/nvme0/shared/zhoujingjing/simuguard` | `/mnt/nvme0/twinguar/simuguard` |
| `$RUNTIME` | `/home/zhoujingjing/lingbot-va-repro` | `/mnt/nvme0/twinguar/RoboTwin-2.0/runtime/lingbot-va-repro` |
| ffmpeg（`TOOLS_BIN`） | `$WS/.tools/bin` | `/mnt/nvme0/twinguar/RoboTwin-2.0/.tools/bin` |

node1 2026-09-22 起 GPU0/6/7 处于 "requires reset"，整机 CUDA 初始化失败，修好前用 node2。

## 二、在现有节点上启动（三步）

```bash
WS=/mnt/nvme0/twinguar/simuguard
RUNTIME=/mnt/nvme0/twinguar/RoboTwin-2.0/runtime/lingbot-va-repro

# 1. 拿最新代码（节点无法直接访问 GitHub，从本机 rsync，或用带 token 的 clone）
rsync -az --exclude .git --exclude runs ~/simuguard/ node2:$WS/SimuGuard/

# 2. 登记要跑的任务（docs/robotwin_remaining_tasks.md 有属性名）
#    simuguard/adapters/robotwin/tasks.py -> TASK_SPECS

# 3. 启动一批任务，每个 50 局
OUT=$WS/runs/tier1_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; cd $WS/SimuGuard
MODEL_FREE_MIB=18800 SIMUGUARD_WORKSPACE=$WS ROBOTWIN_ROOT=$WS/RoboTwin LINGBOT_RUNTIME=$RUNTIME \
LINGBOT_MODEL=$RUNTIME/checkpoints/lingbot-va-posttrain-robotwin-modelscope \
TOOLS_BIN=/mnt/nvme0/twinguar/RoboTwin-2.0/.tools/bin \
SIMUGUARD_CONFIG=$WS/SimuGuard/configs/scale_no_bundles.json \
MANIFEST_PYTHON=$RUNTIME/.venv-client/bin/python \
nohup bash scripts/robotwin/cross_task_campaign.sh $OUT 50 auto auto lift_pot move_can_pot > $OUT/tmux.out 2>&1 &
```

`cross_task_campaign.sh` 每个任务会：等到一张空闲 ≥ `MODEL_FREE_MIB` 的卡（模型服务器约 18 GB）和一张 ≥ 8 GB 的卡；起 LingBot 服务器、XPolicyLab 桥、带 SimuGuard 的官方评测器；结束后写 `manifest.json` / `SEEDS.md` / `videos/`，回放段在 `simuguard/segments/`。进度看 `$OUT/STATUS`。

## 三、换一台机器要重建什么

按顺序，都是一次性的：

1. **RoboTwin 2.0**：`git clone` 官方仓库（我们用的 commit `6dde571`，未做任何修改），按其 README 装 assets（`assets/_download.py`）和 XPolicyLab。
2. **LingBot-VA 运行时**：把 node2 的 `$RUNTIME` 整个目录拷过去最省事（含 45 GB 权重）；否则按 `README_REPRO.md` 依次跑 `install_server_env.sh`、`install_client_env.sh`、`install_curobo.sh`、`install_pytorch3d.sh`（需要 CUDA 12.1 工具链，curobo 与 pytorch3d 从源码编译，各约 30–60 分钟），权重用 `lingbot_download.py` 从 ModelScope 拉。
3. **`.venv-lingbot` 覆盖环境**：`python3.10 -m venv $WS/.venv-lingbot`，然后在 `site-packages` 里放一个 `.pth` 指向 `$RUNTIME/.venv-server/lib/python3.10/site-packages`，再 `pip install h5py`。作用是让服务器进程能读 RoboTwin 的 hdf5 统计而不改动上游环境。
4. **ffmpeg** 静态二进制放到任意目录，`TOOLS_BIN` 指过去（评测器录视频用）。
5. **SimuGuard**：`git clone` 本仓库；`.venv-client` 里 `pip install -e $WS/SimuGuard` 或直接靠脚本设置的 `PYTHONPATH`（脚本已经处理，不装也行）。

自检：

```bash
$RUNTIME/.venv-client/bin/python -c "import torch, sapien, curobo; print(torch.cuda.is_available())"
cd $WS/SimuGuard && $RUNTIME/.venv-client/bin/python -m pytest tests -q
```

## 四、脚本用到的环境变量

| 变量 | 含义 | 默认 |
|---|---|---|
| `SIMUGUARD_WORKSPACE` | 工作区 `$WS` | `/mnt/nvme0/twinguar/simuguard` |
| `ROBOTWIN_ROOT` | RoboTwin 仓库 | `$WS/RoboTwin` |
| `LINGBOT_RUNTIME` | LingBot 复现目录 `$RUNTIME` | node2 路径 |
| `LINGBOT_MODEL` | 权重目录 | — |
| `TOOLS_BIN` | ffmpeg 所在目录 | node2 路径 |
| `SIMUGUARD_CONFIG` | 检测器/记录配置 | `configs/scale_no_bundles.json` |
| `MANIFEST_PYTHON` | 写 manifest 用的解释器 | `python3` |
| `MODEL_FREE_MIB` / `SIM_FREE_MIB` | 自动选卡的显存门槛 | 19500 / 8000 |
| `SIMUGUARD_COMPAT` | flash_attn 垫片目录 | `$WS/compat`，没有则用仓库 `compat/` |
