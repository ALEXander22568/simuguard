# RoboTwin 50 任务统一结果表

LingBot-VA（RoboTwin 后训练权重，末端位姿动作）在 RoboTwin 2.0 全部 50 个任务上的结果。主口径是两列分数：纯 RoboTwin 的官方成功率，和剔除含物理无效事件的局之后的 RoboTwin+SimuGuard 成功率；审计列沿用论文公式 (1)，目前只有 place_can_basket 做过闭环接管。表由 `scripts/robotwin/results_table.py` 从各运行目录的 manifest.json、`gravity_filter.py` 的 events_gravity.json 和接管 / 归因报告直接生成，不手填数字。

| 列 | 含义 |
|---|---|
| N / S | benchmark 打分的局数 / 官方成功数 |
| RoboTwin | 官方成功率 S/N，即纯 RoboTwin 的分数 |
| 剔除 | 含物理无效事件的局数，成功、失败都算（= E_fail + E_succ） |
| RoboTwin+SimuGuard | 剔除这些局之后的成功率 (S − E_succ)/(N − 剔除)，括号内为剔除后的成功数/局数 |
| 跳过 | 评测器在凑满 N 局之前丢弃的种子数，按打分种子范围推算，包含规划前就被判为不稳定、没有 SimuGuard 录制的种子 |
| E_fail / E_succ | 失败局 / 成功局中含物理无效事件的局数（检测器确认 + 重力过滤：峰值速度超过同高度自由落体速度 1.5 倍） |
| R1 / R0 | E_fail 中，事件前 1 s 让策略重新接管后成功的局数：R1 去掉注入的能量（2 m/s 限速），R0 物理不变（对照组） |
| 审计 | (S + R1 − R0)/N；E_fail = 0 时等于官方；接管未覆盖全部 E_fail 时为 - |
| 上界 | (S + E_fail)/N，把每个含事件的失败都算成成功 |
| 开环 | E_fail 中，用原动作回放并限速 2 m/s 后成功的局数（`attribute_failures.py`，诊断用，不计分） |

| task | N | S | RoboTwin % | 剔除 | RoboTwin+SimuGuard % | 跳过 | 失败 | E_fail | E_succ | R1 | R0 | 审计 % | 上界 % | 开环 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `adjust_bottle` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 11 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `beat_block_hammer` | 20 | 19 | 95.0 | 0 | 95.0 (19/20) | 11 | 1 | 0 | 0 | - | - | 95.0 | 95.0 | - |
| `blocks_ranking_rgb` | 20 | 19 | 95.0 | 0 | 95.0 (19/20) | 0 | 1 | 0 | 0 | - | - | 95.0 | 95.0 | - |
| `blocks_ranking_size` | 20 | 17 | 85.0 | 0 | 85.0 (17/20) | 2 | 3 | 0 | 0 | - | - | 85.0 | 85.0 | - |
| `click_alarmclock` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 4 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `click_bell` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 0 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `dump_bin_bigbin` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 28 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `grab_roller` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 1 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `handover_block` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 6 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `handover_mic` | 20 | 19 | 95.0 | 0 | 95.0 (19/20) | 3 | 1 | 0 | 0 | - | - | 95.0 | 95.0 | - |
| `hanging_mug` | 20 | 3 | 15.0 | 1 | 15.8 (3/19) | 2 | 17 | 1 | 0 | - | - | - | 20.0 | - |
| `lift_pot` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 28 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `move_can_pot` | 20 | 18 | 90.0 | 0 | 90.0 (18/20) | 3 | 2 | 0 | 0 | - | - | 90.0 | 90.0 | - |
| `move_pillbottle_pad` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 11 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `move_playingcard_away` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 0 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `move_stapler_pad` | 20 | 14 | 70.0 | 1 | 73.7 (14/19) | 0 | 6 | 1 | 0 | - | - | - | 75.0 | 0 |
| `open_laptop` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 0 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `open_microwave` | 20 | 15 | 75.0 | 0 | 75.0 (15/20) | 15 | 5 | 0 | 0 | - | - | 75.0 | 75.0 | - |
| `pick_diverse_bottles` | 20 | 18 | 90.0 | 0 | 90.0 (18/20) | 39 | 2 | 0 | 0 | - | - | 90.0 | 90.0 | - |
| `pick_dual_bottles` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 1 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `place_a2b_left` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 5 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `place_a2b_right` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 6 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `place_bread_basket` | 20 | 19 | 95.0 | 1 | 100.0 (19/19) | 4 | 1 | 1 | 0 | - | - | - | 100.0 | - |
| `place_bread_skillet` | 20 | 19 | 95.0 | 0 | 95.0 (19/20) | 23 | 1 | 0 | 0 | - | - | 95.0 | 95.0 | - |
| `place_burger_fries` | 20 | 20 | 100.0 | 2 | 100.0 (18/18) | 0 | 0 | 0 | 2 | - | - | 100.0 | 100.0 | - |
| `place_can_basket` | 90 | 67 | 74.4 | 37 | 96.2 (51/53) | 22 | 23 | 21 | 16 | 15 | 10 | 80.0 | 97.8 | 14 |
| `place_cans_plasticbox` | 10 | 10 | 100.0 | 0 | 100.0 (10/10) | 0 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `place_container_plate` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 4 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `place_dual_shoes` | 20 | 20 | 100.0 | 1 | 100.0 (19/19) | 12 | 0 | 0 | 1 | - | - | 100.0 | 100.0 | - |
| `place_empty_cup` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 2 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `place_fan` | 20 | 17 | 85.0 | 1 | 89.5 (17/19) | 4 | 3 | 1 | 0 | - | - | - | 90.0 | 0 |
| `place_mouse_pad` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 0 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `place_object_basket` | 20 | 18 | 90.0 | 2 | 94.4 (17/18) | 8 | 2 | 1 | 1 | - | - | - | 95.0 | - |
| `place_object_scale` | 20 | 19 | 95.0 | 0 | 95.0 (19/20) | 7 | 1 | 0 | 0 | - | - | 95.0 | 95.0 | - |
| `place_object_stand` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 0 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `place_phone_stand` | 20 | 19 | 95.0 | 0 | 95.0 (19/20) | 10 | 1 | 0 | 0 | - | - | 95.0 | 95.0 | - |
| `place_shoe` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 3 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `press_stapler` | 20 | 18 | 90.0 | 0 | 90.0 (18/20) | 2 | 2 | 0 | 0 | - | - | 90.0 | 90.0 | - |
| `put_bottles_dustbin` | 10 | 9 | 90.0 | 0 | 90.0 (9/10) | 4 | 1 | 0 | 0 | - | - | 90.0 | 90.0 | - |
| `put_object_cabinet` | 10 | 7 | 70.0 | 0 | 70.0 (7/10) | 73 | 3 | 0 | 0 | - | - | 70.0 | 70.0 | - |
| `rotate_qrcode` | 20 | 19 | 95.0 | 0 | 95.0 (19/20) | 2 | 1 | 0 | 0 | - | - | 95.0 | 95.0 | - |
| `scan_object` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 44 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `shake_bottle` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 16 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `shake_bottle_horizontally` | 20 | 20 | 100.0 | 0 | 100.0 (20/20) | 16 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `stack_blocks_three` | 20 | 19 | 95.0 | 0 | 95.0 (19/20) | 2 | 1 | 0 | 0 | - | - | 95.0 | 95.0 | - |
| `stack_blocks_two` | 10 | 10 | 100.0 | 0 | 100.0 (10/10) | 0 | 0 | 0 | 0 | - | - | 100.0 | 100.0 | - |
| `stack_bowls_three` | 50 | 38 | 76.0 | 12 | 100.0 (38/38) | 43 | 12 | 12 | 0 | - | - | - | 100.0 | 0 |
| `stack_bowls_two` | 20 | 19 | 95.0 | 1 | 100.0 (19/19) | 7 | 1 | 1 | 0 | - | - | - | 100.0 | - |
| `stamp_seal` | 20 | 19 | 95.0 | 1 | 100.0 (19/19) | 9 | 1 | 1 | 0 | - | - | - | 100.0 | 1 |
| `turn_switch` | 20 | 9 | 45.0 | 0 | 45.0 (9/20) | 3 | 11 | 0 | 0 | - | - | 45.0 | 45.0 | - |
| **合计** | 1060 | 957 | 90.3 | 60 | 93.7 (937/1000) | 496 | 103 | 40 | 20 |  |  |  | 94.1 |  |

## 运行目录

- `4090-node1:/mnt/nvme0/shared/zhoujingjing/simuguard/runs/campaign_20260918T220055Z/default/`：rep1, rep2, rep3
- `4090-node1:/mnt/nvme0/shared/zhoujingjing/simuguard/runs/crosstask_20260921T160736Z/`：put_bottles_dustbin, stack_blocks_two
- `4090-node2:/mnt/nvme0/twinguar/simuguard/runs/crosstask3_20260922T112052Z/`：place_cans_plasticbox, put_object_cabinet
- `4090-node2:/mnt/nvme0/twinguar/simuguard/runs/crosstask4_50seeds/`：stack_bowls_three
- `4090-node2:/mnt/nvme0/twinguar/simuguard/runs/crosstask5_20ep/`：place_object_basket
- `4090-node2:/mnt/nvme0/twinguar/simuguard/runs/crosstask6_remote20/`：dump_bin_bigbin, hanging_mug, lift_pot, move_can_pot, place_bread_basket, place_bread_skillet, stack_bowls_two
- `4090-node2:/mnt/nvme0/twinguar/simuguard/runs/crosstask7_all36/`：blocks_ranking_size, click_alarmclock, grab_roller, handover_block, handover_mic, move_pillbottle_pad, pick_diverse_bottles, pick_dual_bottles, place_a2b_left, place_a2b_right, place_burger_fries, place_container_plate, place_dual_shoes, place_empty_cup, place_fan, place_mouse_pad, place_object_scale, place_object_stand, place_phone_stand, place_shoe, stack_blocks_three
- `4090-node3:/home/zhoujingjing/simuguard/runs/crosstask7_all36/`：adjust_bottle, beat_block_hammer, blocks_ranking_rgb, click_bell, move_playingcard_away, move_stapler_pad, open_laptop, open_microwave, press_stapler, rotate_qrcode, scan_object, shake_bottle, shake_bottle_horizontally, stamp_seal, turn_switch

## 重新生成

```bash
python scripts/robotwin/gravity_filter.py --runs <task_dir> ...
python scripts/robotwin/results_table.py --runs <task_dir> ... \
    --takeover <rollback_retry 输出, vclamp_2.0> --control <rollback_retry 输出, baseline> \
    --attr <attribute_failures 输出> --out docs/robotwin_results_table.csv
```

只需要 manifest.json、eval_command.txt、events_gravity.json 和报告 JSON，可以先把这些小文件从各主机拷到一处再生成。
