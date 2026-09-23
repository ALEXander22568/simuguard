# RoboTwin 50 任务统一结果表

统计口径（所有任务、所有人相同）：

| 列 | 含义 |
|---|---|
| N | benchmark 打分的局数（被 expert 检查跳过的 seed 不在内） |
| S | 官方成功数；**official = S/N** 就是 benchmark 自己报的成功率 |
| E_fail / E_succ | 失败局 / 成功局中含物理无效事件的局数（`gravity_filter.py` 过滤之后，掉落、倒出不算） |
| F+ | 失败局中，去掉伪影（限速反事实）后变成功的局数，判定 `environment_caused` |
| F± | 失败局中，只有质量反事实才翻转的局数，判定 `environment_possible`，只给上界 |
| S- | 成功局中，去掉伪影后变失败的局数，判定 `artifact_assisted_success` |
| **audited** | **(S - S- + F+) / N**，SimuGuard 修正后的成功率。分母不变：被环境弄坏的局按策略本来会得的结果计，不剔除 |
| upper | (S - S- + F+ + F±) / N，把所有 environment_possible 也算成环境导致的上界 |
| attr | 是否跑过反事实归因；`no` 时 F+/F±/S- 一律记 0，audited = official |

注意：不是「成功数 / (总数 - 仿真问题数)」。剔除会让分母随伪影数变化、把无事件失败局的权重抬高（罐子任务剔除法会得到 98.5%，修正法是 86.7%）。

一行怎么产生：跑完 campaign，然后 `gravity_filter.py --runs <task_dir>`，再 `attribute_failures.py --select failures --phases policy`（成功局要算 S- 时再加 `--select successes-with-events`），最后 `results_table.py --runs <task_dir> --attr <attr_dir> --out docs/robotwin_results_table.csv`。脚本会打印 markdown 行并更新 CSV；把 CSV 和这个 md 一起提交。

| task | 档 | N | S | official | E_fail | E_succ | F+ | F± | S- | audited | upper | attr | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adjust_bottle | 3 |  |  |  |  |  |  |  |  |  |  |  |  |
| beat_block_hammer | 4 |  |  |  |  |  |  |  |  |  |  |  |  |
| blocks_ranking_rgb | 3 |  |  |  |  |  |  |  |  |  |  |  |  |
| blocks_ranking_size | 3 |  |  |  |  |  |  |  |  |  |  |  |  |
| click_alarmclock | 4 |  |  |  |  |  |  |  |  |  |  |  |  |
| click_bell | 4 |  |  |  |  |  |  |  |  |  |  |  |  |
| dump_bin_bigbin | done | 10 | 8 | 8/10 | 1 | 0 | 0 | 0 | 0 | 8/10 | 8/10 | yes | 失败局 100010 伪影明确但两条反事实均未翻转（unresolved） |
| grab_roller | 3 |  |  |  |  |  |  |  |  |  |  |  |  |
| handover_block | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| handover_mic | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| hanging_mug | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| lift_pot | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| move_can_pot | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| move_pillbottle_pad | 3 |  |  |  |  |  |  |  |  |  |  |  |  |
| move_playingcard_away | 3 |  |  |  |  |  |  |  |  |  |  |  |  |
| move_stapler_pad | 3 |  |  |  |  |  |  |  |  |  |  |  |  |
| open_laptop | 3 |  |  |  |  |  |  |  |  |  |  |  |  |
| open_microwave | 3 |  |  |  |  |  |  |  |  |  |  |  |  |
| pick_diverse_bottles | 3 |  |  |  |  |  |  |  |  |  |  |  |  |
| pick_dual_bottles | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| place_a2b_left | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| place_a2b_right | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| place_bread_basket | done | 10 | 9 | 9/10 | 1 | 0 | 0 | 1 | 0 | 9/10 | 10/10 | yes |  |
| place_bread_skillet | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| place_burger_fries | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| place_can_basket | done | 90 | 67 | 67/90 | 21 | 16 | 14 | 7 | 3 | 78/90 | 85/90 | yes | 30 seed x 3 配置；主实验 |
| place_cans_plasticbox | done | 10 | 10 | 10/10 | 0 | 0 | 0 | 0 | 0 | 10/10 | 10/10 | no |  |
| place_container_plate | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| place_dual_shoes | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| place_empty_cup | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| place_fan | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| place_mouse_pad | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| place_object_basket | done | 10 | 10 | 10/10 | 0 | 1 | 0 | 0 | 0 | 10/10 | 10/10 | yes |  |
| place_object_scale | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| place_object_stand | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| place_phone_stand | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| place_shoe | 2 |  |  |  |  |  |  |  |  |  |  |  |  |
| press_stapler | 4 |  |  |  |  |  |  |  |  |  |  |  |  |
| put_bottles_dustbin | done | 10 | 9 | 9/10 | 0 | 0 | 0 | 0 | 0 | 9/10 | 9/10 | yes | 20 个掉落事件全部被重力解释 |
| put_object_cabinet | done | 10 | 7 | 7/10 | 0 | 0 | 0 | 0 | 0 | 7/10 | 7/10 | no | 50 seed 排队中；expert 跳过 73 seed |
| rotate_qrcode | 4 |  |  |  |  |  |  |  |  |  |  |  |  |
| scan_object | 4 |  |  |  |  |  |  |  |  |  |  |  |  |
| shake_bottle_horizontally | 4 |  |  |  |  |  |  |  |  |  |  |  |  |
| shake_bottle | 4 |  |  |  |  |  |  |  |  |  |  |  |  |
| stack_blocks_three | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| stack_blocks_two | done | 10 | 10 | 10/10 | 0 | 0 | 0 | 0 | 0 | 10/10 | 10/10 | no | 对照 |
| stack_bowls_three | running |  |  |  |  |  |  |  |  |  |  |  | 50 seed 在跑（node2） |
| stack_bowls_two | done | 10 | 10 | 10/10 | 0 | 0 | 0 | 0 | 0 | 10/10 | 10/10 | no |  |
| stamp_seal | 4 |  |  |  |  |  |  |  |  |  |  |  |  |
| turn_switch | 4 |  |  |  |  |  |  |  |  |  |  |  |  |
