# RoboTwin narrow intervention and control (LingBot-VA, open-loop replay)

> **Caveat.** Open-loop replay replays the recorded actions; the policy cannot react to a changed state. It therefore overestimates S->F (a success whose state is perturbed cannot be corrected by the policy) and underestimates F->S (a rescued state is still followed by the actions recorded after the failure).

Flip = replayed outcome differs from the recorded outcome. Rates: F->S over recorded failures, S->F over recorded successes, among replays that pass the prefix bit-identity gate; in brackets the rate and its 95% Wilson interval. Replay score = (S + #F->S - #S->F) / N per scope (paper convention). `put_object_cabinet` (3 flagged, not bit-exact) and the 10 pass-2 episodes without control logs are not included.

## Totals

| set | condition | cap | scope | valid/eps | F->S | S->F | official | replay score |
|---|---|---|---|---|---|---|---|---|
| flagged | broad | 1.0 | five_tasks | 59/59 | 32/40 (80%, 65-90) | 9/19 (47%, 27-68) | 83.1 | 91.9 |
| flagged | broad | 1.0 | all_analysed | 67/67 | 33/45 (73%, 59-84) | 10/22 (45%, 27-65) | 82.9 | 88.3 |
| flagged | body | 1.0 | five_tasks | 59/59 | 21/40 (52%, 37-67) | 10/19 (53%, 32-73) | 83.1 | 87.3 |
| flagged | body | 1.0 | all_analysed | 67/67 | 22/45 (49%, 35-63) | 11/22 (50%, 31-69) | 82.9 | 85.5 |
| flagged | partner | 1.0 | five_tasks | 59/59 | 23/40 (58%, 42-71) | 10/19 (53%, 32-73) | 83.1 | 88.1 |
| flagged | partner | 1.0 | all_analysed | 67/67 | 24/45 (53%, 39-67) | 11/22 (50%, 31-69) | 82.9 | 86.0 |
| flagged | resolved | 1.0 | five_tasks | 59/59 | 19/40 (48%, 33-63) | 11/19 (58%, 36-77) | 83.1 | 86.2 |
| flagged | resolved | 1.0 | all_analysed | 67/67 | 20/45 (44%, 31-59) | 12/22 (55%, 35-73) | 82.9 | 84.8 |
| flagged | broad | 0.5 | five_tasks | 59/59 | 31/40 (78%, 62-88) | 9/19 (47%, 27-68) | 83.1 | 91.5 |
| flagged | broad | 0.5 | all_analysed | 67/67 | 32/45 (71%, 57-82) | 10/22 (45%, 27-65) | 82.9 | 88.1 |
| flagged | body | 0.5 | five_tasks | 59/59 | 23/40 (58%, 42-71) | 11/19 (58%, 36-77) | 83.1 | 87.7 |
| flagged | body | 0.5 | all_analysed | 67/67 | 24/45 (53%, 39-67) | 12/22 (55%, 35-73) | 82.9 | 85.7 |
| flagged | partner | 0.5 | five_tasks | 59/59 | 26/40 (65%, 50-78) | 11/19 (58%, 36-77) | 83.1 | 88.8 |
| flagged | partner | 0.5 | all_analysed | 67/67 | 27/45 (60%, 45-73) | 12/22 (55%, 35-73) | 82.9 | 86.4 |
| flagged | resolved | 0.5 | five_tasks | 59/59 | 21/40 (52%, 37-67) | 10/19 (53%, 32-73) | 83.1 | 87.3 |
| flagged | resolved | 0.5 | all_analysed | 67/67 | 22/45 (49%, 35-63) | 11/22 (50%, 31-69) | 82.9 | 85.5 |
| flagged | broad | 2.0 | five_tasks | 59/59 | 26/40 (65%, 50-78) | 9/19 (47%, 27-68) | 83.1 | 89.6 |
| flagged | broad | 2.0 | all_analysed | 67/67 | 27/45 (60%, 45-73) | 9/22 (41%, 23-61) | 82.9 | 87.1 |
| flagged | body | 2.0 | five_tasks | 59/59 | 20/40 (50%, 35-65) | 12/19 (63%, 41-81) | 83.1 | 86.2 |
| flagged | body | 2.0 | all_analysed | 67/67 | 21/45 (47%, 33-61) | 12/22 (55%, 35-73) | 82.9 | 85.0 |
| flagged | partner | 2.0 | five_tasks | 59/59 | 22/40 (55%, 40-69) | 12/19 (63%, 41-81) | 83.1 | 86.9 |
| flagged | partner | 2.0 | all_analysed | 67/67 | 23/45 (51%, 37-65) | 12/22 (55%, 35-73) | 82.9 | 85.5 |
| flagged | resolved | 2.0 | five_tasks | 59/59 | 20/40 (50%, 35-65) | 12/19 (63%, 41-81) | 83.1 | 86.2 |
| flagged | resolved | 2.0 | all_analysed | 67/67 | 21/45 (47%, 33-61) | 13/22 (59%, 39-77) | 82.9 | 84.8 |
| control | body | 1.0 | five_tasks | 97/97 | 0/4 (0%, 0-49) | 0/93 (0%, 0-4) | 83.1 | 83.1 |
| control | body | 1.0 | all_analysed | 113/113 | 0/7 (0%, 0-35) | 0/106 (0%, 0-3) | 82.9 | 82.9 |
| control | partner | 1.0 | five_tasks | 97/97 | 0/4 (0%, 0-49) | 0/93 (0%, 0-4) | 83.1 | 83.1 |
| control | partner | 1.0 | all_analysed | 113/113 | 0/7 (0%, 0-35) | 0/106 (0%, 0-3) | 82.9 | 82.9 |
| control | resolved | 1.0 | five_tasks | 97/97 | 0/4 (0%, 0-49) | 0/93 (0%, 0-4) | 83.1 | 83.1 |
| control | resolved | 1.0 | all_analysed | 113/113 | 0/7 (0%, 0-35) | 0/106 (0%, 0-3) | 82.9 | 82.9 |
| control | body | 0.5 | five_tasks | 97/97 | 0/4 (0%, 0-49) | 0/93 (0%, 0-4) | 83.1 | 83.1 |
| control | body | 0.5 | all_analysed | 113/113 | 0/7 (0%, 0-35) | 0/106 (0%, 0-3) | 82.9 | 82.9 |
| control | partner | 0.5 | five_tasks | 97/97 | 0/4 (0%, 0-49) | 0/93 (0%, 0-4) | 83.1 | 83.1 |
| control | partner | 0.5 | all_analysed | 113/113 | 0/7 (0%, 0-35) | 0/106 (0%, 0-3) | 82.9 | 82.9 |
| control | resolved | 0.5 | five_tasks | 97/97 | 0/4 (0%, 0-49) | 0/93 (0%, 0-4) | 83.1 | 83.1 |
| control | resolved | 0.5 | all_analysed | 113/113 | 0/7 (0%, 0-35) | 0/106 (0%, 0-3) | 82.9 | 82.9 |
| control | body | 2.0 | five_tasks | 97/97 | 0/4 (0%, 0-49) | 0/93 (0%, 0-4) | 83.1 | 83.1 |
| control | body | 2.0 | all_analysed | 113/113 | 0/7 (0%, 0-35) | 0/106 (0%, 0-3) | 82.9 | 82.9 |
| control | partner | 2.0 | five_tasks | 97/97 | 0/4 (0%, 0-49) | 0/93 (0%, 0-4) | 83.1 | 83.1 |
| control | partner | 2.0 | all_analysed | 113/113 | 0/7 (0%, 0-35) | 0/106 (0%, 0-3) | 82.9 | 82.9 |
| control | resolved | 2.0 | five_tasks | 97/97 | 0/4 (0%, 0-49) | 0/93 (0%, 0-4) | 83.1 | 83.1 |
| control | resolved | 2.0 | all_analysed | 113/113 | 0/7 (0%, 0-35) | 0/106 (0%, 0-3) | 82.9 | 82.9 |

## Five headline tasks

| set | condition | cap | scope | valid/eps | F->S | S->F | official | replay score |
|---|---|---|---|---|---|---|---|---|
| flagged | broad | 1.0 | place_can_basket | 37/37 | 18/21 (86%, 65-95) | 7/16 (44%, 23-67) | 74.4 | 86.7 |
| flagged | broad | 1.0 | stack_bowls_three | 12/12 | 7/12 (58%, 32-81) | 0/0 | 76.0 | 90.0 |
| flagged | broad | 1.0 | place_object_basket | 6/6 | 3/3 (100%, 44-100) | 2/3 (67%, 21-94) | 87.5 | 90.0 |
| flagged | broad | 1.0 | place_bread_basket | 1/1 | 1/1 (100%, 21-100) | 0/0 | 97.5 | 100.0 |
| flagged | broad | 1.0 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 92.5 | 100.0 |
| flagged | body | 1.0 | place_can_basket | 37/37 | 14/21 (67%, 45-83) | 8/16 (50%, 28-72) | 74.4 | 81.1 |
| flagged | body | 1.0 | stack_bowls_three | 12/12 | 2/12 (17%, 5-45) | 0/0 | 76.0 | 80.0 |
| flagged | body | 1.0 | place_object_basket | 6/6 | 2/3 (67%, 21-94) | 2/3 (67%, 21-94) | 87.5 | 87.5 |
| flagged | body | 1.0 | place_bread_basket | 1/1 | 0/1 (0%, 0-79) | 0/0 | 97.5 | 97.5 |
| flagged | body | 1.0 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 92.5 | 100.0 |
| flagged | partner | 1.0 | place_can_basket | 37/37 | 14/21 (67%, 45-83) | 8/16 (50%, 28-72) | 74.4 | 81.1 |
| flagged | partner | 1.0 | stack_bowls_three | 12/12 | 4/12 (33%, 14-61) | 0/0 | 76.0 | 84.0 |
| flagged | partner | 1.0 | place_object_basket | 6/6 | 2/3 (67%, 21-94) | 2/3 (67%, 21-94) | 87.5 | 87.5 |
| flagged | partner | 1.0 | place_bread_basket | 1/1 | 0/1 (0%, 0-79) | 0/0 | 97.5 | 97.5 |
| flagged | partner | 1.0 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 92.5 | 100.0 |
| flagged | resolved | 1.0 | place_can_basket | 37/37 | 12/21 (57%, 37-76) | 9/16 (56%, 33-77) | 74.4 | 77.8 |
| flagged | resolved | 1.0 | stack_bowls_three | 12/12 | 2/12 (17%, 5-45) | 0/0 | 76.0 | 80.0 |
| flagged | resolved | 1.0 | place_object_basket | 6/6 | 2/3 (67%, 21-94) | 2/3 (67%, 21-94) | 87.5 | 87.5 |
| flagged | resolved | 1.0 | place_bread_basket | 1/1 | 0/1 (0%, 0-79) | 0/0 | 97.5 | 97.5 |
| flagged | resolved | 1.0 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 92.5 | 100.0 |
| flagged | broad | 0.5 | place_can_basket | 37/37 | 18/21 (86%, 65-95) | 6/16 (38%, 18-61) | 74.4 | 87.8 |
| flagged | broad | 0.5 | stack_bowls_three | 12/12 | 7/12 (58%, 32-81) | 0/0 | 76.0 | 90.0 |
| flagged | broad | 0.5 | place_object_basket | 6/6 | 2/3 (67%, 21-94) | 3/3 (100%, 44-100) | 87.5 | 85.0 |
| flagged | broad | 0.5 | place_bread_basket | 1/1 | 1/1 (100%, 21-100) | 0/0 | 97.5 | 100.0 |
| flagged | broad | 0.5 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 92.5 | 100.0 |
| flagged | body | 0.5 | place_can_basket | 37/37 | 16/21 (76%, 55-89) | 8/16 (50%, 28-72) | 74.4 | 83.3 |
| flagged | body | 0.5 | stack_bowls_three | 12/12 | 3/12 (25%, 9-53) | 0/0 | 76.0 | 82.0 |
| flagged | body | 0.5 | place_object_basket | 6/6 | 1/3 (33%, 6-79) | 3/3 (100%, 44-100) | 87.5 | 82.5 |
| flagged | body | 0.5 | place_bread_basket | 1/1 | 0/1 (0%, 0-79) | 0/0 | 97.5 | 97.5 |
| flagged | body | 0.5 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 92.5 | 100.0 |
| flagged | partner | 0.5 | place_can_basket | 37/37 | 16/21 (76%, 55-89) | 8/16 (50%, 28-72) | 74.4 | 83.3 |
| flagged | partner | 0.5 | stack_bowls_three | 12/12 | 5/12 (42%, 19-68) | 0/0 | 76.0 | 86.0 |
| flagged | partner | 0.5 | place_object_basket | 6/6 | 2/3 (67%, 21-94) | 3/3 (100%, 44-100) | 87.5 | 85.0 |
| flagged | partner | 0.5 | place_bread_basket | 1/1 | 0/1 (0%, 0-79) | 0/0 | 97.5 | 97.5 |
| flagged | partner | 0.5 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 92.5 | 100.0 |
| flagged | resolved | 0.5 | place_can_basket | 37/37 | 14/21 (67%, 45-83) | 7/16 (44%, 23-67) | 74.4 | 82.2 |
| flagged | resolved | 0.5 | stack_bowls_three | 12/12 | 3/12 (25%, 9-53) | 0/0 | 76.0 | 82.0 |
| flagged | resolved | 0.5 | place_object_basket | 6/6 | 1/3 (33%, 6-79) | 3/3 (100%, 44-100) | 87.5 | 82.5 |
| flagged | resolved | 0.5 | place_bread_basket | 1/1 | 0/1 (0%, 0-79) | 0/0 | 97.5 | 97.5 |
| flagged | resolved | 0.5 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 92.5 | 100.0 |
| flagged | broad | 2.0 | place_can_basket | 37/37 | 16/21 (76%, 55-89) | 6/16 (38%, 18-61) | 74.4 | 85.6 |
| flagged | broad | 2.0 | stack_bowls_three | 12/12 | 4/12 (33%, 14-61) | 0/0 | 76.0 | 84.0 |
| flagged | broad | 2.0 | place_object_basket | 6/6 | 2/3 (67%, 21-94) | 3/3 (100%, 44-100) | 87.5 | 85.0 |
| flagged | broad | 2.0 | place_bread_basket | 1/1 | 1/1 (100%, 21-100) | 0/0 | 97.5 | 100.0 |
| flagged | broad | 2.0 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 92.5 | 100.0 |
| flagged | body | 2.0 | place_can_basket | 37/37 | 14/21 (67%, 45-83) | 9/16 (56%, 33-77) | 74.4 | 80.0 |
| flagged | body | 2.0 | stack_bowls_three | 12/12 | 3/12 (25%, 9-53) | 0/0 | 76.0 | 82.0 |
| flagged | body | 2.0 | place_object_basket | 6/6 | 0/3 (0%, 0-56) | 3/3 (100%, 44-100) | 87.5 | 80.0 |
| flagged | body | 2.0 | place_bread_basket | 1/1 | 0/1 (0%, 0-79) | 0/0 | 97.5 | 97.5 |
| flagged | body | 2.0 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 92.5 | 100.0 |
| flagged | partner | 2.0 | place_can_basket | 37/37 | 14/21 (67%, 45-83) | 9/16 (56%, 33-77) | 74.4 | 80.0 |
| flagged | partner | 2.0 | stack_bowls_three | 12/12 | 5/12 (42%, 19-68) | 0/0 | 76.0 | 86.0 |
| flagged | partner | 2.0 | place_object_basket | 6/6 | 0/3 (0%, 0-56) | 3/3 (100%, 44-100) | 87.5 | 80.0 |
| flagged | partner | 2.0 | place_bread_basket | 1/1 | 0/1 (0%, 0-79) | 0/0 | 97.5 | 97.5 |
| flagged | partner | 2.0 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 92.5 | 100.0 |
| flagged | resolved | 2.0 | place_can_basket | 37/37 | 15/21 (71%, 50-86) | 9/16 (56%, 33-77) | 74.4 | 81.1 |
| flagged | resolved | 2.0 | stack_bowls_three | 12/12 | 2/12 (17%, 5-45) | 0/0 | 76.0 | 80.0 |
| flagged | resolved | 2.0 | place_object_basket | 6/6 | 1/3 (33%, 6-79) | 3/3 (100%, 44-100) | 87.5 | 82.5 |
| flagged | resolved | 2.0 | place_bread_basket | 1/1 | 0/1 (0%, 0-79) | 0/0 | 97.5 | 97.5 |
| flagged | resolved | 2.0 | stack_bowls_two | 3/3 | 2/3 (67%, 21-94) | 0/0 | 92.5 | 97.5 |
| control | body | 1.0 | place_can_basket | 53/53 | 0/2 (0%, 0-66) | 0/51 (0%, 0-7) | 74.4 | 74.4 |
| control | body | 1.0 | stack_bowls_three | 24/24 | 0/0 | 0/24 (0%, 0-14) | 76.0 | 76.0 |
| control | body | 1.0 | place_object_basket | 12/12 | 0/2 (0%, 0-66) | 0/10 (0%, 0-28) | 87.5 | 87.5 |
| control | body | 1.0 | place_bread_basket | 2/2 | 0/0 | 0/2 (0%, 0-66) | 97.5 | 97.5 |
| control | body | 1.0 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 92.5 | 92.5 |
| control | partner | 1.0 | place_can_basket | 53/53 | 0/2 (0%, 0-66) | 0/51 (0%, 0-7) | 74.4 | 74.4 |
| control | partner | 1.0 | stack_bowls_three | 24/24 | 0/0 | 0/24 (0%, 0-14) | 76.0 | 76.0 |
| control | partner | 1.0 | place_object_basket | 12/12 | 0/2 (0%, 0-66) | 0/10 (0%, 0-28) | 87.5 | 87.5 |
| control | partner | 1.0 | place_bread_basket | 2/2 | 0/0 | 0/2 (0%, 0-66) | 97.5 | 97.5 |
| control | partner | 1.0 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 92.5 | 92.5 |
| control | resolved | 1.0 | place_can_basket | 53/53 | 0/2 (0%, 0-66) | 0/51 (0%, 0-7) | 74.4 | 74.4 |
| control | resolved | 1.0 | stack_bowls_three | 24/24 | 0/0 | 0/24 (0%, 0-14) | 76.0 | 76.0 |
| control | resolved | 1.0 | place_object_basket | 12/12 | 0/2 (0%, 0-66) | 0/10 (0%, 0-28) | 87.5 | 87.5 |
| control | resolved | 1.0 | place_bread_basket | 2/2 | 0/0 | 0/2 (0%, 0-66) | 97.5 | 97.5 |
| control | resolved | 1.0 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 92.5 | 92.5 |
| control | body | 0.5 | place_can_basket | 53/53 | 0/2 (0%, 0-66) | 0/51 (0%, 0-7) | 74.4 | 74.4 |
| control | body | 0.5 | stack_bowls_three | 24/24 | 0/0 | 0/24 (0%, 0-14) | 76.0 | 76.0 |
| control | body | 0.5 | place_object_basket | 12/12 | 0/2 (0%, 0-66) | 0/10 (0%, 0-28) | 87.5 | 87.5 |
| control | body | 0.5 | place_bread_basket | 2/2 | 0/0 | 0/2 (0%, 0-66) | 97.5 | 97.5 |
| control | body | 0.5 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 92.5 | 92.5 |
| control | partner | 0.5 | place_can_basket | 53/53 | 0/2 (0%, 0-66) | 0/51 (0%, 0-7) | 74.4 | 74.4 |
| control | partner | 0.5 | stack_bowls_three | 24/24 | 0/0 | 0/24 (0%, 0-14) | 76.0 | 76.0 |
| control | partner | 0.5 | place_object_basket | 12/12 | 0/2 (0%, 0-66) | 0/10 (0%, 0-28) | 87.5 | 87.5 |
| control | partner | 0.5 | place_bread_basket | 2/2 | 0/0 | 0/2 (0%, 0-66) | 97.5 | 97.5 |
| control | partner | 0.5 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 92.5 | 92.5 |
| control | resolved | 0.5 | place_can_basket | 53/53 | 0/2 (0%, 0-66) | 0/51 (0%, 0-7) | 74.4 | 74.4 |
| control | resolved | 0.5 | stack_bowls_three | 24/24 | 0/0 | 0/24 (0%, 0-14) | 76.0 | 76.0 |
| control | resolved | 0.5 | place_object_basket | 12/12 | 0/2 (0%, 0-66) | 0/10 (0%, 0-28) | 87.5 | 87.5 |
| control | resolved | 0.5 | place_bread_basket | 2/2 | 0/0 | 0/2 (0%, 0-66) | 97.5 | 97.5 |
| control | resolved | 0.5 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 92.5 | 92.5 |
| control | body | 2.0 | place_can_basket | 53/53 | 0/2 (0%, 0-66) | 0/51 (0%, 0-7) | 74.4 | 74.4 |
| control | body | 2.0 | stack_bowls_three | 24/24 | 0/0 | 0/24 (0%, 0-14) | 76.0 | 76.0 |
| control | body | 2.0 | place_object_basket | 12/12 | 0/2 (0%, 0-66) | 0/10 (0%, 0-28) | 87.5 | 87.5 |
| control | body | 2.0 | place_bread_basket | 2/2 | 0/0 | 0/2 (0%, 0-66) | 97.5 | 97.5 |
| control | body | 2.0 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 92.5 | 92.5 |
| control | partner | 2.0 | place_can_basket | 53/53 | 0/2 (0%, 0-66) | 0/51 (0%, 0-7) | 74.4 | 74.4 |
| control | partner | 2.0 | stack_bowls_three | 24/24 | 0/0 | 0/24 (0%, 0-14) | 76.0 | 76.0 |
| control | partner | 2.0 | place_object_basket | 12/12 | 0/2 (0%, 0-66) | 0/10 (0%, 0-28) | 87.5 | 87.5 |
| control | partner | 2.0 | place_bread_basket | 2/2 | 0/0 | 0/2 (0%, 0-66) | 97.5 | 97.5 |
| control | partner | 2.0 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 92.5 | 92.5 |
| control | resolved | 2.0 | place_can_basket | 53/53 | 0/2 (0%, 0-66) | 0/51 (0%, 0-7) | 74.4 | 74.4 |
| control | resolved | 2.0 | stack_bowls_three | 24/24 | 0/0 | 0/24 (0%, 0-14) | 76.0 | 76.0 |
| control | resolved | 2.0 | place_object_basket | 12/12 | 0/2 (0%, 0-66) | 0/10 (0%, 0-28) | 87.5 | 87.5 |
| control | resolved | 2.0 | place_bread_basket | 2/2 | 0/0 | 0/2 (0%, 0-66) | 97.5 | 97.5 |
| control | resolved | 2.0 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 92.5 | 92.5 |

> **Caveat.** Open-loop replay replays the recorded actions; the policy cannot react to a changed state. It therefore overestimates S->F (a success whose state is perturbed cannot be corrected by the policy) and underestimates F->S (a rescued state is still followed by the actions recorded after the failure).

Per-episode rows: `narrow_episodes.csv`.


## HyVLA

> **Caveat.** Open-loop replay replays the recorded actions; the policy cannot react to a changed state. It therefore overestimates S->F (a success whose state is perturbed cannot be corrected by the policy) and underestimates F->S (a rescued state is still followed by the actions recorded after the failure).

Flip = replayed outcome differs from the recorded outcome. Rates: F->S over recorded failures, S->F over recorded successes, among replays that pass the prefix bit-identity gate; in brackets the rate and its 95% Wilson interval. Replay score = (S + #F->S - #S->F) / N per scope (paper convention). Object-basket shards merged to 20 distinct seeds (lower-start shard kept). `place_bread_basket`, `place_object_basket` and the six other tasks have no flagged episodes, so their replay score equals the official score. The analysed total covers only the three tasks with flagged episodes.

### Totals

| set | condition | cap | scope | valid/eps | F->S | S->F | official | replay score |
|---|---|---|---|---|---|---|---|---|
| flagged | broad | 1.0 | five_tasks | 10/10 | 5/6 (83%, 44-97) | 2/4 (50%, 15-85) | 90.0 | 93.0 |
| flagged | broad | 1.0 | all_analysed | 10/10 | 5/6 (83%, 44-97) | 2/4 (50%, 15-85) | 85.0 | 90.0 |
| flagged | body | 1.0 | five_tasks | 10/10 | 4/6 (67%, 30-90) | 3/4 (75%, 30-95) | 90.0 | 91.0 |
| flagged | body | 1.0 | all_analysed | 10/10 | 4/6 (67%, 30-90) | 3/4 (75%, 30-95) | 85.0 | 86.7 |
| flagged | partner | 1.0 | five_tasks | 10/10 | 4/6 (67%, 30-90) | 4/4 (100%, 51-100) | 90.0 | 90.0 |
| flagged | partner | 1.0 | all_analysed | 10/10 | 4/6 (67%, 30-90) | 4/4 (100%, 51-100) | 85.0 | 85.0 |
| flagged | resolved | 1.0 | five_tasks | 10/10 | 4/6 (67%, 30-90) | 1/4 (25%, 5-70) | 90.0 | 93.0 |
| flagged | resolved | 1.0 | all_analysed | 10/10 | 4/6 (67%, 30-90) | 1/4 (25%, 5-70) | 85.0 | 90.0 |
| flagged | broad | 0.5 | five_tasks | 10/10 | 5/6 (83%, 44-97) | 1/4 (25%, 5-70) | 90.0 | 94.0 |
| flagged | broad | 0.5 | all_analysed | 10/10 | 5/6 (83%, 44-97) | 1/4 (25%, 5-70) | 85.0 | 91.7 |
| flagged | body | 0.5 | five_tasks | 10/10 | 5/6 (83%, 44-97) | 2/4 (50%, 15-85) | 90.0 | 93.0 |
| flagged | body | 0.5 | all_analysed | 10/10 | 5/6 (83%, 44-97) | 2/4 (50%, 15-85) | 85.0 | 90.0 |
| flagged | partner | 0.5 | five_tasks | 10/10 | 5/6 (83%, 44-97) | 2/4 (50%, 15-85) | 90.0 | 93.0 |
| flagged | partner | 0.5 | all_analysed | 10/10 | 5/6 (83%, 44-97) | 2/4 (50%, 15-85) | 85.0 | 90.0 |
| flagged | resolved | 0.5 | five_tasks | 10/10 | 5/6 (83%, 44-97) | 2/4 (50%, 15-85) | 90.0 | 93.0 |
| flagged | resolved | 0.5 | all_analysed | 10/10 | 5/6 (83%, 44-97) | 2/4 (50%, 15-85) | 85.0 | 90.0 |
| flagged | broad | 2.0 | five_tasks | 10/10 | 3/6 (50%, 19-81) | 3/4 (75%, 30-95) | 90.0 | 90.0 |
| flagged | broad | 2.0 | all_analysed | 10/10 | 3/6 (50%, 19-81) | 3/4 (75%, 30-95) | 85.0 | 85.0 |
| flagged | body | 2.0 | five_tasks | 10/10 | 3/6 (50%, 19-81) | 2/4 (50%, 15-85) | 90.0 | 91.0 |
| flagged | body | 2.0 | all_analysed | 10/10 | 3/6 (50%, 19-81) | 2/4 (50%, 15-85) | 85.0 | 86.7 |
| flagged | partner | 2.0 | five_tasks | 10/10 | 3/6 (50%, 19-81) | 2/4 (50%, 15-85) | 90.0 | 91.0 |
| flagged | partner | 2.0 | all_analysed | 10/10 | 3/6 (50%, 19-81) | 2/4 (50%, 15-85) | 85.0 | 86.7 |
| flagged | resolved | 2.0 | five_tasks | 10/10 | 3/6 (50%, 19-81) | 2/4 (50%, 15-85) | 90.0 | 91.0 |
| flagged | resolved | 2.0 | all_analysed | 10/10 | 3/6 (50%, 19-81) | 2/4 (50%, 15-85) | 85.0 | 86.7 |
| control | body | 1.0 | five_tasks | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 90.0 | 90.0 |
| control | body | 1.0 | all_analysed | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 85.0 | 85.0 |
| control | partner | 1.0 | five_tasks | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 90.0 | 90.0 |
| control | partner | 1.0 | all_analysed | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 85.0 | 85.0 |
| control | resolved | 1.0 | five_tasks | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 90.0 | 90.0 |
| control | resolved | 1.0 | all_analysed | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 85.0 | 85.0 |
| control | body | 0.5 | five_tasks | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 90.0 | 90.0 |
| control | body | 0.5 | all_analysed | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 85.0 | 85.0 |
| control | partner | 0.5 | five_tasks | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 90.0 | 90.0 |
| control | partner | 0.5 | all_analysed | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 85.0 | 85.0 |
| control | resolved | 0.5 | five_tasks | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 90.0 | 90.0 |
| control | resolved | 0.5 | all_analysed | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 85.0 | 85.0 |
| control | body | 2.0 | five_tasks | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 90.0 | 90.0 |
| control | body | 2.0 | all_analysed | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 85.0 | 85.0 |
| control | partner | 2.0 | five_tasks | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 90.0 | 90.0 |
| control | partner | 2.0 | all_analysed | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 85.0 | 85.0 |
| control | resolved | 2.0 | five_tasks | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 90.0 | 90.0 |
| control | resolved | 2.0 | all_analysed | 20/20 | 0/3 (0%, 0-56) | 0/17 (0%, 0-18) | 85.0 | 85.0 |

### Five headline tasks

| set | condition | cap | scope | valid/eps | F->S | S->F | official | replay score |
|---|---|---|---|---|---|---|---|---|
| flagged | broad | 1.0 | place_can_basket | 4/4 | 1/1 (100%, 21-100) | 2/3 (67%, 21-94) | 80.0 | 75.0 |
| flagged | broad | 1.0 | stack_bowls_three | 3/3 | 1/2 (50%, 9-91) | 0/1 (0%, 0-79) | 90.0 | 95.0 |
| flagged | broad | 1.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| flagged | broad | 1.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| flagged | broad | 1.0 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 85.0 | 100.0 |
| flagged | body | 1.0 | place_can_basket | 4/4 | 1/1 (100%, 21-100) | 2/3 (67%, 21-94) | 80.0 | 75.0 |
| flagged | body | 1.0 | stack_bowls_three | 3/3 | 0/2 (0%, 0-66) | 1/1 (100%, 21-100) | 90.0 | 85.0 |
| flagged | body | 1.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| flagged | body | 1.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| flagged | body | 1.0 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 85.0 | 100.0 |
| flagged | partner | 1.0 | place_can_basket | 4/4 | 1/1 (100%, 21-100) | 3/3 (100%, 44-100) | 80.0 | 70.0 |
| flagged | partner | 1.0 | stack_bowls_three | 3/3 | 0/2 (0%, 0-66) | 1/1 (100%, 21-100) | 90.0 | 85.0 |
| flagged | partner | 1.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| flagged | partner | 1.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| flagged | partner | 1.0 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 85.0 | 100.0 |
| flagged | resolved | 1.0 | place_can_basket | 4/4 | 1/1 (100%, 21-100) | 1/3 (33%, 6-79) | 80.0 | 80.0 |
| flagged | resolved | 1.0 | stack_bowls_three | 3/3 | 0/2 (0%, 0-66) | 0/1 (0%, 0-79) | 90.0 | 90.0 |
| flagged | resolved | 1.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| flagged | resolved | 1.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| flagged | resolved | 1.0 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 85.0 | 100.0 |
| flagged | broad | 0.5 | place_can_basket | 4/4 | 1/1 (100%, 21-100) | 1/3 (33%, 6-79) | 80.0 | 80.0 |
| flagged | broad | 0.5 | stack_bowls_three | 3/3 | 1/2 (50%, 9-91) | 0/1 (0%, 0-79) | 90.0 | 95.0 |
| flagged | broad | 0.5 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| flagged | broad | 0.5 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| flagged | broad | 0.5 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 85.0 | 100.0 |
| flagged | body | 0.5 | place_can_basket | 4/4 | 1/1 (100%, 21-100) | 2/3 (67%, 21-94) | 80.0 | 75.0 |
| flagged | body | 0.5 | stack_bowls_three | 3/3 | 1/2 (50%, 9-91) | 0/1 (0%, 0-79) | 90.0 | 95.0 |
| flagged | body | 0.5 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| flagged | body | 0.5 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| flagged | body | 0.5 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 85.0 | 100.0 |
| flagged | partner | 0.5 | place_can_basket | 4/4 | 1/1 (100%, 21-100) | 2/3 (67%, 21-94) | 80.0 | 75.0 |
| flagged | partner | 0.5 | stack_bowls_three | 3/3 | 1/2 (50%, 9-91) | 0/1 (0%, 0-79) | 90.0 | 95.0 |
| flagged | partner | 0.5 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| flagged | partner | 0.5 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| flagged | partner | 0.5 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 85.0 | 100.0 |
| flagged | resolved | 0.5 | place_can_basket | 4/4 | 1/1 (100%, 21-100) | 2/3 (67%, 21-94) | 80.0 | 75.0 |
| flagged | resolved | 0.5 | stack_bowls_three | 3/3 | 1/2 (50%, 9-91) | 0/1 (0%, 0-79) | 90.0 | 95.0 |
| flagged | resolved | 0.5 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| flagged | resolved | 0.5 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| flagged | resolved | 0.5 | stack_bowls_two | 3/3 | 3/3 (100%, 44-100) | 0/0 | 85.0 | 100.0 |
| flagged | broad | 2.0 | place_can_basket | 4/4 | 1/1 (100%, 21-100) | 3/3 (100%, 44-100) | 80.0 | 70.0 |
| flagged | broad | 2.0 | stack_bowls_three | 3/3 | 1/2 (50%, 9-91) | 0/1 (0%, 0-79) | 90.0 | 95.0 |
| flagged | broad | 2.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| flagged | broad | 2.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| flagged | broad | 2.0 | stack_bowls_two | 3/3 | 1/3 (33%, 6-79) | 0/0 | 85.0 | 90.0 |
| flagged | body | 2.0 | place_can_basket | 4/4 | 1/1 (100%, 21-100) | 2/3 (67%, 21-94) | 80.0 | 75.0 |
| flagged | body | 2.0 | stack_bowls_three | 3/3 | 1/2 (50%, 9-91) | 0/1 (0%, 0-79) | 90.0 | 95.0 |
| flagged | body | 2.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| flagged | body | 2.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| flagged | body | 2.0 | stack_bowls_two | 3/3 | 1/3 (33%, 6-79) | 0/0 | 85.0 | 90.0 |
| flagged | partner | 2.0 | place_can_basket | 4/4 | 1/1 (100%, 21-100) | 2/3 (67%, 21-94) | 80.0 | 75.0 |
| flagged | partner | 2.0 | stack_bowls_three | 3/3 | 1/2 (50%, 9-91) | 0/1 (0%, 0-79) | 90.0 | 95.0 |
| flagged | partner | 2.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| flagged | partner | 2.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| flagged | partner | 2.0 | stack_bowls_two | 3/3 | 1/3 (33%, 6-79) | 0/0 | 85.0 | 90.0 |
| flagged | resolved | 2.0 | place_can_basket | 4/4 | 1/1 (100%, 21-100) | 2/3 (67%, 21-94) | 80.0 | 75.0 |
| flagged | resolved | 2.0 | stack_bowls_three | 3/3 | 1/2 (50%, 9-91) | 0/1 (0%, 0-79) | 90.0 | 95.0 |
| flagged | resolved | 2.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| flagged | resolved | 2.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| flagged | resolved | 2.0 | stack_bowls_two | 3/3 | 1/3 (33%, 6-79) | 0/0 | 85.0 | 90.0 |
| control | body | 1.0 | place_can_basket | 8/8 | 0/3 (0%, 0-56) | 0/5 (0%, 0-43) | 80.0 | 80.0 |
| control | body | 1.0 | stack_bowls_three | 6/6 | 0/0 | 0/6 (0%, 0-39) | 90.0 | 90.0 |
| control | body | 1.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| control | body | 1.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| control | body | 1.0 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 85.0 | 85.0 |
| control | partner | 1.0 | place_can_basket | 8/8 | 0/3 (0%, 0-56) | 0/5 (0%, 0-43) | 80.0 | 80.0 |
| control | partner | 1.0 | stack_bowls_three | 6/6 | 0/0 | 0/6 (0%, 0-39) | 90.0 | 90.0 |
| control | partner | 1.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| control | partner | 1.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| control | partner | 1.0 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 85.0 | 85.0 |
| control | resolved | 1.0 | place_can_basket | 8/8 | 0/3 (0%, 0-56) | 0/5 (0%, 0-43) | 80.0 | 80.0 |
| control | resolved | 1.0 | stack_bowls_three | 6/6 | 0/0 | 0/6 (0%, 0-39) | 90.0 | 90.0 |
| control | resolved | 1.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| control | resolved | 1.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| control | resolved | 1.0 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 85.0 | 85.0 |
| control | body | 0.5 | place_can_basket | 8/8 | 0/3 (0%, 0-56) | 0/5 (0%, 0-43) | 80.0 | 80.0 |
| control | body | 0.5 | stack_bowls_three | 6/6 | 0/0 | 0/6 (0%, 0-39) | 90.0 | 90.0 |
| control | body | 0.5 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| control | body | 0.5 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| control | body | 0.5 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 85.0 | 85.0 |
| control | partner | 0.5 | place_can_basket | 8/8 | 0/3 (0%, 0-56) | 0/5 (0%, 0-43) | 80.0 | 80.0 |
| control | partner | 0.5 | stack_bowls_three | 6/6 | 0/0 | 0/6 (0%, 0-39) | 90.0 | 90.0 |
| control | partner | 0.5 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| control | partner | 0.5 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| control | partner | 0.5 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 85.0 | 85.0 |
| control | resolved | 0.5 | place_can_basket | 8/8 | 0/3 (0%, 0-56) | 0/5 (0%, 0-43) | 80.0 | 80.0 |
| control | resolved | 0.5 | stack_bowls_three | 6/6 | 0/0 | 0/6 (0%, 0-39) | 90.0 | 90.0 |
| control | resolved | 0.5 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| control | resolved | 0.5 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| control | resolved | 0.5 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 85.0 | 85.0 |
| control | body | 2.0 | place_can_basket | 8/8 | 0/3 (0%, 0-56) | 0/5 (0%, 0-43) | 80.0 | 80.0 |
| control | body | 2.0 | stack_bowls_three | 6/6 | 0/0 | 0/6 (0%, 0-39) | 90.0 | 90.0 |
| control | body | 2.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| control | body | 2.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| control | body | 2.0 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 85.0 | 85.0 |
| control | partner | 2.0 | place_can_basket | 8/8 | 0/3 (0%, 0-56) | 0/5 (0%, 0-43) | 80.0 | 80.0 |
| control | partner | 2.0 | stack_bowls_three | 6/6 | 0/0 | 0/6 (0%, 0-39) | 90.0 | 90.0 |
| control | partner | 2.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| control | partner | 2.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| control | partner | 2.0 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 85.0 | 85.0 |
| control | resolved | 2.0 | place_can_basket | 8/8 | 0/3 (0%, 0-56) | 0/5 (0%, 0-43) | 80.0 | 80.0 |
| control | resolved | 2.0 | stack_bowls_three | 6/6 | 0/0 | 0/6 (0%, 0-39) | 90.0 | 90.0 |
| control | resolved | 2.0 | place_object_basket | 0/0 | 0/0 | 0/0 | 95.0 | 95.0 |
| control | resolved | 2.0 | place_bread_basket | 0/0 | 0/0 | 0/0 | 100.0 | 100.0 |
| control | resolved | 2.0 | stack_bowls_two | 6/6 | 0/0 | 0/6 (0%, 0-39) | 85.0 | 85.0 |

> **Caveat.** Open-loop replay replays the recorded actions; the policy cannot react to a changed state. It therefore overestimates S->F (a success whose state is perturbed cannot be corrected by the policy) and underestimates F->S (a rescued state is still followed by the actions recorded after the failure).

Per-episode rows: `narrow_hyvla_episodes.csv`.
