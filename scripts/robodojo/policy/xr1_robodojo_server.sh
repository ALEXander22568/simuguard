#!/usr/bin/env bash
# XR-1 (Xiaomi_Robotics_1, official RoboDojo checkpoint RoboDojo-sim-arx_x5-ee-0) policy server for
# SimuGuard x RoboDojo, bound to 127.0.0.1:${PORT:-16601}.  XPolicyLab code = RoboDojo image copy
# (unmodified); env = ~/xr1-deploy/mibot-env (read-only use) + $R/pylib for the XPolicyLab ws deps.
# usage: GPU=7 PORT=16601 xr1_robodojo_server.sh
R=/data/shared/zhoujingjing/simuguard-robodojo-policy
CK=/data/shared/zhoujingjing/checkpoints/robodojo
cd $R/RoboDojo
export PYTHONPATH=$R/RoboDojo/XPolicyLab/policy/Xiaomi_Robotics_1/xiaomi_robotics_1:$R/RoboDojo:$R/RoboDojo/XPolicyLab:$R/pylib
export CUDA_VISIBLE_DEVICES=${GPU:-7} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore::UserWarning PYTHONUNBUFFERED=1
exec ~/xr1-deploy/mibot-env/bin/python XPolicyLab/setup_policy_server.py \
  --config_path XPolicyLab/policy/Xiaomi_Robotics_1/deploy.yml \
  --overrides port=${PORT:-16601} host=127.0.0.1 bench_name=RoboDojo task_name=robodojo \
    ckpt_name=RoboDojo-sim-arx_x5-ee-0 env_cfg_type=arx_x5 seed=0 policy_name=Xiaomi_Robotics_1 action_type=ee \
    model_dir=$CK/Xiaomi_Robotics_1/RoboDojo-sim-arx_x5-ee-0 vlm_processor_path=$CK/Qwen3-VL-4B-Instruct-processor
