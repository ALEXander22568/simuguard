# 4090-B environment for run_xpolicylab_eval.sh / xpolicylab_campaign.sh with Hy-Embodied-0.5-VLA
export SIMUGUARD_WORKSPACE=/path/to/workspace ROBOTWIN_ROOT=/path/to/workspace/RoboTwin
export SIMUGUARD_CONFIG=/path/to/workspace/SimuGuard/configs/scale_no_bundles.json
export MANIFEST_PYTHON=/path/to/RoboTwin-2.0/runtime/lingbot-va-repro/.venv-client/bin/python
export POLICY_NAME=Hy_Embodied_05_VLA ACTION_TYPE=ee CKPT_NAME=Hy-Embodied-0.5-VLA-RoboTwin
export DEPLOY_YML=XPolicyLab/policy/Hy_Embodied_05_VLA/deploy_robotwin.yml
export POLICY_PY=/path/to/workspace/RoboTwin/XPolicyLab/policy/Hy_Embodied_05_VLA/Hy-Embodied-0.5-VLA/.venv/bin/python
export POLICY_PYTHONPATH=/path/to/workspace/RoboTwin/XPolicyLab/policy/Hy_Embodied_05_VLA/Hy-Embodied-0.5-VLA
export MIN_FREE_POLICY_MIB=14000
