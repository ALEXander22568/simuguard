"""Runtime support for LingBot-VA's 16-dim end-effector actions on RoboTwin.

The ``lingbot-va-posttrain-robotwin`` checkpoint is evaluated on RoboTwin through
the server config ``robotwin``, whose ``used_action_channel_ids`` select 16
channels: left EE pose (7) + left gripper (1) + right EE pose (7) + right
gripper (1).  Those poses are *relative to the episode's initial EE poses*.

Upstream ``XPolicyLab/policy/LingBot_VA/model.py`` only implements the 30-dim
joint path (``robotwin30_train``), so a 16-dim chunk raises.  Driving the joint
channels of ``robotwin30_train`` instead is not equivalent: its de-normalisation
statistics differ from this checkpoint's (q01/q99 differ by up to 1.5/2.6) and
the resulting joint targets jump up to 1.34 rad between policy steps, which
shows up as arm/gripper jitter.

This module installs that path at runtime (upstream files untouched):

* ``update_obs_batch``  captures the initial left/right EE poses,
* ``_convert_to_joint_control_chunk``  converts a 16-dim relative chunk to absolute,
* ``unpack_robot_state``  turns a 16-dim chunk into RoboTwin ``action_type="ee"`` dicts,
* ``reset``  clears the captured initial poses.

Ported from the patch used by the LingBot reproduction runs (93% / 76% success
on other RoboTwin tasks with this checkpoint).  Quaternions are ``xyzw`` here,
matching the RoboTwin EE action layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EE_ACTION_DIM = 16


def compose_quaternion_xyzw(base: Any, delta: Any) -> np.ndarray:
    x1, y1, z1, w1 = base
    x2, y2, z2, w2 = delta
    result = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )
    norm = np.linalg.norm(result)
    if norm <= 1e-8:
        return np.asarray(base, dtype=np.float32)
    return result / norm


def relative_to_absolute(action_chunk: np.ndarray, initial_poses: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    left_base, right_base = initial_poses
    absolute = np.asarray(action_chunk, dtype=np.float32).copy()
    absolute[:, :3] += left_base[:3]
    absolute[:, 3:7] = np.stack([compose_quaternion_xyzw(left_base[3:7], quat) for quat in absolute[:, 3:7]])
    absolute[:, 8:11] += right_base[:3]
    absolute[:, 11:15] = np.stack([compose_quaternion_xyzw(right_base[3:7], quat) for quat in absolute[:, 11:15]])
    return absolute


def chunk_to_action_dicts(action_chunk: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "action_type": "ee",
            "left_ee_pose": action[:7],
            "left_ee_joint_state": action[7:8],
            "right_ee_pose": action[8:15],
            "right_ee_joint_state": action[15:16],
        }
        for action in np.asarray(action_chunk)
    ]


def install(model_module: Any) -> dict[str, Any]:
    """Patch an imported ``XPolicyLab.policy.LingBot_VA.model`` module in place."""

    model_cls = model_module.Model
    if getattr(model_cls, "_simuguard_ee_actions", False):
        return {"already_installed": True}

    original_convert = model_cls._convert_to_joint_control_chunk
    original_update_batch = model_cls.update_obs_batch
    original_reset = model_cls.reset
    original_unpack = model_module.unpack_robot_state

    def _convert_to_joint_control_chunk(self: Any, action_chunk: Any) -> Any:
        action_chunk = np.asarray(action_chunk)
        if action_chunk.ndim == 2 and action_chunk.shape[1] == EE_ACTION_DIM:
            poses = getattr(self, "_simuguard_initial_ee_poses", None)
            if poses is None:
                raise RuntimeError(
                    "16-dim EE actions need the episode's initial EE poses; none were seen in update_obs_batch"
                )
            return relative_to_absolute(action_chunk, poses)
        return original_convert(self, action_chunk)

    def update_obs_batch(self: Any, obs_list: Any) -> Any:
        if getattr(self, "_simuguard_initial_ee_poses", None) is None and obs_list:
            state = obs_list[0].get("state", {}) or {}
            left, right = state.get("left_ee_pose"), state.get("right_ee_pose")
            if left is not None and right is not None:
                self._simuguard_initial_ee_poses = (
                    np.asarray(left, dtype=np.float32).reshape(-1)[:7].copy(),
                    np.asarray(right, dtype=np.float32).reshape(-1)[:7].copy(),
                )
        return original_update_batch(self, obs_list)

    def reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        self._simuguard_initial_ee_poses = None
        return original_reset(self, *args, **kwargs)

    def unpack_robot_state(action_chunk: Any, *args: Any, **kwargs: Any) -> Any:
        array = np.asarray(action_chunk)
        if array.ndim == 2 and array.shape[1] == EE_ACTION_DIM:
            return chunk_to_action_dicts(array)
        return original_unpack(action_chunk, *args, **kwargs)

    model_cls._convert_to_joint_control_chunk = _convert_to_joint_control_chunk
    model_cls.update_obs_batch = update_obs_batch
    model_cls.reset = reset
    model_cls._simuguard_initial_ee_poses = None
    model_cls._simuguard_ee_actions = True
    model_module.unpack_robot_state = unpack_robot_state
    return {"installed": True, "ee_action_dim": EE_ACTION_DIM}
