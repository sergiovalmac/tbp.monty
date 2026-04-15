# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Inverse kinematics solvers for robot-controlled agents.

Two solvers are provided:

- :func:`ik_solve_damped_ls` — A simple iterative damped least-squares solver.
  Effective for the small per-action deltas typical of Monty's movement
  commands.

- :func:`ik_solve_incremental` — Breaks large motions into substeps with
  SLERP rotation interpolation, adaptive damping, and step clamping.
  More robust for large joint-space jumps.
"""
from __future__ import annotations

import mujoco
import numpy as np
from mujoco import MjData, MjModel
from scipy.spatial.transform import Rotation, Slerp


def ik_solve_damped_ls(
    model: MjModel,
    data: MjData,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    ee_site_id: int,
    robot_dof_ids: list[int],
    robot_joint_ids: list[int],
    robot_qpos_addrs: list[int],
    max_iter: int = 500,
    tol: float = 1e-4,
    damping: float = 0.05,
) -> None:
    """Iterative damped least-squares IK for 6-DOF pose.

    A straightforward iterative solver that computes the full pose error
    on every iteration and applies a damped pseudo-inverse update.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (simulation state).
        target_pos: Desired end-effector world position (3,).
        target_mat: Desired end-effector world rotation matrix (3, 3).
        ee_site_id: ID of the end-effector site.
        robot_dof_ids: DOF indices for the robot joints.
        robot_joint_ids: Joint indices for the robot.
        robot_qpos_addrs: qpos addresses for the robot joints.
        max_iter: Maximum number of iterations.  Standard value: 500.
        tol: Convergence tolerance (metres / radians).  Standard value:
            1e-4.
        damping: Damping factor for the least-squares solve.  Standard
            value: 0.05.
    """
    dof_ids = np.array(robot_dof_ids)
    n_dof = len(dof_ids)

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)

        cur_pos = data.site_xpos[ee_site_id]
        cur_mat = data.site_xmat[ee_site_id].reshape(3, 3)

        pos_err = target_pos - cur_pos
        rot_err = Rotation.from_matrix(target_mat @ cur_mat.T).as_rotvec()

        error = np.concatenate([pos_err, rot_err])
        if np.linalg.norm(error) < tol:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
        J = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])

        JJT = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, error)

        for i in range(n_dof):
            data.qpos[robot_qpos_addrs[i]] += dq[i]

        for i, jid in enumerate(robot_joint_ids):
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                addr = robot_qpos_addrs[i]
                data.qpos[addr] = np.clip(data.qpos[addr], lo, hi)


def ik_solve_incremental(
    model: MjModel,
    data: MjData,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    ee_site_id: int,
    robot_dof_ids: list[int],
    robot_joint_ids: list[int],
    robot_qpos_addrs: list[int],
    max_iter: int = 500,
    tol: float = 1e-4,
    damping: float = 0.05,
    max_step: float = 0.2,
) -> None:
    """Incremental IK: interpolate from current to target in substeps.

    Large motions (especially rotations) are broken into small substeps
    so that each Jacobian-based solve handles only a tiny delta, avoiding
    the oscillation problems of single-shot iterative IK.  Uses adaptive
    damping and step clamping for extra robustness.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (simulation state).
        target_pos: Desired end-effector world position (3,).
        target_mat: Desired end-effector world rotation matrix (3, 3).
        ee_site_id: ID of the end-effector site.
        robot_dof_ids: DOF indices for the robot joints.
        robot_joint_ids: Joint indices for the robot.
        robot_qpos_addrs: qpos addresses for the robot joints.
        max_iter: Maximum number of iterations (budget split across
            substeps).  Standard value: 500.
        tol: Convergence tolerance (metres / radians).  Standard value:
            1e-4.
        damping: Base damping factor for the least-squares solve.  Standard
            value: 0.05.
        max_step: Maximum joint delta per iteration (radians).  Standard
            value: 0.2.
    """
    dof_ids = np.array(robot_dof_ids)

    mujoco.mj_forward(model, data)
    start_pos = data.site_xpos[ee_site_id].copy()
    start_mat = data.site_xmat[ee_site_id].reshape(3, 3).copy()

    pos_dist = np.linalg.norm(target_pos - start_pos)
    start_rot = Rotation.from_matrix(start_mat)
    target_rot = Rotation.from_matrix(target_mat)
    rot_dist = np.linalg.norm((target_rot * start_rot.inv()).as_rotvec())

    n_substeps = max(1, int(np.ceil(max(
        pos_dist / 0.002,
        rot_dist / np.radians(2),
    ))))

    key_rots = Rotation.concatenate([start_rot, target_rot])
    slerp = Slerp([0.0, 1.0], key_rots)

    iters_per_substep = max(10, max_iter // n_substeps)

    for substep in range(n_substeps):
        t = (substep + 1) / n_substeps
        sub_pos = start_pos + t * (target_pos - start_pos)
        sub_mat = slerp(t).as_matrix()

        for _ in range(iters_per_substep):
            mujoco.mj_forward(model, data)
            cur_pos = data.site_xpos[ee_site_id]
            cur_mat = data.site_xmat[ee_site_id].reshape(3, 3)

            pos_err = sub_pos - cur_pos
            rot_err = Rotation.from_matrix(
                sub_mat @ cur_mat.T
            ).as_rotvec()

            if (np.linalg.norm(pos_err) < tol
                    and np.linalg.norm(rot_err) < tol):
                break

            error = np.concatenate([pos_err, rot_err])

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
            J = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])

            err_norm = np.linalg.norm(error)
            adaptive_damping = damping * (1.0 + err_norm * 20)
            JJT = J @ J.T + adaptive_damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, error)

            max_abs = np.max(np.abs(dq))
            if max_abs > max_step:
                dq *= max_step / max_abs

            for i in range(len(dof_ids)):
                data.qpos[robot_qpos_addrs[i]] += dq[i]

        for i, jid in enumerate(robot_joint_ids):
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                addr = robot_qpos_addrs[i]
                data.qpos[addr] = np.clip(data.qpos[addr], lo, hi)
