# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Tests that robot agents produce identical movement to their free-floating
counterparts.

The robot agents (RobotDistantAgent, RobotSurfaceAgent) should move the
virtual camera body using the exact same math as the corresponding
free-floating agents (DistantAgent, SurfaceAgent).  The robot arm follows
via IK but must not introduce any position or rotation drift into the
camera body.

Each test is parameterized over both IK methods ("damped_ls" and
"incremental") to verify that the choice of solver does not affect parity.
"""
from __future__ import annotations

import math
import re
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    TurnLeft,
    TurnRight,
)
from tbp.monty.simulators.mujoco.agents import DistantAgent, SurfaceAgent
from tbp.monty.simulators.mujoco.robot_agents import (
    RobotDistantAgent,
    RobotSurfaceAgent,
)
from tbp.monty.simulators.mujoco.simulator import MuJoCoSimulator

AGENT_ID = "agent_id_0"
HOME_QPOS = (0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0)
IK_ARGS = {
    "damped_ls": {"max_iter": 500, "tol": 1e-4, "damping": 0.05},
    "incremental": {"max_iter": 500, "tol": 1e-4, "damping": 0.05, "max_step": 0.2},
}
IK_METHODS = ("damped_ls", "incremental")
OBJ_POS = (-0.492, -0.134, 0.388)
SENSOR_CONFIGS = {
    "patch": {
        "position": (0, 0, 0),
        "rotation": (1, 0, 0, 0),
        "resolution": (64, 64),
        "zoom": 10.0,
    },
    "view_finder": {
        "position": (0, 0, 0),
        "rotation": (1, 0, 0, 0),
        "resolution": (256, 256),
        "zoom": 1.0,
    },
}


def _make_simulator(agent_type, **extra_agent_args):
    """Create a simulator with a single agent and a sphere object."""
    agent_args = {
        "agent_id": AGENT_ID,
        "sensor_configs": SENSOR_CONFIGS,
        **extra_agent_args,
    }
    sim = MuJoCoSimulator(
        agent_configs=[{"agent_type": agent_type, "agent_args": agent_args}],
        ycb_path="/home/sergiov/tbp/data/habitat/objects/ycb",
    )
    sim.add_object("sphere", position=OBJ_POS, scale=(0.03, 0.03, 0.03))
    return sim


def _dispatch_action(agent, action):
    """Call the appropriate actuate method on an agent."""
    name = type(action).__name__
    method = "actuate_" + re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    getattr(agent, method)(action)


def _get_pitch(sim, agent):
    """Read the pitch hinge angle in degrees."""
    addr = sim.model.jnt_qposadr[agent.pitch_joint.id]
    return np.degrees(sim.data.qpos[addr])


def _rotation_from_wxyz(wxyz):
    return Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])


# ---------------------------------------------------------------------------
# Base mixin with shared setup / assertion / action-loop helpers
# ---------------------------------------------------------------------------

class _ParityTestMixin:
    """Shared helpers for robot-agent parity tests.

    Subclasses must set ``ref_agent_type`` and ``robot_agent_type``.
    """

    ref_agent_type = None
    robot_agent_type = None

    def _setup_for_method(self, ik_method):
        self.sim_ref = _make_simulator(self.ref_agent_type)
        self.sim_rob = _make_simulator(
            self.robot_agent_type,
            robot_name="robot:ur5e",
            position=(0.0, 0.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
            ik_method=ik_method,
            ik_args=IK_ARGS[ik_method],
            home_qpos=HOME_QPOS,
        )
        self.ref = self.sim_ref._agents[AGENT_ID]
        self.rob = self.sim_rob._agents[AGENT_ID]
        self.sim_rob.reset()
        self.ref.position = self.rob.position
        self.ref.rotation = self.rob.rotation
        ref_pitch_addr = self.sim_ref.model.jnt_qposadr[self.ref.pitch_joint.id]
        rob_pitch_addr = self.sim_rob.model.jnt_qposadr[self.rob.pitch_joint.id]
        self.sim_ref.data.qpos[ref_pitch_addr] = self.sim_rob.data.qpos[rob_pitch_addr]
        mujoco.mj_forward(self.sim_ref.model, self.sim_ref.data)

    def _assert_poses_equal(self, step_label=""):
        ref_pos = np.array(self.ref.position)
        rob_pos = np.array(self.rob.position)
        np.testing.assert_allclose(
            ref_pos, rob_pos, atol=1e-9,
            err_msg=f"Position mismatch at {step_label}",
        )
        ref_rot = _rotation_from_wxyz(self.ref.rotation)
        rob_rot = _rotation_from_wxyz(self.rob.rotation)
        rot_err_deg = np.degrees(
            np.linalg.norm((ref_rot * rob_rot.inv()).as_rotvec())
        )
        self.assertAlmostEqual(
            rot_err_deg, 0.0, places=6,
            msg=f"Rotation mismatch at {step_label}",
        )
        ref_pitch = _get_pitch(self.sim_ref, self.ref)
        rob_pitch = _get_pitch(self.sim_rob, self.rob)
        self.assertAlmostEqual(
            ref_pitch, rob_pitch, places=6,
            msg=f"Pitch mismatch at {step_label}",
        )

    def _run_action_sequence(self, actions, label_prefix):
        """Execute actions on both agents and assert parity after each step."""
        for ik_method in IK_METHODS:
            with self.subTest(ik_method=ik_method):
                self._setup_for_method(ik_method)
                for i, action in enumerate(actions):
                    _dispatch_action(self.ref, action)
                    _dispatch_action(self.rob, action)
                    self._assert_poses_equal(
                        f"{label_prefix} step {i} ({ik_method})"
                    )


# ---------------------------------------------------------------------------
# Distant-agent parity
# ---------------------------------------------------------------------------

class TestDistantAgentParity(_ParityTestMixin, unittest.TestCase):
    """RobotDistantAgent must produce identical body poses as DistantAgent."""

    ref_agent_type = DistantAgent
    robot_agent_type = RobotDistantAgent

    def test_move_forward_sequence(self):
        """Multiple MoveForward actions produce zero drift."""
        actions = [MoveForward(agent_id=AGENT_ID, distance=0.005)] * 10
        self._run_action_sequence(actions, "MoveForward")

    def test_turn_sequence(self):
        """Alternating TurnRight/TurnLeft produces zero drift."""
        self._run_action_sequence([
            TurnRight(agent_id=AGENT_ID, rotation_degrees=5.0),
            TurnLeft(agent_id=AGENT_ID, rotation_degrees=3.0),
            TurnRight(agent_id=AGENT_ID, rotation_degrees=7.0),
            TurnLeft(agent_id=AGENT_ID, rotation_degrees=5.0),
        ], "Turn")

    def test_look_sequence(self):
        """LookUp/LookDown produces zero drift."""
        self._run_action_sequence([
            LookDown(agent_id=AGENT_ID, rotation_degrees=5.0),
            LookDown(agent_id=AGENT_ID, rotation_degrees=5.0),
            LookUp(agent_id=AGENT_ID, rotation_degrees=3.0),
            LookUp(agent_id=AGENT_ID, rotation_degrees=7.0),
        ], "Look")

    def test_mixed_actions_30_steps(self):
        """30 steps of mixed distant-agent actions produce zero drift."""
        action_cycle = [
            MoveForward(agent_id=AGENT_ID, distance=0.005),
            TurnRight(agent_id=AGENT_ID, rotation_degrees=5.0),
            LookDown(agent_id=AGENT_ID, rotation_degrees=5.0),
            TurnLeft(agent_id=AGENT_ID, rotation_degrees=5.0),
            LookUp(agent_id=AGENT_ID, rotation_degrees=5.0),
        ]
        actions = [action_cycle[i % len(action_cycle)] for i in range(30)]
        self._run_action_sequence(actions, "Mixed")


# ---------------------------------------------------------------------------
# Surface-agent parity
# ---------------------------------------------------------------------------

def _orient_h(degrees=5.0, radius=0.025):
    return OrientHorizontal(
        agent_id=AGENT_ID,
        rotation_degrees=degrees,
        left_distance=radius * math.sin(math.radians(degrees)),
        forward_distance=radius * (1 - math.cos(math.radians(degrees))),
    )


def _orient_v(degrees=5.0, radius=0.025):
    return OrientVertical(
        agent_id=AGENT_ID,
        rotation_degrees=degrees,
        down_distance=radius * math.sin(math.radians(degrees)),
        forward_distance=radius * (1 - math.cos(math.radians(degrees))),
    )


class TestSurfaceAgentParity(_ParityTestMixin, unittest.TestCase):
    """RobotSurfaceAgent must produce identical body poses as SurfaceAgent."""

    ref_agent_type = SurfaceAgent
    robot_agent_type = RobotSurfaceAgent

    def test_move_forward_sequence(self):
        """Multiple MoveForward actions produce zero drift."""
        actions = [MoveForward(agent_id=AGENT_ID, distance=0.005)] * 10
        self._run_action_sequence(actions, "MoveForward")

    def test_orient_horizontal_sequence(self):
        """Repeated OrientHorizontal produces zero drift."""
        actions = [_orient_h()] * 10
        self._run_action_sequence(actions, "OrientH")

    def test_orient_vertical_sequence(self):
        """Repeated OrientVertical produces zero drift."""
        actions = [_orient_v()] * 10
        self._run_action_sequence(actions, "OrientV")

    def test_move_tangentially(self):
        """MoveTangentially produces zero drift."""
        actions = [
            MoveTangentially(
                agent_id=AGENT_ID, distance=0.002, direction=(1.0, 0.0, 0.0),
            )
        ] * 10
        self._run_action_sequence(actions, "MoveTangentially")

    def test_mixed_surface_actions_40_steps(self):
        """40 steps of mixed surface-agent actions produce zero drift."""
        action_cycle = [
            MoveForward(agent_id=AGENT_ID, distance=0.002),
            _orient_h(),
            MoveForward(agent_id=AGENT_ID, distance=0.001),
            _orient_v(),
        ]
        actions = [action_cycle[i % len(action_cycle)] for i in range(40)]
        self._run_action_sequence(actions, "Mixed")
