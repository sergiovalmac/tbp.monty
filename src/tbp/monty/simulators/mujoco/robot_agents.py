# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.transform_utils import (
    rotation_as_quat,
    rotation_from_quat,
)
from tbp.monty.math import QuaternionWXYZ, VectorXYZ
from tbp.monty.frameworks.sensors import SensorConfig
from tbp.monty.simulators.mujoco.agents import Embodiment
from tbp.monty.simulators.mujoco.inverse_kinematics import (
    ik_solve_damped_ls,
    ik_solve_incremental,
)

if TYPE_CHECKING:
    from tbp.monty.simulators.mujoco import MuJoCoSimulator

logger = logging.getLogger(__name__)


class RobotAgentBase(Embodiment):
    """Base class for robot-controlled agents using inverse kinematics.

    Attaches cameras to a "virtual" body that tracks the robot's
    end-effector.  On every action the desired end-effector pose is computed
    with the same math as the corresponding free-floating agent, then solved
    for joint positions using damped least-squares IK.

    Subclasses add the action methods appropriate for their policy
    (e.g. look/turn for :class:`RobotDistantAgent`, orient/tangential for
    :class:`RobotSurfaceAgent`).

    Args:
        simulator: The parent MuJoCo simulator.
        agent_id: Unique identifier for this agent.
        sensor_configs: Camera/sensor configuration per sensor ID.
        robot_name: Robot identifier, e.g. ``"robot:ur5e"``.
        position: Initial position (overridden by forward kinematics on
            reset when *home_qpos* is provided).
        rotation: Initial rotation (overridden by forward kinematics on
            reset when *home_qpos* is provided).
        ik_method: Which IK solver to use (``"damped_ls"`` or
            ``"incremental"``).
        ik_args: Parameters forwarded to the IK solver.  Required keys
            depend on the chosen *ik_method*.
        ee_site_name: Name of the MuJoCo site that marks the end-effector.
        ee_rotation_offset: Quaternion (WXYZ) rotating the EE frame so the
            camera looks outward.
        home_qpos: Joint angles to set on reset.  If ``None`` all joints
            start at zero.
    """

    def __init__(
        self,
        simulator: MuJoCoSimulator,
        agent_id: AgentID,
        sensor_configs: dict[SensorID, SensorConfig],
        robot_name: str,
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
        ik_method: str,
        ik_args: dict,
        ee_site_name: str = "attachment_site",
        ee_rotation_offset: QuaternionWXYZ = (0.0, 1.0, 0.0, 0.0),
        home_qpos: tuple[float, ...] | None = None,
    ):
        # Load the robot MJCF into the spec before the base class adds the
        # agent body.  ``spec.from_file`` replaces the spec contents, so this
        # must happen first; then ``super().__init__`` appends the camera body
        # on top of the robot.
        self._load_robot_into_spec(simulator, robot_name)

        super().__init__(simulator, agent_id, sensor_configs, position, rotation)
        self._robot_name = robot_name
        self._ee_site_name = ee_site_name
        self._ee_rotation_offset = Rotation.from_quat(
            [ee_rotation_offset[1], ee_rotation_offset[2],
             ee_rotation_offset[3], ee_rotation_offset[0]]
        )
        self._ik_method = ik_method
        self._ik_args = ik_args
        self._home_qpos = home_qpos

        # Populated lazily after the model is compiled with robot bodies.
        self._robot_initialized = False
        self._ee_site_id: int = -1
        self._robot_joint_ids: list[int] = []
        self._robot_dof_ids: list[int] = []
        self._robot_qpos_addrs: list[int] = []

    @staticmethod
    def _load_robot_into_spec(simulator: MuJoCoSimulator, robot_name: str) -> None:
        """Load a robot MJCF into the simulator's spec.

        This uses ``spec.from_file`` which replaces the spec contents, so it
        must be called before any other bodies (agents, objects) are added.

        Args:
            simulator: The simulator whose spec will be populated.
            robot_name: Robot identifier, e.g. ``"robot:ur5e"``.
        """
        import importlib

        prefix = "robot:"
        bare_name = robot_name[len(prefix):] if robot_name.startswith(prefix) else robot_name
        module_name = f"robot_descriptions.{bare_name}_mj_description"
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise ValueError(
                f"Robot description not found for {bare_name!r}. "
                f"Expected module {module_name} from the robot_descriptions "
                f"package."
            ) from e

        simulator.spec = mujoco.MjSpec.from_file(str(mod.MJCF_PATH))

        # Remove keyframes — their qpos sizes are for the original model and
        # become invalid once additional bodies (e.g. agents) are added.
        for key in list(simulator.spec.keys):
            simulator.spec.delete(key)

    # ------------------------------------------------------------------
    # Lazy robot discovery
    # ------------------------------------------------------------------

    def _init_robot(self) -> bool:
        """Discover robot joints and EE site after model compilation.

        Returns:
            ``True`` if the robot was successfully initialised.
        """
        if self._robot_initialized:
            return True

        model = self.sim.model
        try:
            self._ee_site_id = model.site(self._ee_site_name).id
        except KeyError:
            return False

        # Walk up from EE site body to find the robot's root body.
        ee_body_id = model.site_bodyid[self._ee_site_id]
        robot_root = ee_body_id
        while model.body_parentid[robot_root] != 0:
            robot_root = model.body_parentid[robot_root]

        # Collect joints that belong to the robot subtree.
        agent_jnt_ids = {self.agent_joint.id, self.pitch_joint.id}
        self._robot_joint_ids = []
        self._robot_dof_ids = []
        self._robot_qpos_addrs = []

        for i in range(model.njnt):
            if i in agent_jnt_ids:
                continue
            body_id = model.jnt_bodyid[i]
            if self._is_descendant_of(body_id, robot_root):
                self._robot_joint_ids.append(i)
                self._robot_dof_ids.append(model.jnt_dofadr[i])
                self._robot_qpos_addrs.append(model.jnt_qposadr[i])

        if not self._robot_joint_ids:
            logger.warning("RobotSurfaceAgent: no robot joints found")
            return False

        self._robot_initialized = True
        logger.info(
            "RobotSurfaceAgent: discovered %d robot joints, EE site '%s'",
            len(self._robot_joint_ids),
            self._ee_site_name,
        )
        return True

    def _is_descendant_of(self, body_id: int, ancestor_id: int) -> bool:
        """Check whether *body_id* is a descendant of *ancestor_id*."""
        while body_id != 0:
            if body_id == ancestor_id:
                return True
            body_id = self.sim.model.body_parentid[body_id]
        return False

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    def _ik_solve(self, target_pos: np.ndarray, target_mat: np.ndarray) -> None:
        """Solve IK for the desired end-effector pose.

        Dispatches to either the simple damped least-squares solver or the
        incremental substep solver based on ``self._ik_method``.

        Args:
            target_pos: Desired end-effector world position (3,).
            target_mat: Desired end-effector world rotation matrix (3, 3).
        """
        solvers = {
            "damped_ls": ik_solve_damped_ls,
            "incremental": ik_solve_incremental,
        }
        solver = solvers.get(self._ik_method)
        if solver is None:
            raise ValueError(
                f"Unknown ik_method {self._ik_method!r}. "
                f"Expected one of {list(solvers)}."
            )
        solver(
            model=self.sim.model,
            data=self.sim.data,
            target_pos=target_pos,
            target_mat=target_mat,
            ee_site_id=self._ee_site_id,
            robot_dof_ids=self._robot_dof_ids,
            robot_joint_ids=self._robot_joint_ids,
            robot_qpos_addrs=self._robot_qpos_addrs,
            **self._ik_args,
        )

    # ------------------------------------------------------------------
    # Agent body ↔ end-effector synchronisation
    # ------------------------------------------------------------------

    def _get_desired_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Combine free-joint rotation and pitch hinge into a single EE target.

        The agent body carries the ``ee_rotation_offset`` so that the camera
        looks outward.  Before sending the target to the IK solver we undo
        the offset to get the physical EE orientation.

        Returns:
            ``(position, rotation_matrix)`` — the desired world-frame EE pose.
        """
        pos = np.array(self.position)
        agent_rot = rotation_from_quat(self.rotation)
        pitch_addr = self.sim.model.jnt_qposadr[self.pitch_joint.id]
        pitch_angle = self.sim.data.qpos[pitch_addr]
        pitch_rot = Rotation.from_euler("x", pitch_angle)
        combined = agent_rot * pitch_rot * self._ee_rotation_offset.inv()
        return pos, combined.as_matrix()

    def _sync_agent_to_ee(
        self,
        sync_position: bool = False,
        sync_rotation: bool = False,
    ) -> None:
        """Optionally sync the virtual camera body to the actual EE pose.

        By default neither position nor rotation is synced from the real
        end-effector.  The body pose is left as set by the action methods
        so the orbit math exactly mirrors :class:`SurfaceAgent` behaviour
        and tiny IK position residuals do not accumulate over time.

        During :meth:`reset` both flags are set to ``True`` so that the
        body starts in the correct pose derived from forward kinematics.
        """
        import mujoco

        mujoco.mj_forward(self.sim.model, self.sim.data)

        if sync_position:
            ee_pos = self.sim.data.site_xpos[self._ee_site_id].copy()
            self.position = tuple(ee_pos)

        if sync_rotation:
            ee_mat = self.sim.data.site_xmat[self._ee_site_id].reshape(3, 3)
            camera_rot = Rotation.from_matrix(ee_mat) * self._ee_rotation_offset
            pitch_addr = self.sim.model.jnt_qposadr[self.pitch_joint.id]
            pitch_angle = self.sim.data.qpos[pitch_addr]
            pitch_rot = Rotation.from_euler("x", pitch_angle)
            free_joint_rot = camera_rot * pitch_rot.inv()
            self.rotation = rotation_as_quat(free_joint_rot)

    def _move_along_local_axis(self, distance: float, axis: int) -> None:
        """Translate ``self.position`` along a local body axis.

        Args:
            distance: Signed distance to move (metres).
            axis: Local axis index (0=x, 1=y, 2=z).
        """
        direction = np.zeros(3)
        direction[axis] = distance
        agent_rot = rotation_from_quat(self.rotation)
        world_delta = agent_rot.apply(direction)
        self.position = tuple(np.array(self.position) + world_delta)

    def _actuate_yaw(self, degrees: float) -> None:
        """Rotate ``self.rotation`` about the agent body's local Y (up) axis.

        Positive ``degrees`` rotates left, mirroring the convention used by
        :class:`DistantAgent` actions.
        """
        delta = Rotation.from_euler("y", np.deg2rad(degrees))
        current = rotation_from_quat(self.rotation)
        self.rotation = rotation_as_quat(current * delta)

    def _actuate_pitch(self, degrees: float) -> None:
        """Increment the pitch hinge joint by ``degrees``.

        Positive ``degrees`` tilts the camera up (matches LookUp semantics).
        """
        pitch_addr = self.sim.model.jnt_qposadr[self.pitch_joint.id]
        self.sim.data.qpos[pitch_addr] += np.deg2rad(degrees)

    def _solve_and_sync(self) -> None:
        """IK-solve for the desired EE pose.

        The body position and rotation are left as set by the action
        methods — only the robot joints are updated.  A forward-kinematics
        pass is run so that subsequent sensor renders reflect the new joint
        configuration.
        """
        target_pos, target_mat = self._get_desired_ee_pose()
        self._ik_solve(target_pos, target_mat)
        mujoco.mj_forward(self.sim.model, self.sim.data)

    # ------------------------------------------------------------------
    # Surface-agent actions (same intent, routed through IK)
    # ------------------------------------------------------------------

    def actuate_move_forward(self, action: MoveForward) -> None:
        self._init_robot()
        self._move_along_local_axis(-action.distance, 2)
        self._solve_and_sync()

    def actuate_turn_right(self, action: TurnRight) -> None:
        self._init_robot()
        self._actuate_yaw(-action.rotation_degrees)
        self._solve_and_sync()

    def actuate_turn_left(self, action: TurnLeft) -> None:
        self._init_robot()
        self._actuate_yaw(action.rotation_degrees)
        self._solve_and_sync()

    def actuate_look_up(self, action: LookUp) -> None:
        self._init_robot()
        self._actuate_pitch(action.rotation_degrees)
        self._solve_and_sync()

    def actuate_look_down(self, action: LookDown) -> None:
        self._init_robot()
        self._actuate_pitch(-action.rotation_degrees)
        self._solve_and_sync()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        if self._init_robot():
            # Reset robot joints to home configuration.
            if self._home_qpos is not None:
                for i, addr in enumerate(self._robot_qpos_addrs):
                    self.sim.data.qpos[addr] = self._home_qpos[i]
            else:
                for addr in self._robot_qpos_addrs:
                    self.sim.data.qpos[addr] = 0.0
            # Zero pitch hinge so the sync starts with a clean decomposition.
            pitch_addr = self.sim.model.jnt_qposadr[self.pitch_joint.id]
            self.sim.data.qpos[pitch_addr] = 0.0
            self._sync_agent_to_ee(sync_position=True, sync_rotation=True)
        else:
            super().reset()

class RobotDistantAgent(RobotAgentBase):
    """A robot-controlled agent for sensing an object from a distance.

    Uses the same look/turn/move actions as :class:`DistantAgent`, routed
    through inverse kinematics so a physical robot arm tracks the camera.
    """

    pass


class RobotSurfaceAgent(RobotAgentBase):
    """A robot-controlled agent for sensing an object up close.

    Extends :class:`RobotAgentBase` with the surface-exploration actions
    (orient and tangential movement) used by :class:`SurfaceAgent`.
    """

    def actuate_move_tangentially(self, action: MoveTangentially) -> None:
        self._init_robot()
        if action.distance == 0.0:
            return
        direction = np.array(action.direction)
        direction_length = np.linalg.norm(direction)
        if np.isclose(direction_length, 0.0):
            return
        direction = direction / direction_length

        rotation = rotation_from_quat(self.rotation)
        direction_rel_world = rotation.apply(direction)
        self.position = np.array(self.position) + direction_rel_world * action.distance
        self._solve_and_sync()

    def actuate_orient_horizontal(self, action: OrientHorizontal) -> None:
        self._init_robot()
        self._move_along_local_axis(-action.left_distance, 0)
        self._actuate_yaw(-action.rotation_degrees)
        self._move_along_local_axis(-action.forward_distance, 2)
        self._solve_and_sync()

    def actuate_orient_vertical(self, action: OrientVertical) -> None:
        self._init_robot()
        self._move_along_local_axis(-action.down_distance, 1)
        self._actuate_pitch(-action.rotation_degrees)
        self._move_along_local_axis(-action.forward_distance, 2)
        self._solve_and_sync()
