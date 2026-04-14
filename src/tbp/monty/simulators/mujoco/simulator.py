# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from mujoco import (
    MjData,
    MjModel,
    MjSpec,
    mj_forward,
)
from typing_extensions import override

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    ObjectInfo,
    SemanticID,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import (
    ProprioceptiveState,
)
from tbp.monty.math import QuaternionWXYZ, VectorXYZ
from tbp.monty.simulators.mujoco import Agent, AgentConfig, Size
from tbp.monty.simulators.mujoco.agents import RobotSurfaceAgent
from tbp.monty.simulators.mujoco.object_builders import (
    YCBObjectBuilder,
    MJCFObjectBuilder,
    ObjectBuilderBase,
    PrimitiveObjectBuilder,
    RobotDescriptionBuilder,
)
from tbp.monty.simulators.simulator import Simulator

if TYPE_CHECKING:
    from os import PathLike


logger = logging.getLogger(__name__)


class UnknownShapeType(RuntimeError):
    """Raised when an unknown shape is requested."""


class MuJoCoSimulator(Simulator):
    """Simulator implementation for MuJoCo.

    MuJoCo's data model consists of three parts, a spec defining the scene, a
    model representing a scene generated from a spec, and the associated data or state
    of the simulation based on the model.

    To allow programmatic editing of the scene, we're using an MjSpec that we will
    recompile the model and data from whenever an object is added or removed.
    """

    def __init__(
        self,
        agent_configs: Sequence[AgentConfig] = (),
        data_path: PathLike | None = None,
        mjcf_file: Path | str | None = None,
        # TODO: remove after adding remaining arguments
        **kwargs,  # noqa: ARG002
    ) -> None:
        self.spec = MjSpec()
        self.model: MjModel = self.spec.compile()
        self.data = MjData(self.model)

        self.data_path = Path(data_path)
        self._agent_configs = agent_configs
        self._agents: dict[AgentID, Agent] = {}
        self._create_agents()

        self._object_count = 0

        self._primitive_builder = PrimitiveObjectBuilder()
        self._data_path_ycb_builder = YCBObjectBuilder(self.data_path)
        self._mjcf_builder = MJCFObjectBuilder()
        self._robot_builder = RobotDescriptionBuilder()

        self._recompile()

    def _max_sensor_resolution(self) -> Size:
        """Determine the maximum resolution of a sensor.

        We need this to set the off-screen buffer size in MuJoCo to support the
        highest resolution sensor configured.

        Returns:
            max_x, max_y
        """
        max_x = 0
        max_y = 0
        for agent_cfg in self._agent_configs:
            for sensor_cfg in agent_cfg["agent_args"]["sensor_configs"].values():
                max_x = max(max_x, sensor_cfg["resolution"][0])
                max_y = max(max_y, sensor_cfg["resolution"][1])
        return max_x, max_y

    def _create_agents(self):
        for agent_config in self._agent_configs:
            agent_type = agent_config["agent_type"]
            agent_args = agent_config["agent_args"]
            agent = agent_type(simulator=self, **agent_args)
            self._agents[agent.id] = agent

    def _recompile(self) -> None:
        """Recompile the MuJoCo model while retaining any state data."""
        # spec.option and spec.visual broken in mujoco 3.2.x — set on compiled model
        self.model, self.data = self.spec.recompile(self.model, self.data)
        self.model.opt.gravity[:] = [0.0, 0.0, 0.0]
        mj_forward(self.model, self.data)
        w, h = self._max_sensor_resolution()
        self.model.vis.global_.offwidth = w
        self.model.vis.global_.offheight = h

    def remove_all_objects(self) -> None:
        self.spec = MjSpec()

        # If any agent requires a robot arm, reload it before agents are
        # created so that the robot bodies exist in the spec.
        robot_name = self._get_required_robot_name()
        if robot_name:
            self._robot_builder.add_to_spec(
                self.spec, 0, (0, 0, 0), (1, 0, 0, 0), (1, 1, 1), robot_name
            )

        self._create_agents()
        self._recompile()
        self._object_count = 0

    def _get_required_robot_name(self) -> str | None:
        """Return the robot name required by any RobotSurfaceAgent, or None."""
        for cfg in self._agent_configs:
            if issubclass(cfg["agent_type"], RobotSurfaceAgent):
                return cfg["agent_args"].get("robot_name", "robot:ur5e")
        return None

    @override
    def add_object(
        self,
        name: str | Path,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,
        **kwargs,
    ) -> ObjectInfo:
        builder = self._get_object_builder(name)
        builder.add_to_spec(
            self.spec, self._object_count, position, rotation, scale, name, **kwargs
        )
        self._object_count += 1

        # Robot builders replace the spec via from_file, wiping any existing
        # bodies (including agents).  Re-create agents so their spec-level
        # joints are valid before recompile.
        if isinstance(builder, RobotDescriptionBuilder) and self._agent_configs:
            self._create_agents()

        self._recompile()

        return ObjectInfo(
            object_id=ObjectID(self._object_count),
            semantic_id=semantic_id,
        )

    def _get_object_builder(self, name: str | Path) -> ObjectBuilderBase:
        """Get the appropriate builder for the given object name.

        Checks builders in order: MJCF file, primitive shape, data-path YCB.

        Args:
            name: Object name or path to identify the builder.

        Returns:
            The matching builder instance.

        Raises:
            UnknownShapeType: If no builder can handle the given name.
        """
        if self._robot_builder.is_object(name):
            return self._robot_builder
        if self._mjcf_builder.is_object(name):
            return self._mjcf_builder
        if self._primitive_builder.is_object(name):
            return self._primitive_builder
        if self._data_path_ycb_builder and self._data_path_ycb_builder.is_object(name):
            return self._data_path_ycb_builder
        raise UnknownShapeType(
            f"No builder found for object: {name!r}. "
            f"Expected a primitive name, MJCF file path, or YCB object in "
            f"{self.data_path}"
        )

    @override
    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        logger.debug(f"{actions=}")
        for action in actions:
            agent = self._agents[action.agent_id]
            try:
                action.act(agent)
            except AttributeError:
                logger.warning(f"{agent} does not understand {action}")
                continue
        mj_forward(self.model, self.data)
        return self.observations, self.states

    @property
    def observations(self) -> Observations:
        obs = Observations()
        for agent in self._agents.values():
            obs[agent.id] = agent.observations

        return obs

    @property
    def states(self) -> ProprioceptiveState:
        states = ProprioceptiveState()
        for agent in self._agents.values():
            states[agent.id] = agent.state
        return states

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        for agent in self._agents.values():
            agent.reset()
        mj_forward(self.model, self.data)
        return self.observations, self.states

    def close(self) -> None:
        pass
