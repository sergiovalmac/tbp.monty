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
from typing import Callable, Sequence

import numpy as np
from mujoco.viewer import launch_passive
from mujoco import (
    MjData,
    MjModel,
    MjsBody,
    MjSpec,
    Renderer,
    mj_forward,
    mjtGeom,
    mjtTexture,
    mjtTextureRole,
)
from typing_extensions import Self, override

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    ObjectInfo,
    SemanticID,
    SimulatedObjectEnvironment,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.simulators.mujoco.object_builders import YCBObjectBuilder
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState
from tbp.monty.frameworks.sensors import Resolution2D
from tbp.monty.math import IDENTITY_QUATERNION, ZERO_VECTOR, QuaternionWXYZ, VectorXYZ
from tbp.monty.simulators.mujoco.agents import Agent
from tbp.monty.simulators.mujoco.objects import (
    ObjectMetadata,
    load_object_metadata,
)

logger = logging.getLogger(__name__)


# Map of names to MuJoCo primitive object types
PRIMITIVE_OBJECTS = {
    "box": mjtGeom.mjGEOM_BOX,
    "capsule": mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mjtGeom.mjGEOM_CYLINDER,
    "ellipsoid": mjtGeom.mjGEOM_ELLIPSOID,
    "sphere": mjtGeom.mjGEOM_SPHERE,
}

DEFAULT_RESOLUTION = Resolution2D((64, 64))


MuJoCoAgentFactory = Callable[["MuJoCoSimulator"], Agent]


class UnknownObjectType(RuntimeError):
    """An unknown object type is requested."""


class MissingObjectModel(RuntimeError):
    """An object type is missing an object model file."""


class MissingObjectTexture(RuntimeError):
    """An object type is missing a texture file."""


class DataPathNotConfigured(RuntimeError):
    """The simulator data_path is not configured and a custom object is requested."""


class ActuateMethodMissing(RuntimeError):
    """The simulator applied an action to an agent that lacks that actuate method."""


class MuJoCoSimulator(SimulatedObjectEnvironment):
    """Simulator implementation for MuJoCo.

    MuJoCo's data model consists of three parts, a spec defining the scene, a
    model representing a scene generated from a spec, and the associated data or state
    of the simulation based on the model.

    To allow programmatic editing of the scene, we're using an MjSpec that we will
    recompile the model and data from whenever an object is added or removed.
    """

    def __init__(
        self,
        agents: Sequence[MuJoCoAgentFactory] | None = None,
        data_path: str | Path | None = None,
        raise_actuate_missing: bool = True,
        show_viewer: bool = False,
    ) -> None:
        """Constructs a MuJoCo simulated environment.

        Args:
            agents: the agents to set up in the environment.
              These are provided by Hydra as partially applied constructors that
              are missing the `simulator` argument.
            data_path: the path to where custom object data should be loaded from.
            raise_actuate_missing: whether to raise an exception when an agent
              does not have an actuate method for an Action.
        """
        if agents is None:
            agents: Sequence[MuJoCoAgentFactory] = []

        self._show_viewer = show_viewer
        self._viewer = None

        self.spec = MjSpec()
        self.model: MjModel = self.spec.compile()
        self.data = MjData(self.model)
        self.data_path = Path(data_path) if data_path else None
        self._ycb_builder = YCBObjectBuilder(self.data_path) if self.data_path else None
        self._raise_actuate_missing = raise_actuate_missing

        self._agent_partials = agents
        self._agents: dict[AgentID, Agent] = {}
        self._create_agents()
        self._loaded_custom_types: set[str] = set()

        # Start with a default resolution in case we don't have agents, e.g. in tests.
        self._render_resolution = DEFAULT_RESOLUTION
        if self._agents:
            self._render_resolution = self._max_sensor_resolution()

        # Track how many objects we add to the environment.
        # Note: We can't use the `model.ngeoms` for this since that will include parts
        # of the agents, especially when we start to add more structure to them.
        self._object_count = 0

        # Maps geom names (stable across recompiles) to the semantic id of the
        # object they belong to. Geoms not in this map (e.g. robot links) are
        # treated as background by `geom_id_to_semantic_lut`.
        self._geom_name_to_semantic: dict[str, int] = {}
        self._known_geom_names: set[str] = set()
        # LUT indexed by ``geom_id + 1`` (so segmentation's -1 background maps
        # to ``lut[0] = 0``). Refreshed after every model recompile.
        self._geom_id_to_semantic_lut: np.ndarray = np.zeros(1, dtype=np.int32)

        self.renderer = None
        self._recompile()

        # Snapshot the names of all geoms present after the initial compile
        # (i.e. robot/agent geoms) so they are treated as background and never
        # reassigned to an object's semantic id.
        self._known_geom_names = {
            self.model.geom(gid).name for gid in range(self.model.ngeom)
        }

    def _recompile(self) -> None:
        """Recompile the MuJoCo model while retaining any state data."""
        # The spec might be new, so reset all the options
        self.spec.option.gravity = (0.0, 0.0, 0.0)
        g = self.spec.visual.global_
        g.offwidth, g.offheight = self._render_resolution
        self.model, self.data = self.spec.recompile(self.model, self.data)
        # The renderer has to be recreated when the model is updated.
        self._create_renderer()
        # Geom IDs may shift across recompiles, so rebuild the semantic LUT.
        self._refresh_geom_semantic_lut()
        # Step the simulation so all objects are in their initial positions.
        mj_forward(self.model, self.data)
        self._reopen_viewer()

    def _refresh_geom_semantic_lut(self) -> None:
        """Rebuild the geom-id -> semantic-id LUT from the current model.

        The LUT is indexed by ``geom_id + 1`` so that the background sentinel
        (``geom_id == -1``) returned by MuJoCo's segmentation rendering maps
        to ``lut[0] == 0``.
        """
        ngeom = self.model.ngeom
        lut = np.zeros(ngeom + 1, dtype=np.int32)
        for gid in range(ngeom):
            name = self.model.geom(gid).name
            sem = self._geom_name_to_semantic.get(name, 0)
            lut[gid + 1] = sem
        self._geom_id_to_semantic_lut = lut

    @property
    def geom_id_to_semantic_lut(self) -> np.ndarray:
        """LUT mapping ``geom_id + 1`` to the semantic id of its parent object."""
        return self._geom_id_to_semantic_lut

    def _reopen_viewer(self) -> None:
        """Reopen the viewer after model recompilation."""
        if not self._show_viewer:
            return
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.close()
        self._viewer = launch_passive(self.model, self.data)
        self._viewer.sync()

    def _sync_viewer(self) -> None:
        """Sync the viewer to reflect the current data state."""
        if not self._show_viewer:
            return
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def _create_renderer(self) -> None:
        """Create a new MuJoCo renderer, closing the existing one if needed."""
        if self.renderer:
            self.renderer.close()
        self.renderer = Renderer(
            width=self._render_resolution[0],
            height=self._render_resolution[1],
            model=self.model,
        )

    def _create_agents(self) -> None:
        self._agents = {}
        for agent_partial in self._agent_partials:
            agent = agent_partial(self)
            self._agents[agent.id] = agent

    def _max_sensor_resolution(self) -> Resolution2D:
        """Determine the maximum resolution of all the sensors.

        We need this to set the off-screen buffer size in MuJoCo to support the
        highest resolution sensor configured.

        Returns:
            max_width, max_height
        """
        max_width = max_height = 0
        for agent in self._agents.values():
            width, height = agent.max_sensor_resolution
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        return Resolution2D((max_width, max_height))

    def remove_all_objects(self) -> None:
        # TODO: is there a better way to do this?
        self.spec = MjSpec()
        self._create_agents()
        self._recompile()
        self._object_count = 0
        self._loaded_custom_types = set()

    @override
    def add_object(
        self,
        name: str,
        position: VectorXYZ = ZERO_VECTOR,
        rotation: QuaternionWXYZ = IDENTITY_QUATERNION,
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,
        **kwargs,
    ) -> ObjectInfo:
        obj_name = f"{name}_{self._object_count}"

        if name in PRIMITIVE_OBJECTS:
            self._add_primitive_object(obj_name, name, position, rotation, scale)
        elif self._ycb_builder is not None and self._ycb_builder.is_object(name):
            self._ycb_builder.add_to_spec(
                self.spec, self._object_count, position, rotation, scale, name,
            )
        else:
            self._add_custom_object(obj_name, name, position, rotation, scale)
        self._object_count += 1

        self._recompile()

        if not semantic_id:
            semantic_id = SemanticID(self._object_count)

        # Register the geoms added by this object so the segmentation renderer
        # can label them with the object's semantic id. Skip unnamed geoms,
        # which are not safe to attribute to a single object.
        current_names = {
            self.model.geom(gid).name for gid in range(self.model.ngeom)
        }
        new_names = {n for n in current_names - self._known_geom_names if n}
        for gname in new_names:
            self._geom_name_to_semantic[gname] = int(semantic_id)
        self._known_geom_names |= new_names
        self._refresh_geom_semantic_lut()

        return ObjectInfo(
            object_id=ObjectID(self._object_count),
            semantic_id=semantic_id,
        )

    def _add_custom_object(
        self,
        obj_name: str,
        object_type: str,
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
        scale: VectorXYZ,
    ):
        """Adds a custom object loaded from the data_path to the scene.

        This assumes that each object's files are stored in a directory in the
        `data_path` matching the shape_type. It should contain the mesh in
        'textured.obj', the texture in 'texture_map.png', as well as a 'metadata.json'
        file with additional information we need to correctly add the object to
        the scene.

        Arguments:
            obj_name: Name for the object in the scene.
            object_type: Type of object to add, determines directory to look in.
            position: Initial position of the object.
            rotation: Initial orientation of the object.
            scale: Initial scale of the object.
        """
        if scale != (1.0, 1.0, 1.0):
            # TODO: In order to support this, we need to update the
            #  object loading code to set the scale on the "mesh" object,
            #  which also means we need to track loaded objects with the
            #  scale included.
            raise NotImplementedError(
                "Custom objects do not currently support "
                "'scale' other than (1.0, 1.0, 1.0)."
            )

        if object_type not in self._loaded_custom_types:
            self._load_custom_object(object_type)

        self.spec.worldbody.add_geom(
            name=obj_name,
            type=mjtGeom.mjGEOM_MESH,
            meshname=f"{object_type}_mesh",
            material=f"{object_type}_mat",
            pos=position,
            quat=rotation,
        )

    def _load_custom_object(self, object_type: str) -> None:
        """Loads a custom object from the data_path into the spec.

        This should only be done once per custom object type.

        Raises:
            DataPathNotConfigured: if data_path is not configured
            UnknownObjectType: When the directory for the object_type is missing.
            MissingObjectTexture: When the texture map is missing.
            MissingObjectModel: When the object is missing.
        """
        if not self.data_path:
            raise DataPathNotConfigured(
                "Cannot load custom objects in simulator, "
                "'data_path' is not configured."
            )
        path = self.data_path / object_type
        texture_path = path / "texture_map.png"
        model_path = path / "textured.obj"

        if not path.exists():
            raise UnknownObjectType(f"Unknown object type: {object_type}")
        if not texture_path.exists():
            raise MissingObjectTexture(
                f"The {object_type} is missing 'texture_map.png'."
            )
        if not model_path.exists():
            raise MissingObjectModel(f"The {object_type} is missing 'textured.obj'.")

        # MuJoCo doesn't seem to be able to load the referenced texture from the
        # 'texture.obj' file directly, so we have to load the texture separately and
        # create a material for it that we can add to the mesh.
        self.spec.add_texture(
            name=f"{object_type}_tex",
            type=mjtTexture.mjTEXTURE_2D,
            file=str(texture_path),
        )
        mat = self.spec.add_material(
            name=f"{object_type}_mat",
        )
        mat.textures[mjtTextureRole.mjTEXROLE_RGB] = f"{object_type}_tex"

        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            metadata = load_object_metadata(metadata_path, object_type)
        else:
            metadata = ObjectMetadata()

        self.spec.add_mesh(
            name=f"{object_type}_mesh",
            file=str(model_path),
            refquat=metadata.refquat,
            refpos=metadata.refpos,
        )

        self._loaded_custom_types.add(object_type)

    def _add_primitive_object(
        self,
        obj_name: str,
        object_type: str,
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
        scale: VectorXYZ,
    ) -> None:
        """Adds a built-in MuJoCo primitive geom to the scene spec.

        Arguments:
            obj_name: Name for the object in the scene.
            object_type: The primitive object type to add.
            position: Initial position of the object.
            rotation: Initial orientation of the object.
            scale: Initial scale of the object.

        Raises:
            UnknownObjectType: When the shape_type is unknown.
        """
        world_body: MjsBody = self.spec.worldbody

        try:
            geom_type = PRIMITIVE_OBJECTS[object_type]
        except KeyError:
            raise UnknownObjectType(
                f"Unknown MuJoCo primitive: {object_type}"
            ) from None

        # TODO: should we encapsulate primitive objects into bodies?
        world_body.add_geom(
            name=obj_name,
            type=geom_type,
            size=scale,
            pos=position,
            quat=rotation,
        )

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

    @override
    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        logger.debug(f"{actions=}")
        for action in actions:
            agent = self._agents[action.agent_id]
            try:
                action.act(agent)  # type: ignore[attr-defined]
            except AttributeError as exc:
                # Only catch missing actuate methods, propagate any other errors
                if exc.name and exc.name.startswith("actuate_"):
                    msg = f"{exc.obj} does not understand '{exc.name}'"
                    if self._raise_actuate_missing:
                        raise ActuateMethodMissing(msg) from None
                    logger.warning(msg)
                    continue
                raise
        mj_forward(self.model, self.data)
        self._sync_viewer()
        return self.observations, self.states

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        for agent in self._agents.values():
            agent.reset()
        mj_forward(self.model, self.data)
        self._sync_viewer()
        return self.observations, self.states

    def close(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.close()
        self._viewer = None
        if self.renderer:
            self.renderer.close()
        self.renderer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
