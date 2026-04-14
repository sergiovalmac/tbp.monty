# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Object builders for MuJoCo simulator.

This module contains builder classes that add objects to MuJoCo scenes:
- PrimitiveObjectBuilder: Geometric primitives (sphere, box, cylinder, etc.)
- DataPathYCBBuilder: YCB objects from a pre-converted data directory
- MJCFObjectBuilder: Objects from MJCF (XML) files
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from mujoco import MjSpec, MjsBody, mjtGeom, mjtTexture, mjtTextureRole

from tbp.monty.math import QuaternionWXYZ, VectorXYZ

VALID_MJCF_EXTENSIONS = {".xml", ".mjcf"}

#: Map primitive shape names to MuJoCo geometry type enums.
PRIMITIVE_GEOM_TYPES: dict[str, mjtGeom] = {
    "box": mjtGeom.mjGEOM_BOX,
    "capsule": mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mjtGeom.mjGEOM_CYLINDER,
    "ellipsoid": mjtGeom.mjGEOM_ELLIPSOID,
    "sphere": mjtGeom.mjGEOM_SPHERE,
}


class ObjectBuilderBase(ABC):
    """Abstract base class for MuJoCo object builders."""

    @abstractmethod
    def add_to_spec(
        self,
        spec: MjSpec,
        object_count: int,
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
        scale: VectorXYZ,
        name: str | Path,
        **kwargs,
    ) -> None:
        """Add an object to the MuJoCo spec.

        Args:
            spec: MuJoCo spec to add the object to.
            object_count: Monotonically increasing index used to generate a
                unique name for the object in the scene.
            position: Initial position of the object.
            rotation: Initial orientation of the object.
            scale: Scale factors for x, y, z dimensions.
            name: Object identifier -- a primitive name, YCB name, or MJCF path.
            **kwargs: Builder-specific keyword arguments.
        """

    @abstractmethod
    def is_object(self, name: str | Path) -> bool:
        """Check whether *name* refers to an object this builder can handle.

        Args:
            name: Object name or path to check.

        Returns:
            ``True`` if this builder can handle the given name.
        """


class PrimitiveObjectBuilder(ObjectBuilderBase):
    """Builder for primitive MuJoCo geometric objects."""

    def add_to_spec(
        self,
        spec: MjSpec,
        object_count: int,
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
        scale: VectorXYZ,
        name: str | Path,
        *,
        rgba: tuple[float, float, float, float] | None = None,
        **kwargs,
    ) -> None:
        assert isinstance(name, str), "Primitive object name must be a string"

        geom_type = PRIMITIVE_GEOM_TYPES.get(name)
        if geom_type is None:
            raise ValueError(
                f"Unknown MuJoCo primitive: {name!r}. "
                f"Valid primitives: {sorted(PRIMITIVE_GEOM_TYPES)}"
            )

        obj_name = f"{name}_{object_count}"
        geom = spec.worldbody.add_geom(
            name=obj_name,
            type=geom_type,
            size=scale,
            pos=position,
            quat=rotation,
        )
        if rgba is not None:
            geom.rgba = rgba

    def is_object(self, name: str | Path) -> bool:
        return isinstance(name, str) and name in PRIMITIVE_GEOM_TYPES


class DataPathYCBBuilder(ObjectBuilderBase):
    """Builder for YCB objects stored as pre-converted files.

    Expects each object's files in a directory under ``data_path`` matching
    the object name, containing ``textured.obj``, ``texture_map.png``, and
    ``metadata.json``.
    """

    def __init__(self, data_path: Path):
        self.data_path = data_path

    def add_to_spec(
        self,
        spec: MjSpec,
        object_count: int,
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
        scale: VectorXYZ,
        name: str | Path,
        **kwargs,
    ) -> None:
        assert isinstance(name, str), "YCB object name must be a string"
        path = self.data_path / name
        if not path.exists():
            raise FileNotFoundError(f"YCB object directory not found: {path}")

        obj_name = f"{name}_{object_count}"

        spec.add_texture(
            name=f"{obj_name}_tex",
            type=mjtTexture.mjTEXTURE_2D,
            file=str(path / "texture_map.png"),
        )
        mat = spec.add_material(name=f"{obj_name}_mat")
        mat.textures[mjtTextureRole.mjTEXROLE_RGB] = f"{obj_name}_tex"

        metadata_path = path / "metadata.json"
        with metadata_path.open() as f:
            metadata = json.load(f)

        spec.add_mesh(
            name=f"{obj_name}_mesh",
            file=str(path / "textured.obj"),
            refquat=metadata["refquat"],
            refpos=metadata["refpos"],
        )
        spec.worldbody.add_geom(
            name=obj_name,
            type=mjtGeom.mjGEOM_MESH,
            meshname=f"{obj_name}_mesh",
            material=f"{obj_name}_mat",
            size=scale,
            pos=position,
            quat=rotation,
        )

    def is_object(self, name: str | Path) -> bool:
        if not isinstance(name, str):
            return False
        return (self.data_path / name).exists()


class MJCFObjectBuilder(ObjectBuilderBase):
    """Builder for adding objects from MJCF XML files."""

    def add_to_spec(
        self,
        spec: MjSpec,
        object_count: int,
        position: VectorXYZ,
        rotation: QuaternionWXYZ,
        scale: VectorXYZ,
        name: str | Path,
        **kwargs,
    ) -> None:
        try:
            temp_spec = MjSpec()
            mjcf_path = Path(name)
            source_dir = mjcf_path.parent
            temp_spec.from_file(str(name))

            obj_name = f"mjcf_{object_count}"

            parent_body = spec.worldbody.add_body()
            parent_body.name = obj_name
            parent_body.pos = position
            parent_body.quat = rotation

            self._copy_assets(spec, temp_spec, obj_name, source_dir)
            self._copy_bodies_recursive(
                source_body=temp_spec.worldbody,
                dest_body=parent_body,
                scale=scale,
                prefix=obj_name,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to add object from MJCF source: {e}"
            ) from e

    @staticmethod
    def _copy_bodies_recursive(
        source_body: MjsBody,
        dest_body: MjsBody,
        scale: VectorXYZ,
        prefix: str,
    ) -> None:
        """Recursively copy bodies and their geoms from source to destination.

        Args:
            source_body: Source body to copy from.
            dest_body: Destination body to copy to.
            scale: Scale factors for positions and sizes.
            prefix: Prefix to add to names to avoid conflicts.
        """
        scale_array = np.array(scale)

        src_child = source_body.first_body()
        while src_child:
            dest_child = dest_body.add_body()

            if hasattr(src_child, "name") and src_child.name:
                dest_child.name = f"{prefix}_{src_child.name}"
            if hasattr(src_child, "pos") and src_child.pos is not None:
                dest_child.pos = tuple(np.array(src_child.pos) * scale_array)
            if hasattr(src_child, "quat") and src_child.quat is not None:
                dest_child.quat = src_child.quat

            src_geom = src_child.first_geom()
            while src_geom:
                dest_geom = dest_child.add_geom()
                if hasattr(src_geom, "name") and src_geom.name:
                    dest_geom.name = f"{prefix}_{src_geom.name}"
                if hasattr(src_geom, "type"):
                    dest_geom.type = src_geom.type
                if hasattr(src_geom, "size") and src_geom.size is not None:
                    dest_geom.size = tuple(np.array(src_geom.size) * scale_array)
                if hasattr(src_geom, "pos") and src_geom.pos is not None:
                    dest_geom.pos = tuple(np.array(src_geom.pos) * scale_array)
                if hasattr(src_geom, "quat") and src_geom.quat is not None:
                    dest_geom.quat = src_geom.quat
                if hasattr(src_geom, "rgba") and src_geom.rgba is not None:
                    dest_geom.rgba = src_geom.rgba
                if hasattr(src_geom, "meshname") and src_geom.meshname:
                    dest_geom.meshname = f"{prefix}_{src_geom.meshname}"

                src_geom = src_child.next_geom(src_geom)

            MJCFObjectBuilder._copy_bodies_recursive(
                src_child, dest_child, scale, prefix
            )
            src_child = source_body.next_body(src_child)

    @staticmethod
    def _copy_assets(
        dest_spec: MjSpec,
        source_spec: MjSpec,
        prefix: str,
        source_dir: Path | None = None,
    ) -> None:
        """Copy asset definitions (meshes, textures, materials) from source spec.

        Args:
            dest_spec: Destination spec to add assets to.
            source_spec: Source spec containing assets.
            prefix: Prefix to add to asset names to avoid conflicts.
            source_dir: Directory of source MJCF file for resolving relative
                mesh paths.
        """
        for idx, src_mesh in enumerate(source_spec.mesh):
            dest_mesh = dest_spec.add_mesh()

            if hasattr(src_mesh, "name") and src_mesh.name:
                dest_mesh.name = f"{prefix}_{src_mesh.name}"
            elif hasattr(src_mesh, "file") and src_mesh.file:
                mesh_filename = Path(src_mesh.file).stem
                dest_mesh.name = f"{prefix}_{mesh_filename}"
            else:
                dest_mesh.name = f"{prefix}_mesh_{idx}"

            if hasattr(src_mesh, "file") and src_mesh.file:
                mesh_file = src_mesh.file
                if source_dir is not None:
                    mesh_path = Path(mesh_file)
                    if not mesh_path.is_absolute():
                        meshdir = getattr(source_spec, "meshdir", None)
                        if meshdir:
                            mesh_path = source_dir / meshdir / mesh_path
                        else:
                            mesh_path = source_dir / mesh_path
                        mesh_path = mesh_path.resolve()
                        mesh_file = str(mesh_path)
                dest_mesh.file = mesh_file
            if hasattr(src_mesh, "scale") and src_mesh.scale is not None:
                dest_mesh.scale = src_mesh.scale

    def is_object(self, name: str | Path) -> bool:
        if not isinstance(name, (Path, str)):
            return False
        path_obj = Path(name) if isinstance(name, str) else name
        return path_obj.exists() and path_obj.suffix.lower() in VALID_MJCF_EXTENSIONS
