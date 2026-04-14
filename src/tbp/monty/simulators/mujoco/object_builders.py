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
import quaternion as qt
import trimesh

from mujoco import MjSpec, MjsBody, mjtGeom, mjtTexture, mjtTextureRole
from PIL import Image

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

def quaternion_to_list(rotation) -> list[float]:
    """Convert a quaternion to a flat list in wxyz order for MuJoCo.

    Accepts the heterogeneous quaternion representations used throughout the
    codebase and normalises them into the ``[w, x, y, z]`` list that MuJoCo's
    spec API expects.

    Leverages :func:`quaternion.as_float_array` for ``quaternion.quaternion``
    objects (the same helper used by the framework's ``BufferEncoder``).

    Args:
        rotation: Quaternion in any supported format —
            ``quaternion.quaternion``, ``numpy.ndarray``, or 4-tuple/list.

    Returns:
        List of quaternion components ``[w, x, y, z]``.

    Raises:
        ValueError: If the format is not recognised.
    """
    if isinstance(rotation, qt.quaternion):
        return qt.as_float_array(rotation).tolist()
    elif isinstance(rotation, np.ndarray):
        return rotation.tolist()
    elif isinstance(rotation, (tuple, list)) and len(rotation) == 4:
        return list(rotation)
    else:
        raise ValueError(f"Unsupported quaternion format: {type(rotation)}")


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


class YCBObjectBuilder(ObjectBuilderBase):
    """Builder for YCB objects stored as pre-converted files.

    Expects each object's files in a directory under ``data_path`` matching
    the object name, containing ``textured.obj``, ``texture_map.png``, and
    ``metadata.json``.
    """

    def __init__(self, ycb_path: Path):
        self.data_path = ycb_path
        self.ycb_path = ycb_path
        self.cache_dir = Path.home() / ".cache" / "tbp" / "mujoco_meshes"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_object(self, name: str | Path) -> bool:
        """Check if the given name refers to a YCB object.

        Args:
            name: Object name to check

        Returns:
            True if this is a YCB object, False otherwise
        """
        if not isinstance(name, str):
            return False
        ycb_mesh_path = self.ycb_path / "meshes" / name
        return ycb_mesh_path.exists()

    def add_to_spec(
        self,
        spec: MjSpec,
        object_count: int,
        position: VectorXYZ,
        rotation: QuaternionWXYZ | qt.quaternion | np.ndarray,
        scale: VectorXYZ,
        name: str | Path,
        **kwargs,
    ) -> None:
        """Add a YCB object mesh to the MuJoCo spec.

        When the source GLB contains a texture, this method also creates the
        corresponding MuJoCo texture, material, and assigns the material to the
        mesh geom so the object renders with its original appearance.

        Args:
            spec: MuJoCo spec to add the object to
            object_count: Unique identifier for the object instance
            name: YCB object name (e.g., '002_master_chef_can')
            position: Initial position of the object
            rotation: Initial orientation of the object
            scale: Scale factors for x, y, z dimensions

        Raises:
            FileNotFoundError: If the mesh file cannot be found
        """
        assert isinstance(name, str), "YCB object name must be a string"
        mesh_file = self._find_mesh_file(name)
        obj_name = f"{name}_{object_count}"
        obj_file, texture_file = self._convert_glb_to_obj(mesh_file, obj_name)

        # Add mesh to spec
        mesh = spec.add_mesh()
        mesh.name = obj_name
        mesh.file = str(obj_file)
        mesh.scale = scale

        # Add texture + material if a texture was extracted
        material_name = None
        if texture_file is not None:
            material_name = self._add_texture_and_material(spec, obj_name, texture_file)

        # Add body for the mesh
        body = spec.worldbody.add_body()
        body.name = f"{obj_name}_body"
        body.pos = position
        body.quat = quaternion_to_list(rotation)

        # Add geom that uses the mesh
        geom = body.add_geom()
        geom.name = obj_name
        geom.type = mjtGeom.mjGEOM_MESH
        geom.meshname = obj_name

        if material_name is not None:
            geom.material = material_name

    @staticmethod
    def _add_texture_and_material(
        spec: MjSpec, obj_name: str, texture_file: Path
    ) -> str:
        """Create a MuJoCo texture and material referencing a PNG file.

        MuJoCo does not auto-parse MTL files, so texture and material must be
        created explicitly in the spec.

        Args:
            spec: MuJoCo spec to add assets to
            obj_name: Base name used to derive unique asset names
            texture_file: Absolute path to the PNG texture image

        Returns:
            The name of the created material (to assign to the geom).
        """
        tex_name = f"{obj_name}_tex"
        mat_name = f"{obj_name}_mat"

        # Read image dimensions
        with Image.open(str(texture_file)) as img:
            width, height = img.size

        # Add texture
        tex = spec.add_texture()
        tex.name = tex_name
        tex.type = mjtTexture.mjTEXTURE_2D
        tex.file = str(texture_file)
        tex.width = width
        tex.height = height
        tex.nchannel = 3

        # Add material that references the texture.
        # NOTE: ``mat.textures`` returns a *copy* of the internal list, so
        # item assignment (``mat.textures[i] = …``) silently fails.  We must
        # assign the whole list at once.  Index 1 corresponds to the "2d"
        # texture role (``mjTEXROLE_RGB``) which is what ``<material
        # texture="…"/>`` maps to in MJCF XML.
        mat = spec.add_material()
        mat.name = mat_name
        tex_slots = [""] * 10
        tex_slots[1] = tex_name
        mat.textures = tex_slots
        mat.texrepeat = [1, 1]

        return mat_name

    def _find_mesh_file(self, ycb_name: str) -> Path:
        """Find the mesh file for a YCB object.

        Prefers ``textured.glb.orig`` because it contains standard PNG textures
        that trimesh can decode.  The regular ``textured.glb`` uses Basis
        Universal compressed textures (``image/x-basis``) which trimesh cannot
        read, so textures would be lost.

        Args:
            ycb_name: YCB object name (e.g., '002_master_chef_can')

        Returns:
            Path to the mesh file

        Raises:
            FileNotFoundError: If no suitable mesh file is found
        """
        mesh_dir = self.ycb_path / "meshes" / ycb_name / "google_16k"

        if not mesh_dir.exists():
            raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")

        # Prefer .glb.orig (has PNG textures that trimesh can decode)
        glb_orig = mesh_dir / "textured.glb.orig"
        if glb_orig.exists():
            return glb_orig

        # Fall back to .glb (Basis Universal textures – no texture support)
        glb_file = mesh_dir / "textured.glb"
        if glb_file.exists():
            return glb_file

        raise FileNotFoundError(f"No mesh file found in {mesh_dir}")

    def _convert_glb_to_obj(
        self, glb_file: Path, obj_name: str
    ) -> tuple[Path, Path | None]:
        """Convert GLB mesh file to OBJ format for MuJoCo.

        When the GLB contains a PNG base-color texture (as in ``.glb.orig``
        files), the texture image is extracted and saved alongside the OBJ so
        that MuJoCo can reference it.

        Args:
            glb_file: Path to the GLB file
            obj_name: Name for the output OBJ file

        Returns:
            Tuple of (obj_path, texture_path).  ``texture_path`` is ``None``
            when the mesh has no extractable texture.
        """
        obj_file = self.cache_dir / f"{obj_name}.obj"
        tex_file = self.cache_dir / f"{obj_name}_texture.png"

        # Skip conversion if OBJ already exists
        if obj_file.exists():
            return obj_file, tex_file if tex_file.exists() else None

        # Load GLB (force file_type='glb' for .glb.orig extension)
        mesh = trimesh.load(str(glb_file), file_type="glb")

        # Handle scene vs single mesh
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [
                    geom
                    for geom in mesh.geometry.values()
                    if isinstance(geom, trimesh.Trimesh)
                ]
            )

        # Export as OBJ (preserves UV coordinates)
        mesh.export(str(obj_file))

        # Extract texture if available
        texture_path = self._extract_texture(mesh, tex_file)

        return obj_file, texture_path

    @staticmethod
    def _extract_texture(mesh: trimesh.Trimesh, output_path: Path) -> Path | None:
        """Extract the base-color texture image from a trimesh object.

        Args:
            mesh: Trimesh with potential texture material
            output_path: Where to save the extracted PNG

        Returns:
            Path to the saved texture, or ``None`` if no texture was found.
        """
        try:
            material = mesh.visual.material
            base_color = getattr(material, "baseColorTexture", None)
            if base_color is None:
                base_color = getattr(material, "image", None)
            if base_color is None:
                return None

            # Convert to RGB (MuJoCo expects 3-channel textures)
            rgb = base_color.convert("RGB")
            rgb.save(str(output_path))
            return output_path
        except Exception:
            return None
