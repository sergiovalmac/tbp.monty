# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""MuJoCo-specific positioning procedures."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import SemanticID
from tbp.monty.frameworks.environments.positioning_procedures import (
    GOOD_VIEW_DISTANCE_DEFAULT,
    GOOD_VIEW_PERCENTAGE_DEFAULT,
    GetGoodView,
    GetGoodViewFactory,
)
from tbp.monty.frameworks.sensors import SensorID

__all__ = [
    "MuJoCoGetGoodView",
    "MuJoCoGetGoodViewFactory",
]


class MuJoCoGetGoodView(GetGoodView):
    """`GetGoodView` variant that tolerates small/off-centre target objects.

    The base class' :meth:`is_on_target_object` only inspects the single central
    pixel. For very small objects (e.g. a ~1.7 cm dice viewed from 35 cm) or
    objects whose mesh origin is offset from the body position, the centre pixel
    may miss the target by a fraction of a pixel even when the camera is well
    aimed. This subclass instead checks a small central region of size
    ``2 * central_region_radius + 1`` pixels per side.
    """

    def __init__(
        self,
        agent_id: AgentID,
        good_view_distance: float,
        good_view_percentage: float,
        multiple_objects_present: bool,
        sensor_id: SensorID,
        target_semantic_id: SemanticID,
        allow_translation: bool = True,
        max_orientation_attempts: int = 1,
        central_region_radius: int = 2,
    ) -> None:
        """Initialise the procedure.

        Args:
            central_region_radius: Half-width (in pixels) of the central window
                used by :meth:`is_on_target_object`. The window is
                ``2 * central_region_radius + 1`` pixels per side. Must be >= 0;
                a value of 0 reproduces the base class' single-pixel check.

        See :class:`GetGoodView` for the remaining arguments.
        """
        super().__init__(
            agent_id=agent_id,
            good_view_distance=good_view_distance,
            good_view_percentage=good_view_percentage,
            multiple_objects_present=multiple_objects_present,
            sensor_id=sensor_id,
            target_semantic_id=target_semantic_id,
            allow_translation=allow_translation,
            max_orientation_attempts=max_orientation_attempts,
        )
        if central_region_radius < 0:
            raise ValueError("central_region_radius must be >= 0")
        self._central_region_radius = int(central_region_radius)

    def is_on_target_object(self, observation: Mapping) -> bool:
        image_shape = observation[self._agent_id][self._sensor_id]["depth"].shape[0:2]
        semantic_3d = observation[self._agent_id][self._sensor_id]["semantic_3d"]
        semantic = semantic_3d[:, 3].reshape(image_shape).astype(int)
        if not self._multiple_objects_present:
            semantic[semantic > 0] = self._target_semantic_id

        h, w = image_shape
        y_mid, x_mid = h // 2, w // 2
        r = self._central_region_radius
        if r == 0:
            return semantic[y_mid, x_mid] == self._target_semantic_id

        y0, y1 = max(0, y_mid - r), min(h, y_mid + r + 1)
        x0, x1 = max(0, x_mid - r), min(w, x_mid + r + 1)
        return bool(np.any(semantic[y0:y1, x0:x1] == self._target_semantic_id))


class MuJoCoGetGoodViewFactory(GetGoodViewFactory):
    """Factory for :class:`MuJoCoGetGoodView`."""

    def __init__(
        self,
        agent_id: AgentID,
        sensor_id: SensorID,
        allow_translation: bool = True,
        good_view_distance: float = GOOD_VIEW_DISTANCE_DEFAULT,
        good_view_percentage: float = GOOD_VIEW_PERCENTAGE_DEFAULT,
        max_orientation_attempts: int = 1,
        multiple_objects_present: bool = False,
        central_region_radius: int = 2,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            sensor_id=sensor_id,
            allow_translation=allow_translation,
            good_view_distance=good_view_distance,
            good_view_percentage=good_view_percentage,
            max_orientation_attempts=max_orientation_attempts,
            multiple_objects_present=multiple_objects_present,
        )
        self._central_region_radius = int(central_region_radius)

    def create(self, target_semantic_id: SemanticID) -> MuJoCoGetGoodView:
        return MuJoCoGetGoodView(
            agent_id=self._agent_id,
            good_view_distance=self._good_view_distance,
            good_view_percentage=self._good_view_percentage,
            multiple_objects_present=self._multiple_objects_present,
            sensor_id=self._sensor_id,
            target_semantic_id=target_semantic_id,
            allow_translation=self._allow_translation,
            max_orientation_attempts=self._max_orientation_attempts,
            central_region_radius=self._central_region_radius,
        )
