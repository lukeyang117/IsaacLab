# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass


@dataclass
class RayCasterData:
    """Data container for the ray-cast sensor."""

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where N is the number of sensors.
    """
    quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame.

    Shape is (N, 4), where N is the number of sensors.
    """
    ray_hits_w: torch.Tensor = None
    """The ray hit positions in the world frame.

    Shape is (N, B, 3), where N is the number of sensors, B is the number of rays
    in the scan pattern per sensor.
    """
    
    semantic_labels: torch.Tensor = None
    """Semantic labels of the hit objects.
    
    Shape is (N, B), where N is the number of sensors, B is the number of rays.
    
    Values:
        - 0: 'terrain' (static meshes from mesh_prim_paths)
        - 1+: dynamic object IDs (from dynamic_env_mesh_prim_paths)
        - -1: no hit (ray missed or exceeded max_distance)
    
    Example:
        If dynamic_env_mesh_prim_paths = [
            "{ENV_REGEX_NS}/Object_0/_03_cracker_box",
            "{ENV_REGEX_NS}/Object_1/_04_sugar_box",
            "{ENV_REGEX_NS}/Object_2/_05_tomato_soup_can"
        ]
        Then:
        - semantic_labels = 0 → terrain
        - semantic_labels = 1 → cracker_box
        - semantic_labels = 2 → sugar_box  
        - semantic_labels = 3 → tomato_soup_can
        - semantic_labels = -1 → no hit
    """
    
    hit_mesh_source: torch.Tensor = None
    """Source of the mesh that was hit.
    
    Shape is (N, B), where N is the number of sensors, B is the number of rays.
    
    Values:
        - 0: hit from combined_mesh (static/global meshes)
        - 1: hit from env_dynamic_mesh (environment-specific dynamic objects)
        - -1: no hit
    """
