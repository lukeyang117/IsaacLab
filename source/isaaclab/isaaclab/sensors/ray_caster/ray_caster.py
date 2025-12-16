# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import re
import torch
import trimesh
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import warp as wp
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.prims import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from ..sensor_base import SensorBase
from .ray_caster_data import RayCasterData

if TYPE_CHECKING:
    from .ray_caster_cfg import RayCasterCfg


class RayCaster(SensorBase):
    """A ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the `warp_meshes` list. The ray-caster then ray-casts against
    these warp meshes using the ray pattern provided in the configuration.

    .. note::
        Currently, only static meshes are supported. Extending the warp mesh to support dynamic meshes
        is a work in progress.
    """

    cfg: RayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: RayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        # check if sensor path is valid
        # note: currently we do not handle environment indices if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Sensor_[1,2]".
        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the ray-caster sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )
        # Initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = RayCasterData()
        # the warp meshes used for raycasting.
        self.meshes: dict[str, wp.Mesh] = {}
        
        # ========================================
        # 第一组：全局网格的动态跟踪系统
        # （注意：名字有误导性，这不是"静态"网格！）
        # ========================================
        # 用途：跟踪 cfg.mesh_prim_paths 中配置的网格
        # 特点：所有环境共享，但支持动态位置更新
        # 应用场景：地形、建筑物、全局移动障碍物
        
        # combined_mesh: 组合后的 Warp 网格（用于动态更新）
        # - 包含所有全局网格的顶点和面
        # - 每帧通过 _update_combined_mesh_efficiently() 更新
        # - 用于第一次光线投射
        self.combined_mesh: wp.Mesh | None = None
        
        # all_mesh_view: XFormPrim 视图（批量获取全局网格的世界位姿）
        # - 跟踪所有全局网格 prim 的变换
        # - 例如：["/World/ground", "/World/obstacle1", "/World/obstacle2"]
        self.all_mesh_view: XFormPrim | None = None
        
        # all_base_points: 全局网格的局部坐标顶点（GPU 张量）
        # - 存储所有顶点在各自局部坐标系下的位置
        # - 每帧通过变换矩阵转换到世界坐标
        self.all_base_points: torch.Tensor | None = None
        
        # vertex_counts_per_instance: 每个全局网格实例的顶点数
        # - 例如：[5000, 3000, 2000] 表示3个网格
        # - 用于 repeat_interleave 的向量化操作
        self.vertex_counts_per_instance: torch.Tensor | None = None
        
        # mesh_instance_indices: 全局网格实例索引 [0, 1, 2, ...]
        # - 用于索引和映射操作
        self.mesh_instance_indices: torch.Tensor | None = None

        # ========================================
        # 第二组：环境特定的动态网格系统
        # （每个环境有独立的物体实例）
        # ========================================
        # 用途：跟踪 cfg.dynamic_env_mesh_prim_paths 中配置的网格
        # 特点：每个环境有独立实例，位置可以不同
        # 应用场景：可抓取物体（箱子、球体）、环境独立障碍物
        
        # all_env_dynamic_mesh_view: XFormPrim 视图（批量获取环境动态物体的世界位姿）
        # - 跟踪所有环境中所有动态物体的父节点变换
        # - 例如：env0_box, env0_sphere, env1_box, env1_sphere, ...
        # - 注意：跟踪的是父节点（不是网格 prim 本身）
        self.all_env_dynamic_mesh_view: XFormPrim | None = None
        
        # env_dynamic_mesh: 环境动态网格的 Warp 网格
        # - 包含所有环境的所有动态物体的顶点和面
        # - 每帧通过 _update_env_dynamic_mesh_efficiently() 更新
        # - 用于第二次光线投射（与 combined_mesh 结果合并）
        self.env_dynamic_mesh: wp.Mesh | None = None
        
        # env_base_points: 环境动态网格的局部坐标顶点（相对于父节点）
        # - 存储在父节点坐标系下的顶点位置
        # - 通过父节点的世界变换计算最终世界坐标
        self.env_base_points: torch.Tensor | None = None
        
        # env_vertex_counts_per_instance: 每个环境动态物体实例的顶点数
        # - 例如：[2000, 2000, 2000, 1500, 1500, 1500] 
        #   表示2种物体 × 3个环境 = 6个实例
        self.env_vertex_counts_per_instance: torch.Tensor | None = None
        
        # env_mesh_instance_indices: 环境动态网格实例索引
        # - 长度 = num_patterns × num_envs
        self.env_mesh_instance_indices: torch.Tensor | None = None
        
        # ========================================
        # 第三组：语义标签系统
        # （用于识别击中物体的类别）
        # ========================================
        # 用途：将面索引映射到语义类别
        # 特点：支持静态网格和动态网格的语义识别
        
        # static_face_counts_per_instance: 静态网格每个实例的面数
        # - 用于从 face_id 推算击中了哪个静态网格实例
        # - 例如：[100, 50] 表示2个静态网格
        self.static_face_counts_per_instance: torch.Tensor | None = None
        
        # env_face_counts_per_instance: 动态网格每个实例的面数
        # - 用于从 face_id 推算击中了哪个动态物体实例
        # - 例如：[200, 200, 200, 150, 150, 150] 表示2种物体×3个环境
        self.env_face_counts_per_instance: torch.Tensor | None = None
        
        # semantic_class_names: 语义类别名称列表
        # - 索引0: 'terrain' (静态网格)
        # - 索引1+: 动态物体名称（从prim路径提取）
        # - 例如：['terrain', 'cracker_box', 'sugar_box', 'tomato_soup_can']
        self.semantic_class_names: list[str] = ['terrain']  # 默认只有terrain
        
        # env_dynamic_mesh_pattern_names: 动态物体模式名称（从路径提取最后一段）
        # - 用于构建语义类别名称
        # - 例如：['_03_cracker_box', '_04_sugar_box', '_05_tomato_soup_can']
        self.env_dynamic_mesh_pattern_names: list[str] = []

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> RayCasterData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
            num_envs_ids = self._view.count
        else:
            num_envs_ids = len(env_ids)
        # resample the drift
        r = torch.empty(num_envs_ids, 3, device=self.device)
        self.drift[env_ids] = r.uniform_(*self.cfg.drift_range)
        # resample the height drift
        r = torch.empty(num_envs_ids, device=self.device)
        self.ray_cast_drift[env_ids, 0] = r.uniform_(*self.cfg.ray_cast_drift_range["x"])
        self.ray_cast_drift[env_ids, 1] = r.uniform_(*self.cfg.ray_cast_drift_range["y"])
        self.ray_cast_drift[env_ids, 2] = r.uniform_(*self.cfg.ray_cast_drift_range["z"])

    """
    Helper Methods for Semantic Labeling.
    """

    def _get_hit_instance_ids(self, face_ids: torch.Tensor, face_counts: torch.Tensor) -> torch.Tensor:
        """根据击中的面索引推算网格实例 ID
        
        原理：
        - 每个网格实例包含若干个三角形面
        - 面索引是连续的：实例0的面 [0-99]，实例1的面 [100-149]，...
        - 通过二分查找确定 face_id 落在哪个实例的范围内
        
        Args:
            face_ids: 击中的面索引 [num_envs, num_rays]，-1 表示未击中
            face_counts: 每个实例的面数量 [num_instances]
            
        Returns:
            instance_ids: 实例索引 [num_envs, num_rays]，-1 表示未击中任何实例
            
        示例:
            face_counts = [100, 50, 200]  # 3个实例
            face_id = 120 → instance_id = 1 (第二个实例)
            face_id = -1  → instance_id = -1 (未击中)
        """
        # 计算每个实例的面索引起始位置
        # 例如：[100, 50, 200] → [0, 100, 150, 350]
        face_cumsum = torch.cat([
            torch.zeros(1, device=face_ids.device, dtype=torch.int32),
            face_counts.cumsum(0)
        ])
        
        # 使用二分查找确定 face_id 属于哪个实例
        # searchsorted 返回插入位置，减1得到实例索引
        # ✅ 修复：确保返回 int32 类型
        instance_ids = (torch.searchsorted(face_cumsum, face_ids, right=False) - 1).to(torch.int32)
        
        # 处理特殊情况
        instance_ids[face_ids < 0] = -1  # 未击中
        max_face_id = face_cumsum[-1] - 1
        instance_ids[face_ids > max_face_id] = -1  # 超出范围
        
        return instance_ids

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # check if the prim at path is an articulated or rigid prim
        # we do this since for physics-based view classes we can access their data directly
        # otherwise we need to use the xform view class which is slower
        found_supported_prim_class = False
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")
        # create view based on the type of prim
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._view = self._physics_sim_view.create_articulation_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        else:
            self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
            found_supported_prim_class = True
            omni.log.warn(f"The prim at path {prim.GetPath().pathString} is not a physics prim! Using XFormPrim.")
        # check if prim view class is found
        if not found_supported_prim_class:
            raise RuntimeError(f"Failed to find a valid prim view class for the prim paths: {self.cfg.prim_path}")

        # load the meshes by parsing the stage
        self._initialize_enhanced_warp_meshes()
        # initialize the ray start and directions
        self._initialize_rays_impl()
        # initialize dynamic environment meshes if configured
        self._initialize_env_dynamic_meshes()
        
        # ✅ 新增：初始化语义标签缓冲区
        self._data.semantic_labels = torch.full(
            (self._view.count, self.num_rays),
            -1,  # 默认值：未击中
            device=self._device,
            dtype=torch.int32
        )
        self._data.hit_mesh_source = torch.full(
            (self._view.count, self.num_rays),
            -1,  # 默认值：未击中
            device=self._device,
            dtype=torch.int32
        )
        
        # 记录语义类别信息
        omni.log.info(f"[RayCaster] Semantic classes initialized: {self.semantic_class_names}")


    def _initialize_warp_meshes(self):
        """初始化 Warp 网格系统 - 加载场景中的静态网格用于光线投射
        
        该方法负责：
        1. 从配置的路径中查找并提取所有几何体（网格、平面、球体等）
        2. 将所有几何体合并成单个 Warp 网格（提高光线投射效率）
        3. 应用世界坐标变换到所有顶点
        4. 存储最终的组合网格供光线投射使用
        
        注意：此方法处理的是"初始"网格加载，不涉及动态更新
        动态更新由 _setup_dynamic_mesh_system() 负责设置
        """
        # ========================================
        # 初始化数据结构
        # ========================================
        # 用于存储所有网格的顶点坐标（世界坐标系）
        combined_points = []
        # 用于存储所有网格的三角形索引（扁平化格式）
        combined_indices = []
        # 顶点偏移量：用于合并多个网格时调整索引
        vertex_offset = 0
        # 成功找到的网格总数计数器
        total_meshes_found = 0
        
        # ========================================
        # 定义支持的几何类型
        # ========================================
        # 支持 USD 中常见的几何 primitive 类型
        # Mesh: 自定义网格; Plane: 平面; Sphere: 球体; Cube: 立方体
        # Cylinder: 圆柱; Capsule: 胶囊; Cone: 圆锥
        supported_geometry_types = ["Mesh", "Plane", "Sphere", "Cube", "Cylinder", "Capsule", "Cone"]
        
        # ========================================
        # 显式路径模式：处理配置中提供的每个网格路径
        # ========================================
        # 记录开始处理的日志信息
        omni.log.info(f"Processing {len(self.cfg.mesh_prim_paths)} explicit mesh paths for ray casting...")
        
        # 遍历配置文件中指定的每个网格路径
        # 例如：["/World/ground", "/World/obstacles"]
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            # 记录当前处理的路径
            omni.log.info(f"Processing mesh path: {mesh_prim_path}")
            
            # ========================================
            # 第一步：验证路径是否存在
            # ========================================
            # 使用 sim_utils 检查路径是否指向有效的 prim
            if not sim_utils.find_first_matching_prim(mesh_prim_path):
                # 路径不存在，记录警告并跳过
                omni.log.warn(f"Mesh prim path does not exist: {mesh_prim_path} - skipping.")
                continue
            
            # ========================================
            # 第二步：查找该路径下的所有支持的几何体
            # ========================================
            # 初始化列表，用于存储找到的所有几何 prim
            all_geometry_prims = []
            # 遍历每种支持的几何类型
            for geom_type in supported_geometry_types:
                # 使用 sim_utils 查找指定类型的所有子 prim
                # lambda 函数：检查 prim 的类型名是否匹配当前几何类型
                prims = sim_utils.get_all_matching_child_prims(
                    mesh_prim_path,  # 搜索的根路径
                    lambda prim, gt=geom_type: prim.GetTypeName() == gt  # 过滤条件
                )
                # 将找到的 prim 添加到列表
                all_geometry_prims.extend(prims)

            # ========================================
            # 第三步：处理直接匹配的情况（路径本身就是几何体）
            # ========================================
            # 如果没有找到子几何体，尝试检查路径本身是否是几何体
            if len(all_geometry_prims) == 0:
                # 尝试直接获取该路径的 prim
                exact_prim = sim_utils.find_first_matching_prim(mesh_prim_path)
                # 检查：prim 存在 且 有效 且 类型在支持列表中
                if exact_prim and exact_prim.IsValid() and exact_prim.GetTypeName() in supported_geometry_types:
                    # 将该 prim 添加到列表（作为唯一元素）
                    all_geometry_prims = [exact_prim]
            
            # ========================================
            # 第四步：处理找到的每个几何体
            # ========================================
            # 记录当前路径下找到的网格数量
            meshes_for_this_path = 0
            # 遍历所有找到的几何 prim
            for geom_prim in all_geometry_prims:
                # 调用提取函数，获取几何数据（顶点和索引）
                # 该函数会应用世界坐标变换
                mesh_data = self._extract_mesh_data_from_prim(geom_prim)
                # 检查提取是否成功
                if mesh_data is not None:
                    # 解包：获取顶点坐标和三角形索引
                    points, indices = mesh_data
                    
                    # ========================================
                    # 调整索引以合并网格
                    # ========================================
                    # 为什么要加偏移量？
                    # 因为多个网格合并后，索引需要指向正确的顶点
                    # 例如：第一个网格有100个顶点（索引0-99）
                    #       第二个网格的索引需要从100开始（100-199）
                    offset_indices = indices + vertex_offset
                    
                    # ========================================
                    # 添加到组合数组
                    # ========================================
                    combined_points.append(points)  # 添加顶点
                    combined_indices.append(offset_indices)  # 添加调整后的索引
                    vertex_offset += len(points)  # 更新偏移量（为下一个网格准备）
                    total_meshes_found += 1  # 增加计数
                    meshes_for_this_path += 1  # 当前路径的网格计数
                    
                    # 记录成功添加的网格信息
                    prim_path_str = geom_prim.GetPath().pathString
                    omni.log.info(f"Added {geom_prim.GetTypeName()}: {prim_path_str} with {len(points)} vertices, {len(indices)} faces, transform applied.")
            
            # 记录当前路径处理完成
            omni.log.info(f"Found {meshes_for_this_path} geometries for path: {mesh_prim_path}")
        
        # ========================================
        # 第五步：创建组合网格（如果找到了网格）
        # ========================================
        if total_meshes_found > 0:
            # 垂直堆叠所有顶点数组：将多个 [N, 3] 数组合并为一个 [总N, 3] 数组
            final_points = np.vstack(combined_points)
            # 连接所有索引数组：将多个索引数组首尾相连
            final_indices = np.concatenate(combined_indices)
                        
            # ========================================
            # 创建 Warp 网格对象
            # ========================================
            # convert_to_warp_mesh: 将 numpy 数组转换为 Warp 的 GPU 加速网格格式
            # Warp 网格支持高效的 GPU 光线投射
            wp_mesh = convert_to_warp_mesh(final_points, final_indices, device=self.device)
            
            # ========================================
            # 存储网格到字典
            # ========================================
            # 使用第一个配置路径作为键（约定俗成）
            combined_mesh_key = self.cfg.mesh_prim_paths[0]
            self.meshes[combined_mesh_key] = wp_mesh
            
            # 记录成功信息
            omni.log.info(f"Successfully combined {total_meshes_found} meshes into single mesh for ray casting.")
            omni.log.info(f"Combined mesh has {len(final_points)} vertices and {len(final_indices)} faces.")
        else:
            # ========================================
            # 后备方案：创建默认地面（如果没找到任何网格）
            # ========================================
            omni.log.warn("No meshes found for ray-casting! Creating default ground plane.")
            # 创建一个非常大的平面作为默认地面（2e6 x 2e6 米）
            plane_mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
            # 转换为 Warp 网格
            wp_mesh = convert_to_warp_mesh(plane_mesh.vertices, plane_mesh.faces, device=self.device)
            # 存储为默认地面
            self.meshes["default_ground"] = wp_mesh

    def _initialize_enhanced_warp_meshes(self):
        """增强的网格初始化方法 - 支持动态网格跟踪
        
        该方法是网格初始化的入口点，分两个阶段：
        1. 调用原始的 _initialize_warp_meshes() 加载和合并静态网格
        2. 调用 _setup_dynamic_mesh_system() 设置动态跟踪系统
        
        为什么需要两步？
        - 第一步：加载网格数据并应用初始变换（生成静态 Warp 网格）
        - 第二步：为这些网格建立动态跟踪机制（支持实时位置更新）
        
        这种设计允许：
        - 向后兼容：不需要动态更新时，只执行第一步
        - 性能优化：只有需要时才启用动态跟踪
        """
        # ========================================
        # 第一步：调用原始方法加载静态网格
        # ========================================
        # 该方法会：
        # 1. 从配置路径查找所有几何体
        # 2. 提取顶点和索引数据
        # 3. 应用世界坐标变换
        # 4. 合并成单个 Warp 网格
        # 5. 存储到 self.meshes 字典
        self._initialize_warp_meshes()
        
        # ========================================
        # 第二步：设置动态网格跟踪系统（如果有网格）
        # ========================================
        # 检查是否成功加载了网格
        if len(self.meshes) > 0:
            # 为已加载的网格建立动态跟踪系统
            # 该方法会：
            # 1. 重新提取网格的局部坐标顶点（不含世界变换）
            # 2. 创建 XFormPrim 视图用于批量获取变换
            # 3. 设置 GPU 张量用于高效更新
            # 4. 创建独立的 combined_mesh 用于动态更新
            self._setup_dynamic_mesh_system()

    def _resolve_env_regex_ns(self) -> str:
        """Resolve environment regex namespace from sensor prim path.

        Returns the substring ending with '/env_.*' if present, otherwise '/World/envs/env_.*'.
        """
        try:
            result = re.search(r"(.*/envs/env_\\\.\*)", self.cfg.prim_path)
            if result:
                return result.group(1)
        except Exception:
            pass
        return "/World/envs/env_.*"

    def _initialize_env_dynamic_meshes(self):
        """Initialize the env dynamic mesh system using parent XFormPrims for transforms.

        New logic (requested):
        1. Use the configured `dynamic_env_mesh_prim_paths` (each points to a mesh prim) and resolve them for `env_0` only.
        2. For each such mesh prim (env_0 instance), obtain its parent XFormPrim path. The parent will be used for world pose updates.
        3. Replace the `env_0` portion with the regex namespace (e.g. `env_.*`) to build pattern parent paths and create a single XFormPrim view.
        4. Extract geometry ONCE per pattern from the env_0 mesh prim, transform vertices into the parent frame (apply relative transform mesh->parent),
           then replicate that geometry across all environments while building a combined warp mesh.
        5. Maintain view ordering assumption: pattern0 [env0..envN], pattern1 [env0..envN], ... so the existing efficient update code works unchanged.
        """
        omni.log.info("[_initialize_env_dynamic_meshes] ===== 开始初始化动态环境网格 =====")
        try:
            # 快速退出检查：如果配置对象没有 dynamic_env_mesh_prim_paths 属性，直接返回
            if not hasattr(self.cfg, "dynamic_env_mesh_prim_paths"):
                omni.log.info("[_initialize_env_dynamic_meshes] 配置中没有 dynamic_env_mesh_prim_paths 属性，跳过")
                return
            # 快速退出检查：如果 dynamic_env_mesh_prim_paths 列表为空，直接返回
            if not self.cfg.dynamic_env_mesh_prim_paths:
                omni.log.info("[_initialize_env_dynamic_meshes] dynamic_env_mesh_prim_paths 列表为空，跳过")
                return
            
            omni.log.info(f"[_initialize_env_dynamic_meshes] 找到 {len(self.cfg.dynamic_env_mesh_prim_paths)} 个动态网格路径配置")

            # 导入 USD 几何相关的模块
            from pxr import UsdGeom

            # 解析环境正则表达式命名空间，例如：/World/envs/env_.*
            env_regex_ns = self._resolve_env_regex_ns()  
            # 从正则表达式命名空间派生具体的 env_0 命名空间，用于采样代表性网格
            # 处理 'env_.*' 和转义变体 'env_\.*' 两种情况
            # 使用正则替换将 env_\.\* 替换为 env_0
            env0_ns = re.sub(r"env_\\\.\*", "env_0", env_regex_ns)
            # 使用字符串替换将 env_.* 替换为 env_0
            env0_ns = env0_ns.replace("env_.*", "env_0")

            # 初始化三个列表用于存储从 env_0 提取的模式数据
            parent_pattern_paths: list[str] = []  # 存储带正则表达式的父节点路径
            pattern_points: list[np.ndarray] = []  # 存储在父节点坐标系下的几何顶点（每个模式一个数组）
            pattern_indices: list[np.ndarray] = []  # 存储扁平化的三角形索引（每个模式一个数组）
            resolved_env0_mesh_paths: list[str] = []  # ✅ 新增：存储 env_0 的网格路径（用于提取语义类别名称）

            # 辅助函数：将网格 prim 的几何数据转换到其父节点坐标系
            def _mesh_vertices_in_parent_frame(mesh_prim) -> tuple[np.ndarray, np.ndarray] | None:
                # 从网格 prim 提取网格数据（本地坐标系）
                mesh_data = self._extract_mesh_data_from_prim_dynamic(mesh_prim)
                # 如果提取失败，返回 None
                if mesh_data is None:
                    return None
                # 解包：获取网格本地坐标系下的顶点和索引
                points, indices = mesh_data  # points in mesh local frame
                try:
                    # 将网格 prim 转换为可变换对象
                    mesh_xf = UsdGeom.Xformable(mesh_prim)
                    # 获取网格的父节点 prim
                    parent_prim = mesh_prim.GetParent()
                    # 将父节点 prim 转换为可变换对象
                    parent_xf = UsdGeom.Xformable(parent_prim)
                    # 计算网格从本地到世界坐标系的 4x4 变换矩阵
                    mesh_world = np.array(mesh_xf.ComputeLocalToWorldTransform(0.0))  # 4x4
                    # 计算父节点从本地到世界坐标系的 4x4 变换矩阵
                    parent_world = np.array(parent_xf.ComputeLocalToWorldTransform(0.0))
                    # 计算父节点世界变换矩阵的逆矩阵
                    parent_world_inv = np.linalg.inv(parent_world)
                    # 计算从父节点到网格的相对变换矩阵：parent_world_inv @ mesh_world
                    rel = parent_world_inv @ mesh_world  # parent->mesh
                    # 提取相对变换矩阵的旋转部分（3x3）
                    rot = rel[:3, :3]
                    # 提取相对变换矩阵的平移部分（3x1）
                    trans = rel[:3, 3]
                    # 应用相对变换到网格本地顶点，将它们表达在父节点坐标系下
                    # 公式：points_parent = points @ rot^T + trans
                    points_parent = points @ rot.T + trans
                    # 返回转换后的顶点（父坐标系）和索引，都转换为 float32 和 int32 类型
                    return points_parent.astype(np.float32), indices.astype(np.int32)
                except Exception as e:
                    # 如果相对变换失败，记录警告信息
                    omni.log.warn(f"Failed relative transform for mesh {mesh_prim.GetPath()}: {e}")
                    # 返回原始顶点和索引（未变换），转换为 float32 和 int32 类型
                    return points.astype(np.float32), indices.astype(np.int32)

            # 步骤 1 & 2：为每个动态路径解析 env_0 实例，获取网格 prim 和其父节点
            # 直接访问 stage 以确保可靠地获取 prim
            stage = omni.usd.get_context().get_stage()
            # 定义支持的几何类型集合
            supported_geom_types = {"Mesh", "Sphere", "Cube", "Cylinder", "Capsule", "Cone", "Plane"}

            # 遍历配置中的每个动态环境网格路径模式
            for raw_path in self.cfg.dynamic_env_mesh_prim_paths:
                # 从模式构建 env_0 的具体路径
                try:
                    # 尝试使用 format 方法替换 {ENV_REGEX_NS} 占位符
                    mesh_path_env0 = raw_path.format(ENV_REGEX_NS=env0_ns)
                except Exception:
                    # 如果 format 失败，使用 replace 方法替换正则表达式命名空间
                    mesh_path_env0 = raw_path.replace(env_regex_ns, env0_ns)

                # 从 stage 获取指定路径的网格 prim
                mesh_prim = stage.GetPrimAtPath(mesh_path_env0)
                # 检查网格 prim 是否存在且有效
                if not mesh_prim or not mesh_prim.IsValid():
                    # 如果不存在或无效，记录警告并跳过此模式
                    omni.log.warn(f"Env0 mesh prim path not found: {mesh_path_env0} (skipping pattern)")
                    continue

                # 获取几何类型名称
                geom_type = mesh_prim.GetTypeName()
                # 检查几何类型是否在支持的类型集合中
                if geom_type not in supported_geom_types:
                    # 如果不支持，记录警告并跳过此模式
                    omni.log.warn(
                        f"Env0 mesh prim at {mesh_path_env0} has unsupported type '{geom_type}' (skipping pattern)"
                    )
                    continue

                # 获取网格 prim 的父节点
                parent_prim = mesh_prim.GetParent()
                # 检查父节点是否存在且有效
                if parent_prim is None or not parent_prim.IsValid():
                    # 如果父节点无效，记录警告并跳过此模式
                    omni.log.warn(f"Parent prim invalid for mesh {mesh_path_env0} (skipping pattern)")
                    continue

                # 获取父节点的路径字符串（env_0 版本）
                parent_env0_path = parent_prim.GetPath().pathString
                # 将父节点路径中的 env_0 替换为正则表达式，生成模式路径
                parent_pattern_path = parent_env0_path.replace(env0_ns, env_regex_ns)
                # 将父节点模式路径添加到列表中
                parent_pattern_paths.append(parent_pattern_path)
                
                # ✅ 新增：保存 env_0 的网格路径用于语义类别提取
                resolved_env0_mesh_paths.append(mesh_path_env0)

                # 调用辅助函数，将网格顶点转换到父节点坐标系
                mesh_parent_frame = _mesh_vertices_in_parent_frame(mesh_prim)  # 返回点和面
                # 检查是否成功提取几何数据
                if mesh_parent_frame is None:
                    # 如果提取失败，记录警告
                    omni.log.warn(f"Failed to extract geometry (parent frame) for {mesh_path_env0}")
                    # 移除刚才添加的父节点路径（因为几何提取失败）
                    parent_pattern_paths.pop()
                    # ✅ 新增：同时移除对应的 env0 网格路径
                    resolved_env0_mesh_paths.pop()
                    # 跳过此模式
                    continue
                # 解包：获取父坐标系下的顶点和索引
                pts_parent, idx_parent = mesh_parent_frame
                # 将顶点添加到模式顶点列表
                pattern_points.append(pts_parent)
                # 将索引添加到模式索引列表
                pattern_indices.append(idx_parent)

            # 检查是否成功解析了任何有效的父节点模式路径
            if not parent_pattern_paths:
                # 如果没有，记录警告并返回
                omni.log.warn("No valid dynamic env mesh parent patterns resolved.")
                return

            # 步骤 5：基于父节点模式路径创建 XFormPrim 视图（自动扩展所有环境实例）
            self.all_env_dynamic_mesh_view = XFormPrim(parent_pattern_paths, reset_xform_properties=False)
            # 获取 XFormPrim 视图扩展后的所有具体路径列表
            prim_paths_expanded = list(self.all_env_dynamic_mesh_view.prim_paths)
            # 检查扩展后的路径列表是否为空
            if len(prim_paths_expanded) == 0:
                # 如果为空，记录警告
                omni.log.warn("Parent pattern XFormPrim expansion yielded no prim paths.")
                # 重置视图为 None
                self.all_env_dynamic_mesh_view = None
                # 返回
                return

            # 计算模式数量
            num_patterns = len(parent_pattern_paths)
            # 检查扩展后的路径数量是否能被模式数量整除
            if len(prim_paths_expanded) % num_patterns != 0:
                # 如果不能整除，说明扩展结果不一致，记录警告
                omni.log.warn(
                    "Expanded parent prim count not divisible by number of patterns. Aborting env dynamic mesh setup."
                )
                # 重置视图为 None
                self.all_env_dynamic_mesh_view = None
                # 返回
                return
            # 计算环境数量 = 扩展后的路径总数 / 模式数量
            num_envs = len(prim_paths_expanded) // num_patterns

            # 跨环境复制每个模式的几何数据；维护排序假设：pattern0[env0..envN], pattern1[env0..envN], ...
            # 初始化用于组合所有实例的列表
            combined_points: list[np.ndarray] = []  # 存储所有实例的顶点
            combined_indices: list[np.ndarray] = []  # 存储所有实例的索引
            vertex_counts: list[int] = []  # 存储每个实例的顶点数量
            face_counts: list[int] = []  # ✅ 新增：存储每个实例的面数量（用于语义识别）
            vertex_offset = 0  # 顶点偏移量，用于索引调整
            total_instances = 0  # 总实例数计数器

            # 外层循环：遍历每个模式
            for p in range(num_patterns):
                # 获取当前模式的基础顶点数据
                base_points = pattern_points[p]
                # 获取当前模式的基础索引数据
                base_indices = pattern_indices[p]
                # ✅ 新增：计算当前模式的面数
                face_count = len(base_indices) // 3
                
                # 内层循环：为每个环境复制当前模式的几何数据
                for _env in range(num_envs):
                    # 计算调整后的索引（加上当前的顶点偏移量）
                    offset_indices = base_indices + vertex_offset
                    # 将基础顶点添加到组合列表（所有环境共享相同的几何拓扑）
                    combined_points.append(base_points)
                    # 将调整后的索引添加到组合列表
                    combined_indices.append(offset_indices)
                    # 记录当前实例的顶点数量
                    vertex_counts.append(len(base_points))
                    # ✅ 新增：记录当前实例的面数
                    face_counts.append(face_count)
                    # 更新顶点偏移量（为下一个实例准备）
                    vertex_offset += len(base_points)
                    # 增加总实例计数
                    total_instances += 1

            # 检查是否成功组装了任何实例
            if total_instances == 0:
                # 如果没有实例，记录警告
                omni.log.warn("After replication no env dynamic mesh instances were assembled.")
                # 重置视图为 None
                self.all_env_dynamic_mesh_view = None
                # 返回
                return

            # 垂直堆叠所有顶点数组，形成最终的顶点数组
            final_points = np.vstack(combined_points)
            # 连接所有索引数组，形成最终的索引数组
            final_indices = np.concatenate(combined_indices)

            # 构建 warp 网格并缓存张量以供快速更新
            # 将最终的顶点和索引转换为 warp 网格对象
            self.env_dynamic_mesh = convert_to_warp_mesh(final_points, final_indices, device=self.device)
            # 将最终顶点转换为 PyTorch 张量，存储在指定设备上（用于后续高效更新）
            self.env_base_points = torch.tensor(final_points, device=self.device, dtype=torch.float32)
            # 将每个实例的顶点数量转换为 PyTorch 张量（用于向量化操作）
            self.env_vertex_counts_per_instance = torch.tensor(vertex_counts, device=self.device, dtype=torch.int32)
            # 创建网格实例索引张量（0 到 total_instances-1）
            self.env_mesh_instance_indices = torch.arange(total_instances, device=self.device, dtype=torch.int32)
            
            # ✅ 新增：存储环境动态网格的面数信息（用于语义识别）
            self.env_face_counts_per_instance = torch.tensor(
                face_counts,
                device=self.device,
                dtype=torch.int32
            )
            
            # ✅ 新增：构建语义类别名称列表
            # 从每个模式的路径中提取最后一段作为类别名称
            self.env_dynamic_mesh_pattern_names = []
            for resolved_path in resolved_env0_mesh_paths:
                # 提取路径最后一段（例如："/World/envs/env_0/Object_0/_03_cracker_box" -> "_03_cracker_box"）
                mesh_name = resolved_path.split("/")[-1]
                # 清理名称（去掉下划线前缀）
                clean_name = mesh_name.lstrip("_")
                self.env_dynamic_mesh_pattern_names.append(clean_name)
            
            # 更新全局语义类别名称列表
            # semantic_class_names[0] = 'terrain' (已在__init__中初始化)
            # semantic_class_names[1+] = 动态物体名称
            self.semantic_class_names = ['terrain'] + self.env_dynamic_mesh_pattern_names

            # 记录初始化成功的信息日志，包含详细统计数据
            omni.log.info(
                "Initialized env dynamic mesh (parent-based): patterns=%d, envs=%d, instances=%d, vertices=%d, faces=%d" % (
                    num_patterns,      # 模式数量
                    num_envs,          # 环境数量
                    total_instances,   # 总实例数
                    len(final_points), # 总顶点数
                    len(final_indices) // 3,  # 总面数（索引数/3）
                )
            )
        except Exception as e:
            # 捕获所有异常，记录错误日志
            omni.log.error(f"Failed to initialize env dynamic meshes (parent-based): {e}")
            # 重置所有动态网格相关的成员变量为 None（清理状态）
            self.all_env_dynamic_mesh_view = None
            self.env_dynamic_mesh = None
            self.env_base_points = None
            self.env_vertex_counts_per_instance = None
            self.env_mesh_instance_indices = None
    
    def _setup_dynamic_mesh_system(self):
        """设置动态网格跟踪系统 - 用于高效更新网格位置
        
        该方法的目标：
        为已加载的静态网格建立动态更新机制，使网格可以实时跟踪物体的运动。
        
        核心思想：
        1. 存储网格的"局部坐标"顶点（不含世界变换）
        2. 创建 XFormPrim 视图批量获取物体的实时位姿
        3. 每帧通过向量化操作快速计算世界坐标顶点
        4. 更新 Warp 网格的 BVH 加速结构
        
        性能优势：
        - 避免每帧重新从 USD 提取几何数据（慢）
        - 使用 GPU 向量化操作代替 CPU 循环（快100倍）
        - 使用 BVH refit 而非 rebuild（快10倍）
        
        应用场景：
        - 静态网格需要应用初始变换
        - 全局动态障碍物需要实时更新位置
        """
        # ========================================
        # 初始化数据结构
        # ========================================
        # 存储所有网格 prim 的完整路径（用于创建 XFormPrim 视图）
        all_mesh_prim_paths = []
        # 存储所有网格的局部坐标顶点
        combined_points = []
        # 存储所有网格的三角形索引
        combined_indices = []
        # 存储每个网格实例的顶点数量（用于向量化更新）
        vertex_counts = []
        # ✅ 新增：存储每个网格实例的面数量（用于语义识别）
        face_counts = []
        # 顶点偏移量：合并网格时调整索引
        vertex_offset = 0
        # 成功处理的网格总数
        total_meshes_found = 0
        
        # ========================================
        # 定义支持的几何类型（与静态加载相同）
        # ========================================
        supported_geometry_types = ["Mesh", "Plane", "Sphere", "Cube", "Cylinder", "Capsule", "Cone"]
        
        # 记录开始设置
        omni.log.info("Setting up dynamic mesh system for ray caster...")
        
        # ========================================
        # 遍历配置的网格路径，重新提取局部坐标几何数据
        # ========================================
        # 为什么要重新提取？
        # - _initialize_warp_meshes() 提取的是"世界坐标"顶点（含初始变换）
        # - 动态更新需要"局部坐标"顶点（不含变换，每帧重新应用）
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            # ========================================
            # 第一步：验证路径存在性
            # ========================================
            # 检查路径是否有效，无效则跳过
            if not sim_utils.find_first_matching_prim(mesh_prim_path):
                continue
            
            # ========================================
            # 第二步：查找所有支持的几何体
            # ========================================
            # 初始化几何体列表
            all_geometry_prims = []
            # 遍历每种几何类型，查找匹配的子 prim
            for geom_type in supported_geometry_types:
                prims = sim_utils.get_all_matching_child_prims(
                    mesh_prim_path, 
                    lambda prim, gt=geom_type: prim.GetTypeName() == gt
                )
                all_geometry_prims.extend(prims)

            # ========================================
            # 第三步：处理直接匹配（路径本身是几何体）
            # ========================================
            if len(all_geometry_prims) == 0:
                # 尝试直接获取该路径的 prim
                exact_prim = sim_utils.find_first_matching_prim(mesh_prim_path)
                # 检查是否是支持的几何类型
                if exact_prim and exact_prim.IsValid() and exact_prim.GetTypeName() in supported_geometry_types:
                    all_geometry_prims = [exact_prim]
            
            # ========================================
            # 第四步：处理每个找到的几何体
            # ========================================
            for geom_prim in all_geometry_prims:
                # 关键：使用 _dynamic 版本提取"局部坐标"数据
                # _extract_mesh_data_from_prim_dynamic() 不应用世界变换
                # 与 _extract_mesh_data_from_prim() 的区别：
                # - _from_prim: 返回世界坐标顶点（用于静态网格）
                # - _from_prim_dynamic: 返回局部坐标顶点（用于动态更新）
                mesh_data = self._extract_mesh_data_from_prim_dynamic(geom_prim)
                # 检查提取是否成功
                if mesh_data is not None:
                    # 解包：获取局部坐标顶点和索引
                    points, indices = mesh_data
                    
                    # ========================================
                    # 调整索引以合并多个网格
                    # ========================================
                    # 加上偏移量，确保索引指向正确的顶点位置
                    offset_indices = indices + vertex_offset
                    
                    # ========================================
                    # 添加到组合数组
                    # ========================================
                    combined_points.append(points)  # 局部坐标顶点
                    combined_indices.append(offset_indices)  # 调整后的索引
                    vertex_counts.append(len(points))  # 记录顶点数（重要！）
                    # ✅ 新增：记录面数
                    face_counts.append(len(indices) // 3)  # 每3个索引构成1个三角形面
                    vertex_offset += len(points)  # 更新偏移量
                    total_meshes_found += 1  # 增加计数
                    
                    # ========================================
                    # 存储 prim 路径（用于创建 XFormPrim 视图）
                    # ========================================
                    # 存储路径用于批量查询世界位姿（详见 document.md 中的 XFormPrim 文档）
                    all_mesh_prim_paths.append(geom_prim.GetPath().pathString)
        
        # ========================================
        # 第五步：创建动态网格系统（如果找到了网格）
        # ========================================
        if total_meshes_found > 0:
            try:
                # ========================================
                # 步骤 5.1：合并所有局部坐标数据
                # ========================================
                # 垂直堆叠顶点：[N1+N2+...+Nn, 3]
                final_points = np.vstack(combined_points)
                # 连接索引：[M1+M2+...+Mn]
                final_indices = np.concatenate(combined_indices)
                
                # ========================================
                # 步骤 5.2：创建 Warp 网格（用于动态更新）
                # ========================================
                # 注意：这是一个独立的网格对象，存储在 self.combined_mesh
                # 与 self.meshes[...] 中的静态网格分开
                # 为什么需要两个？
                # - self.meshes: 包含初始世界变换的静态网格
                # - self.combined_mesh: 可动态更新的网格（每帧重新计算顶点）
                self.combined_mesh = convert_to_warp_mesh(final_points, final_indices, device=self.device)
                
                # ========================================
                # 步骤 5.3：创建 XFormPrim 视图（批量获取变换）
                # ========================================
                # 创建 XFormPrim 视图用于实时查询所有网格的世界位姿
                # 详细说明请参见 document.md 中的 XFormPrim 文档
                self.all_mesh_view = XFormPrim(all_mesh_prim_paths, reset_xform_properties=False)
                
                # ========================================
                # 步骤 5.4：设置 GPU 张量用于高效更新
                # ========================================
                # 将局部坐标顶点转换为 PyTorch 张量（GPU）
                # 这些是"基础"顶点，每帧通过变换矩阵转换到世界坐标
                self.all_base_points = torch.tensor(final_points, device=self.device, dtype=torch.float32)
                
                # 将顶点计数转换为张量（用于 repeat_interleave 操作）
                # 例如：[1000, 500, 800] 表示3个网格分别有1000、500、800个顶点
                self.vertex_counts_per_instance = torch.tensor(vertex_counts, device=self.device, dtype=torch.int32)
                
                # 创建网格实例索引：[0, 1, 2, ..., N-1]
                # 用于索引和映射操作
                self.mesh_instance_indices = torch.arange(total_meshes_found, device=self.device, dtype=torch.int32)
                
                # ✅ 新增：存储静态网格的面数信息（用于语义识别）
                self.static_face_counts_per_instance = torch.tensor(
                    face_counts,
                    device=self.device,
                    dtype=torch.int32
                )
                
                # ========================================
                # 记录成功信息
                # ========================================
                omni.log.info(f"Successfully set up dynamic mesh system:")
                omni.log.info(f"  - Tracking {total_meshes_found} mesh instances")  # 跟踪的网格数
                omni.log.info(f"  - Total vertices: {len(final_points)}")  # 总顶点数
                omni.log.info(f"  - Total faces: {len(final_indices) // 3}")  # 总面数（索引数/3）
                
            except Exception as e:
                # ========================================
                # 错误处理：设置失败时清理状态
                # ========================================
                omni.log.warn(f"Failed to setup dynamic mesh system: {str(e)}")
                # 将所有动态网格相关变量重置为 None
                # 这样传感器会回退到使用静态网格（self.meshes）
                self.combined_mesh = None
                self.all_mesh_view = None
                self.all_base_points = None
                self.vertex_counts_per_instance = None
                self.mesh_instance_indices = None

    def _initialize_rays_impl(self):
        # compute ray stars and directions
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(self.ray_directions)
        # apply offset transformation to the rays
        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos
        # repeat the rays for each sensor
        self.ray_starts = self.ray_starts.repeat(self._view.count, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self._view.count, 1, 1)
        # prepare drift
        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        self.ray_cast_drift = torch.zeros(self._view.count, 3, device=self.device)
        # fill the data buffer
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """填充传感器数据缓冲区 - 执行光线投射并更新传感器读数
        
        该方法是传感器更新的核心，主要步骤：
        1. 获取传感器的世界位姿（位置和旋转）
        2. 更新动态网格（如果存在）
        3. 根据配置的对齐方式计算光线起点和方向
        4. 执行光线投射获取碰撞点
        5. 应用漂移/噪声模拟真实传感器
        
        Args:
            env_ids: 需要更新的环境索引序列
        """
        # ========================================
        # 第一步：获取传感器的世界位姿
        # ========================================
        # 根据不同的视图类型（XForm/关节体/刚体）获取位姿
        if isinstance(self._view, XFormPrim):
            # XFormPrim：直接获取世界位姿（位置+四元数旋转）
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            # ArticulationView（关节体）：从根变换中提取位姿
            # get_root_transforms 返回 [pos(3), quat(4)] 共7个值
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            # PhysX使用xyzw格式，转换为wxyz格式（Isaac Sim标准）
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            # RigidBodyView（刚体）：从刚体变换中提取位姿
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            # 同样需要转换四元数格式
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            # 不支持的视图类型，抛出异常
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        
        # 克隆张量以避免修改原始数据（后续操作是只读的）
        pos_w = pos_w.clone()  # 位置克隆 [num_envs, 3]
        quat_w = quat_w.clone()  # 四元数克隆 [num_envs, 4]
        
        # 应用传感器位置漂移（模拟传感器安装误差或振动）
        # drift是预先采样的随机漂移 [num_envs, 3]
        pos_w += self.drift[env_ids]
        
        # 将计算好的位姿存储到数据缓冲区中
        self._data.pos_w[env_ids] = pos_w  # 存储世界坐标系下的位置
        self._data.quat_w[env_ids] = quat_w  # 存储世界坐标系下的旋转

        # ========================================
        # 第二步：更新动态网格（如果存在）
        # ========================================
        # 动态网格支持：在每次更新时重新计算网格顶点的世界坐标
        if self.combined_mesh is not None:
            # 更新全局动态网格（例如：场景中移动的障碍物）
            self._update_combined_mesh_efficiently()
        if self.env_dynamic_mesh is not None:
            # 更新环境特定的动态网格（例如：每个环境中的动态物体）
            self._update_env_dynamic_mesh_efficiently()

        # ========================================
        # 第三步：根据光线对齐模式计算光线起点和方向
        # ========================================
        # 支持三种对齐模式：world（世界系）、yaw（偏航系）、base（基座系）
        if self.cfg.ray_alignment == "world":
            # 世界对齐模式：光线方向固定在世界坐标系中（不随传感器旋转）
            # 用途：垂直向下的高度传感器
            
            # 应用水平漂移（XY平面）到传感器位置
            pos_w[:, 0:2] += self.ray_cast_drift[env_ids, 0:2]
            
            # 光线起点 = 预定义起点 + 传感器世界位置
            ray_starts_w = self.ray_starts[env_ids]  # [num_envs, num_rays, 3]
            ray_starts_w += pos_w.unsqueeze(1)  # 广播加法：[num_envs, 1, 3] + [num_envs, num_rays, 3]
            
            # 光线方向保持不变（世界坐标系方向）
            ray_directions_w = self.ray_directions[env_ids]  # [num_envs, num_rays, 3]
            
        elif self.cfg.ray_alignment == "yaw" or self.cfg.attach_yaw_only:
            # 偏航对齐模式：只考虑传感器的偏航角（绕Z轴旋转），忽略俯仰和横滚
            # 用途：水平扫描的2D激光雷达
            
            # 兼容性检查：attach_yaw_only是旧参数名
            if self.cfg.attach_yaw_only:
                self.cfg.ray_alignment = "yaw"
                omni.log.warn(
                    "The `attach_yaw_only` property will be deprecated in a future release. Please use"
                    " `ray_alignment='yaw'` instead."
                )

            # 应用水平漂移，但要先转到传感器坐标系（考虑偏航角）
            # quat_apply_yaw只应用偏航旋转，忽略其他轴
            pos_w[:, 0:2] += quat_apply_yaw(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            
            # 光线起点：应用偏航旋转后加上传感器位置
            # repeat(1, num_rays)将四元数复制num_rays次以匹配光线数量
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            
            # 光线方向保持在水平面内（不随俯仰/横滚旋转）
            ray_directions_w = self.ray_directions[env_ids]
            
        elif self.cfg.ray_alignment == "base":
            # 基座对齐模式：完全考虑传感器的所有旋转（6DOF）
            # 用途：3D激光雷达（如Velodyne、Livox）
            
            # 应用水平漂移到传感器局部坐标系
            # quat_apply应用完整的3D旋转
            pos_w[:, 0:2] += quat_apply(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            
            # 光线起点：应用完整旋转后加上传感器位置
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            
            # 光线方向：也要应用完整的旋转
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        else:
            # 不支持的对齐模式
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}.")

        # ========================================
        # 第四步：执行双重光线投射（静态网格 + 动态网格）
        # ========================================
        # 双重光线投射策略：分别对静态和动态网格投射，然后取最近的碰撞点
        
        # 选择要使用的静态网格
        if self.combined_mesh is not None:
            # 如果有组合的动态网格，使用它（支持网格运动）
            mesh_to_use = self.combined_mesh
        else:
            # 否则使用原始的静态网格
            mesh_to_use = self.meshes[self.cfg.mesh_prim_paths[0]]

        # 第一次光线投射：对静态/主要网格投射
        # ✅ 修改：不再忽略 face_ids，用于语义识别
        ray_hits_1, dist1, _, face_ids_1 = raycast_mesh(
            ray_starts_w,  # 光线起点 [num_envs, num_rays, 3]
            ray_directions_w,  # 光线方向 [num_envs, num_rays, 3]
            max_dist=self.cfg.max_distance,  # 最大检测距离
            mesh=mesh_to_use,  # 要投射的网格
            return_distance=True,  # 返回距离信息
            return_face_id=True,  # ✅ 返回面索引用于语义识别
        )

        # 第二次光线投射：对环境动态网格投射（如果存在）
        if self.env_dynamic_mesh is not None:
            # 对动态物体（如箱子、球体）执行光线投射
            # ✅ 修改：保留 face_ids_2 用于语义识别
            ray_hits_2, dist2, _, face_ids_2 = raycast_mesh(
                ray_starts_w,
                ray_directions_w,
                max_dist=self.cfg.max_distance,
                mesh=self.env_dynamic_mesh,  # 动态网格
                return_distance=True,
                return_face_id=True,  # ✅ 返回面索引用于语义识别
            )
            # 合并结果：取两次投射中的最小距离（最近的碰撞）
            final_dist = torch.minimum(dist1, dist2)
            
            # ✅ 新增：确定击中来自哪个网格（0=static, 1=dynamic）
            hit_source = (dist2 < dist1).to(torch.int32)  # [num_envs, num_rays] - 修复dtype
            
        else:
            # 没有动态网格，直接使用静态网格的结果
            final_dist = dist1
            # ✅ 新增：所有击中都来自静态网格
            hit_source = torch.zeros_like(dist1, dtype=torch.int32)  # 全为0
            face_ids_2 = torch.full(face_ids_1.shape, -1, device=self._device, dtype=torch.int32)  # 占位符

        # ========================================
        # ✅ 新增步骤：计算语义标签
        # ========================================
        # 初始化语义标签为 -1（未击中）
        semantic_labels = torch.full(
            (len(env_ids), self.num_rays),
            -1,
            device=self._device,
            dtype=torch.int32
        )
        
        # 方案1：如果击中静态网格（hit_source == 0）→ 语义标签 = 0 (terrain)
        # 方案2：如果击中动态网格（hit_source == 1）→ 根据 face_id 推算物体类别
        
        # 处理静态网格击中（terrain）
        static_hits = (hit_source == 0) & (face_ids_1 >= 0)
        semantic_labels[static_hits] = 0  # terrain
        
        # 处理动态网格击中（动态物体）
        if self.env_dynamic_mesh is not None and self.env_face_counts_per_instance is not None:
            # 只处理击中动态网格的光线
            dynamic_hits_mask = (hit_source == 1) & (face_ids_2 >= 0)
            
            if dynamic_hits_mask.any():
                # 从 face_id 推算实例 ID
                # 实例布局：[pattern0_env0, pattern0_env1, ..., pattern1_env0, pattern1_env1, ...]
                instance_ids = self._get_hit_instance_ids(
                    face_ids_2,
                    self.env_face_counts_per_instance
                )
                
                # ✅ 新增：环境隔离检查
                # 从实例 ID 恢复击中物体所属的环境 ID
                num_envs_total = self._view.count
                hit_env_ids = (instance_ids % num_envs_total).to(torch.int32)  # 确保 int32 类型
                
                # 构建当前环境 ID 矩阵（每行对应一个环境的所有光线）
                current_env_ids = torch.arange(len(env_ids), device=self._device, dtype=torch.int32)
                current_env_ids = env_ids[current_env_ids].unsqueeze(-1).expand(-1, self.num_rays)
                
                # 判断是否同环境击中（True=同环境，False=跨环境）
                same_env_mask = (hit_env_ids == current_env_ids)
                
                # 将实例 ID 映射到模式 ID（物体类别）
                # 例如：3个环境，2个模式
                # instance_id: 0,1,2 → pattern 0; instance_id: 3,4,5 → pattern 1
                pattern_ids = (instance_ids // num_envs_total).to(torch.int32)  # 确保 int32 类型
                
                # 语义标签 = pattern_id + 1（因为0是terrain）
                # 例如：pattern0 → label 1, pattern1 → label 2, ...
                semantic_labels_dynamic = (pattern_ids + 1).to(torch.int32)  # 确保 int32 类型
                
                # 只更新同环境击中的动态网格光线（跨环境击中保持-1，稍后处理）
                same_env_dynamic_hits = dynamic_hits_mask & same_env_mask
                semantic_labels[same_env_dynamic_hits] = semantic_labels_dynamic[same_env_dynamic_hits]
                
                # ✅ 新增：处理跨环境击中 → 替换为静态网格结果
                cross_env_mask = dynamic_hits_mask & (~same_env_mask)
                if cross_env_mask.any():
                    # 将跨环境击中的语义标签设为 0 (terrain)
                    semantic_labels[cross_env_mask] = 0
                    # 将击中来源标记为静态网格
                    hit_source[cross_env_mask] = 0
        
        # 存储语义标签到数据缓冲区
        self._data.semantic_labels[env_ids] = semantic_labels
        
        # 存储击中来源信息
        # 将未击中的标记为 -1
        hit_source[final_dist >= self.cfg.max_distance] = -1
        hit_source[torch.isinf(final_dist)] = -1
        self._data.hit_mesh_source[env_ids] = hit_source

        # ========================================
        # 第五步：计算最终碰撞点并应用垂直漂移
        # ========================================
        # 根据距离计算实际的3D碰撞点坐标
        # 公式：hit_point = ray_start + distance * ray_direction
        # 对于未命中的光线，final_dist为inf，结果也会是inf
        
        # ✅ 修改：使用动态选择的距离和击中点
        # 默认使用合并后的最近距离
        final_dist_to_use = final_dist.clone()
        
        # 如果存在动态网格，需要根据 hit_source 选择正确的距离
        if self.env_dynamic_mesh is not None:
            # 对于击中静态网格的光线（hit_source == 0），使用 dist1
            static_mask = (hit_source == 0)
            final_dist_to_use[static_mask] = dist1[static_mask]
            
            # 对于击中动态网格的光线（hit_source == 1），使用 dist2
            # 但如果是跨环境击中，已在上面将 hit_source 改为 0，所以会使用 dist1
            dynamic_mask = (hit_source == 1)
            final_dist_to_use[dynamic_mask] = dist2[dynamic_mask]
        
        # 计算最终的3D碰撞点坐标
        final_hits = ray_starts_w + final_dist_to_use.unsqueeze(-1) * ray_directions_w
        # unsqueeze(-1)将 [num_envs, num_rays] 扩展为 [num_envs, num_rays, 1]
        # 以便与方向向量 [num_envs, num_rays, 3] 相乘
        
        # 存储碰撞点到数据缓冲区
        self._data.ray_hits_w[env_ids] = final_hits  # [num_envs, num_rays, 3]

        # 应用垂直（Z轴）漂移到碰撞点（模拟高度测量误差）
        # unsqueeze(-1)将 [num_envs] 扩展为 [num_envs, 1] 以便广播到所有光线
        self._data.ray_hits_w[env_ids, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.ray_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # remove possible inf values
        viz_points = self._data.ray_hits_w.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        # show ray hit positions
        self.ray_visualizer.visualize(viz_points)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None
        # Invalidate dynamic mesh view as well
        self.all_mesh_view = None
        # Invalidate environment dynamic mesh view as well
        self.all_env_dynamic_mesh_view = None

    def _create_trimesh_from_usd_primitive(self, geom_prim, geom_type):
        """Create a trimesh object from USD primitive parameters.
        
        Args:
            geom_prim: USD geometry primitive 
            geom_type: Type of the primitive (Sphere, Cube, Cylinder, Capsule, Cone)
            
        Returns:
            trimesh.Trimesh object or None if creation failed
        """
        try:
            import trimesh
            from pxr import UsdGeom
            
            if geom_type == "Sphere":
                # Get sphere parameters
                sphere_geom = UsdGeom.Sphere(geom_prim)
                radius_attr = sphere_geom.GetRadiusAttr()
                radius = radius_attr.Get() if radius_attr else 1.0
                
                # Create trimesh sphere
                return trimesh.creation.uv_sphere(radius=radius)
                
            elif geom_type == "Cube":
                # Get cube parameters (size attribute)
                cube_geom = UsdGeom.Cube(geom_prim)
                size_attr = cube_geom.GetSizeAttr()
                size = size_attr.Get() if size_attr else 2.0  # USD Cube default size is 2.0
                
                # Create trimesh box
                return trimesh.creation.box(extents=[size, size, size])
                
            elif geom_type == "Cylinder":
                # Get cylinder parameters
                cylinder_geom = UsdGeom.Cylinder(geom_prim)
                radius_attr = cylinder_geom.GetRadiusAttr()
                height_attr = cylinder_geom.GetHeightAttr()
                axis_attr = cylinder_geom.GetAxisAttr()
                
                radius = radius_attr.Get() if radius_attr else 1.0
                height = height_attr.Get() if height_attr else 2.0
                axis = axis_attr.Get() if axis_attr else "Z"
                
                # Create transform for axis alignment
                transform = None
                if axis == "X":
                    transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
                elif axis == "Y":
                    transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
                
                # Create trimesh cylinder
                return trimesh.creation.cylinder(radius=radius, height=height, transform=transform)
                
            elif geom_type == "Capsule":
                # Get capsule parameters
                capsule_geom = UsdGeom.Capsule(geom_prim)
                radius_attr = capsule_geom.GetRadiusAttr()
                height_attr = capsule_geom.GetHeightAttr()
                axis_attr = capsule_geom.GetAxisAttr()
                
                radius = radius_attr.Get() if radius_attr else 1.0
                height = height_attr.Get() if height_attr else 2.0
                axis = axis_attr.Get() if axis_attr else "Z"
                
                # Create transform for axis alignment
                transform = None
                if axis == "X":
                    transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
                elif axis == "Y":
                    transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
                
                # Create trimesh capsule
                return trimesh.creation.capsule(radius=radius, height=height, transform=transform)
                
            elif geom_type == "Cone":
                # Get cone parameters
                cone_geom = UsdGeom.Cone(geom_prim)
                radius_attr = cone_geom.GetRadiusAttr()
                height_attr = cone_geom.GetHeightAttr()
                axis_attr = cone_geom.GetAxisAttr()
                
                radius = radius_attr.Get() if radius_attr else 1.0
                height = height_attr.Get() if height_attr else 2.0
                axis = axis_attr.Get() if axis_attr else "Z"
                
                # Create transform for axis alignment
                transform = None
                if axis == "X":
                    transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
                elif axis == "Y":
                    transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
                
                # Create trimesh cone
                return trimesh.creation.cone(radius=radius, height=height, transform=transform)
                
            else:
                omni.log.warn(f"Unsupported primitive type for trimesh creation: {geom_type}")
                return None
                
        except Exception as e:
            omni.log.warn(f"Failed to create trimesh for {geom_type}: {str(e)}")
            return None

    def _extract_mesh_data_from_prim(self, geom_prim):
        """Extract mesh data from any supported USD geometry primitive.
        
        Args:
            geom_prim: USD geometry primitive (Mesh, Plane, Sphere, Cube, Cylinder, Capsule, Cone)
            
        Returns:
            tuple: (points, indices) as numpy arrays, or None if extraction failed
        """
        from pxr import UsdGeom
        try:
            geom_type = geom_prim.GetTypeName()

            if geom_type == "Plane":
                # Handle Plane using make_plane utility (keeps existing logic)
                plane_mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                
                # Apply world transformation to plane
                try:
                    from pxr import UsdGeom
                    xform = UsdGeom.Xformable(geom_prim)
                    if xform:
                        transform_matrix = np.array(xform.ComputeLocalToWorldTransform(0.0)).T
                    else:
                        transform_matrix = np.eye(4)
                except:
                    transform_matrix = np.eye(4)
                    
                points = np.matmul(plane_mesh.vertices, transform_matrix[:3, :3].T)
                points += transform_matrix[:3, 3]
                indices = plane_mesh.faces.flatten()
                
                return points, indices
                
            else:
                # Handle all other geometry types (Mesh, Sphere, Cube, Cylinder, Capsule, Cone)
                from pxr import UsdGeom

                if geom_type == "Mesh":
                    # Direct mesh access
                    mesh_geom = UsdGeom.Mesh(geom_prim)
                    points_attr = mesh_geom.GetPointsAttr()
                    face_indices_attr = mesh_geom.GetFaceVertexIndicesAttr()
                    face_counts_attr = mesh_geom.GetFaceVertexCountsAttr()
                    
                    if not (points_attr and face_indices_attr and face_counts_attr):
                        omni.log.warn(f"Could not find mesh attributes for {geom_type}: {geom_prim.GetPath()}")
                        return None
                        
                    # Get the actual data
                    points_data = points_attr.Get()
                    faces_data = face_indices_attr.Get()
                    face_counts_data = face_counts_attr.Get()
                    
                    if points_data is None or faces_data is None or face_counts_data is None:
                        omni.log.warn(f"Mesh attribute data is None for {geom_type}: {geom_prim.GetPath()}")
                        return None
                    
                    points = list(points_data)
                    points = [np.ravel(x) for x in points]
                    points = np.array(points)
                    
                    if len(points) == 0:
                        omni.log.warn(f"Empty points array for {geom_type}: {geom_prim.GetPath()}")
                        return None
                        
                    faces = list(faces_data)
                    face_counts = list(face_counts_data)
                    
                    if len(faces) == 0 or len(face_counts) == 0:
                        omni.log.warn(f"Empty faces/face_counts for {geom_type}: {geom_prim.GetPath()}")
                        return None
                    
                    # Check if triangulation is needed
                    if not all(count == 3 for count in face_counts):
                        omni.log.info(f"Triangulating {geom_type} {geom_prim.GetPath()} - found faces with {set(face_counts)} vertices")
                        faces = self._triangulate_faces_from_list(faces, face_counts)
                    
                    # Convert to proper triangle format
                    triangulated_indices = np.array(faces)
                    
                    # Apply world transformation
                    try:
                        xform = UsdGeom.Xformable(geom_prim)
                        if xform:
                            transform_matrix = np.array(xform.ComputeLocalToWorldTransform(0.0)).T
                        else:
                            transform_matrix = np.eye(4)
                    except:
                        transform_matrix = np.eye(4)
                        
                    points = np.matmul(points, transform_matrix[:3, :3].T)
                    points += transform_matrix[:3, 3]
                    
                    return points, triangulated_indices
                    
                else:
                    # Handle primitive shapes (Sphere, Cube, Cylinder, Capsule, Cone) using trimesh
                    trimesh_mesh = self._create_trimesh_from_usd_primitive(geom_prim, geom_type)
                    
                    if trimesh_mesh is None:
                        omni.log.warn(f"Failed to create trimesh for {geom_type}: {geom_prim.GetPath()}")
                        return None
                    
                    # Get scale from USD prim attribute
                    try:
                        path = geom_prim.GetPath().pathString
                        prim = prim_utils.get_prim_at_path(path)
                        scale_attr = prim.GetAttribute("xformOp:scale")
                        if scale_attr and scale_attr.IsValid() and scale_attr.HasValue():
                            scale = tuple(scale_attr.Get())
                            # Apply scale to trimesh object if scale is not uniform [1,1,1]
                            if not all(abs(s - 1.0) < 1e-6 for s in scale):
                                trimesh_mesh.apply_scale(scale)
                    except Exception as e:
                        omni.log.warn(f"Could not get scale for {geom_prim.GetPath()}: {e}")
                    
                    # Apply world transformation (rotation and translation)
                    try:
                        xform = UsdGeom.Xformable(geom_prim)
                        if xform:
                            transform_matrix = np.array(xform.ComputeLocalToWorldTransform(0.0)).T
                        else:
                            transform_matrix = np.eye(4)
                    except:
                        transform_matrix = np.eye(4)
                        
                    # Transform mesh vertices to world coordinates
                    points = np.matmul(trimesh_mesh.vertices, transform_matrix[:3, :3].T)
                    points += transform_matrix[:3, 3]
                    indices = trimesh_mesh.faces.flatten()
                    
                    return points, indices
                
        except Exception as e:
            omni.log.warn(f"Failed to extract mesh data from {geom_prim.GetTypeName()} {geom_prim.GetPath()}: {str(e)}")
            return None

    def _triangulate_faces_from_list(self, faces: list, face_counts: list) -> list:
        """Convert polygonal faces to triangles using list format.
        
        Args:
            faces: Flattened list of face vertex indices
            face_counts: List containing number of vertices per face
            
        Returns:
            Triangulated face indices as flat list
        """
        triangulated_faces = []
        face_idx = 0
        
        for count in face_counts:
            if count == 3:
                # Already a triangle
                triangulated_faces.extend(faces[face_idx:face_idx + 3])
            elif count == 4:
                # Quad to two triangles
                v0, v1, v2, v3 = faces[face_idx:face_idx + 4]
                triangulated_faces.extend([v0, v1, v2])  # First triangle
                triangulated_faces.extend([v0, v2, v3])  # Second triangle
            else:
                # General polygon triangulation (fan triangulation)
                v0 = faces[face_idx]
                for i in range(1, count - 1):
                    v1 = faces[face_idx + i]
                    v2 = faces[face_idx + i + 1]
                    triangulated_faces.extend([v0, v1, v2])
            
            face_idx += count
        
        return triangulated_faces

    def _extract_mesh_data_from_prim_dynamic(self, geom_prim):
        """Extract mesh data from USD geometry primitive for dynamic tracking.
        
        从 USD 几何 primitive 中提取网格数据用于动态跟踪。
        该方法提取本地坐标系下的几何数据（不包含世界变换），稍后通过父节点的位姿实时更新。
        
        Args:
            geom_prim: USD geometry primitive (USD 几何 primitive 对象)
            
        Returns:
            tuple: (points, indices) as numpy arrays in local coordinates, or None if extraction failed
                   (本地坐标系下的顶点和索引数组元组，失败返回 None)
        """
        # 导入 USD 几何模块，用于访问几何属性
        from pxr import UsdGeom
        try:
            # 获取几何体的类型名称（如 "Mesh", "Sphere", "Cube" 等）
            geom_type = geom_prim.GetTypeName()

            # ============================================
            # 分支 1: 处理平面（Plane）类型
            # ============================================
            if geom_type == "Plane":
                # 在本地坐标系创建一个简单的平面网格
                # 尺寸: 2e6 x 2e6 米（非常大的地面），高度 0，中心在原点
                plane_mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                
                # 尝试应用 USD prim 上的缩放属性（如果存在）
                try:
                    # 获取 prim 的完整路径字符串
                    path = geom_prim.GetPath().pathString
                    # 通过路径重新获取 prim 对象（用于访问属性）
                    prim = prim_utils.get_prim_at_path(path)
                    # 获取 "xformOp:scale" 属性（USD 中的缩放变换操作）
                    scale_attr = prim.GetAttribute("xformOp:scale")
                    # 检查缩放属性是否存在、有效且有值
                    if scale_attr and scale_attr.IsValid() and scale_attr.HasValue():
                        # 获取缩放值并转换为元组 (sx, sy, sz)
                        scale = tuple(scale_attr.Get())
                        # 只在缩放不是默认的 [1,1,1] 时应用（避免不必要的计算）
                        # 检查所有缩放分量是否都接近 1.0（误差小于 1e-6）
                        if not all(abs(s - 1.0) < 1e-6 for s in scale):
                            # 将缩放应用到平面顶点：逐元素相乘
                            vertices_scaled = plane_mesh.vertices * np.array(scale)
                            # 返回缩放后的顶点和扁平化的面索引
                            return vertices_scaled, plane_mesh.faces.flatten()
                except Exception as e:
                    # 如果获取缩放失败，记录警告但继续（使用未缩放的网格）
                    omni.log.warn(f"Could not get scale for {geom_prim.GetPath()}: {e}")
                
                # 返回原始平面顶点（本地坐标）和扁平化的面索引
                return plane_mesh.vertices, plane_mesh.faces.flatten()
            
            # ============================================
            # 分支 2: 处理网格（Mesh）类型
            # ============================================
            elif geom_type == "Mesh":
                # 将 USD Prim 转换为 Mesh 几何对象，提供网格特定的 API
                mesh_geom = UsdGeom.Mesh(geom_prim)
                # 获取顶点位置属性对象
                points_attr = mesh_geom.GetPointsAttr()
                # 获取面顶点索引属性对象（定义每个面使用哪些顶点）
                face_indices_attr = mesh_geom.GetFaceVertexIndicesAttr()
                # 获取面顶点计数属性对象（定义每个面有多少个顶点）
                face_counts_attr = mesh_geom.GetFaceVertexCountsAttr()
                
                # 验证所有必需的属性都存在，如果缺少任何一个则返回 None
                if not (points_attr and face_indices_attr and face_counts_attr):
                    return None
                
                # 从属性对象获取实际数据值
                points_data = points_attr.Get()  # 顶点数据（VtArray of Vec3f）
                faces_data = face_indices_attr.Get()  # 面索引数据（扁平化的索引数组）
                face_counts_data = face_counts_attr.Get()  # 每个面的顶点数（例如 [3,3,4] 表示2个三角形和1个四边形）
                
                # 再次验证数据值不为 None（即使属性存在，值也可能为 None）
                if points_data is None or faces_data is None or face_counts_data is None:
                    return None
                
                # 将 USD 的 Vec3f 数组转换为标准的 numpy 数组
                # 列表推导式提取每个 3D 点的 x, y, z 分量
                points = np.array([[x[0], x[1], x[2]] for x in points_data])
                # 将面索引转换为 Python 列表（便于后续处理）
                faces = list(faces_data)
                # 将面顶点计数转换为 Python 列表
                face_counts = list(face_counts_data)
                
                # 检查数据是否为空（防止空网格）
                if len(points) == 0 or len(faces) == 0:
                    return None
                
                # 检查是否需要三角化（光线投射需要三角形网格）
                # 如果不是所有面都有 3 个顶点（即存在四边形或多边形）
                if not all(count == 3 for count in face_counts):
                    # 调用三角化函数将多边形分解为三角形
                    faces = self._triangulate_faces_from_list_dynamic(faces, face_counts)
                
                # 尝试应用 USD prim 上的缩放属性
                try:
                    # 获取 prim 的完整路径字符串
                    path = geom_prim.GetPath().pathString
                    # 通过路径重新获取 prim 对象
                    prim = prim_utils.get_prim_at_path(path)
                    # 获取 "xformOp:scale" 属性
                    scale_attr = prim.GetAttribute("xformOp:scale")
                    # 检查缩放属性是否存在、有效且有值
                    if scale_attr and scale_attr.IsValid() and scale_attr.HasValue():
                        # 获取缩放值元组 (sx, sy, sz)
                        scale = tuple(scale_attr.Get())
                        # 只在缩放不是默认的 [1,1,1] 时应用
                        if not all(abs(s - 1.0) < 1e-6 for s in scale):
                            # 将缩放应用到网格顶点（逐元素相乘）
                            points_scaled = points * np.array(scale)
                            # 返回缩放后的顶点和面索引（转换为 numpy 数组）
                            return points_scaled, np.array(faces)
                except Exception as e:
                    # 如果获取缩放失败，记录警告但继续
                    omni.log.warn(f"Could not get scale for {geom_prim.GetPath()}: {e}")
                
                # 返回原始顶点（本地坐标）和面索引数组
                return points, np.array(faces)
            
            # ============================================
            # 分支 3: 处理其他基本几何类型（Sphere, Cube, Cylinder 等）
            # ============================================
            else:
                # 使用 trimesh 库在本地坐标系中创建基本几何形状
                # 调用辅助函数根据 USD primitive 参数生成 trimesh 对象
                trimesh_mesh = self._create_trimesh_from_usd_primitive_dynamic(geom_prim, geom_type)
                # 检查 trimesh 创建是否成功
                if trimesh_mesh is not None:
                    # 尝试获取并应用 USD prim 的缩放属性
                    try:
                        # 获取 prim 的完整路径字符串
                        path = geom_prim.GetPath().pathString
                        # 通过路径重新获取 prim 对象
                        prim = prim_utils.get_prim_at_path(path)
                        # 获取 "xformOp:scale" 属性
                        scale_attr = prim.GetAttribute("xformOp:scale")
                        # 检查缩放属性是否存在、有效且有值
                        if scale_attr and scale_attr.IsValid() and scale_attr.HasValue():
                            # 获取缩放值元组 (sx, sy, sz)
                            scale = tuple(scale_attr.Get())
                            # 只在缩放不是默认的 [1,1,1] 时应用
                            if not all(abs(s - 1.0) < 1e-6 for s in scale):
                                # 使用 trimesh 的内置方法应用缩放变换
                                trimesh_mesh.apply_scale(scale)
                    except Exception as e:
                        # 如果获取缩放失败，记录警告但继续
                        omni.log.warn(f"Could not get scale for {geom_prim.GetPath()}: {e}")
                    
                    # 返回 trimesh 的顶点和扁平化的面索引
                    # trimesh.vertices: numpy array of shape (N, 3)
                    # trimesh.faces.flatten(): 将 (M, 3) 的面数组扁平化为 (M*3,) 的索引数组
                    return trimesh_mesh.vertices, trimesh_mesh.faces.flatten()
        
        # 捕获所有异常，防止单个网格提取失败影响整个初始化过程
        except Exception as e:
            # 记录详细的错误信息，包括几何类型和路径
            omni.log.warn(f"Failed to extract dynamic mesh data from {geom_prim.GetTypeName()} {geom_prim.GetPath()}: {str(e)}")
            # 返回 None 表示提取失败
            return None

    def _create_trimesh_from_usd_primitive_dynamic(self, geom_prim, geom_type):
        """Create a trimesh object from USD primitive parameters in local coordinates.
        
        Args:
            geom_prim: USD geometry primitive 
            geom_type: Type of the primitive
            
        Returns:
            trimesh.Trimesh object or None if creation failed
        """
        try:
            from pxr import UsdGeom
            
            if geom_type == "Sphere":
                sphere_geom = UsdGeom.Sphere(geom_prim)
                radius_attr = sphere_geom.GetRadiusAttr()
                radius = radius_attr.Get() if radius_attr else 1.0
                return trimesh.creation.uv_sphere(radius=radius)
                
            elif geom_type == "Cube":
                cube_geom = UsdGeom.Cube(geom_prim)
                size_attr = cube_geom.GetSizeAttr()
                size = size_attr.Get() if size_attr else 2.0
                return trimesh.creation.box(extents=[size, size, size])
                
            elif geom_type == "Cylinder":
                cylinder_geom = UsdGeom.Cylinder(geom_prim)
                radius_attr = cylinder_geom.GetRadiusAttr()
                height_attr = cylinder_geom.GetHeightAttr()
                radius = radius_attr.Get() if radius_attr else 1.0
                height = height_attr.Get() if height_attr else 2.0
                return trimesh.creation.cylinder(radius=radius, height=height)
                
            else:
                return None
                
        except Exception as e:
            omni.log.warn(f"Failed to create dynamic trimesh for {geom_type}: {str(e)}")
            return None

    def _triangulate_faces_from_list_dynamic(self, faces: list, face_counts: list) -> list:
        """Convert polygonal faces to triangles for dynamic meshes."""
        triangulated_faces = []
        face_idx = 0
        
        for count in face_counts:
            if count == 3:
                triangulated_faces.extend(faces[face_idx:face_idx + 3])
            elif count == 4:
                v0, v1, v2, v3 = faces[face_idx:face_idx + 4]
                triangulated_faces.extend([v0, v1, v2, v0, v2, v3])
            else:
                # Fan triangulation for polygons
                for i in range(1, count - 1):
                    triangulated_faces.extend([faces[face_idx], faces[face_idx + i], faces[face_idx + i + 1]])
            face_idx += count
        
        return triangulated_faces

    def _update_combined_mesh_efficiently(self):
        """高效更新组合网格（静态网格 + 全局动态障碍物）- 使用向量化操作
        
        ⚠️ 重要：这个方法可以检测动态障碍物！
        
        该方法的核心功能：
        每帧实时更新 cfg.mesh_prim_paths 中所有网格的世界坐标位置。
        
        ✅ 支持的场景：
        1. 静态网格的初始变换（如：旋转45度的地面）
        2. 全局动态障碍物的实时运动（如：移动的平台、旋转的门、摆动的障碍物）
        3. 由物理引擎驱动的刚体运动（如：受重力影响的物体）
        4. 动画驱动的物体运动（如：巡逻的障碍物）
        
        检测原理：
        - 每帧通过 XFormPrim.get_world_poses() 获取最新的物体位姿
        - 如果物体移动了，位姿会自动更新
        - 通过变换矩阵重新计算所有顶点的世界坐标
        - 更新 Warp 网格的 BVH 加速结构
        - 光线投射会自动检测到新位置的物体
        
        示例：动态障碍物配置
        ```python
        cfg.mesh_prim_paths = [
            "/World/ground",              # 静态地面
            "/World/moving_platform",     # 移动平台（动态）
            "/World/rotating_obstacle"    # 旋转障碍物（动态）
        ]
        ```
        当 moving_platform 移动时，这个方法会：
        1. 获取其新的世界位置和旋转
        2. 重新计算其所有顶点的世界坐标
        3. 更新组合网格
        4. 激光雷达可以检测到移动后的位置
        
        技术特点：
        1. 使用PyTorch的向量化操作，避免Python循环，大幅提升性能
        2. 使用XFormPrim视图批量获取变换，避免逐个查询USD场景图
        3. 使用Warp的GPU加速BVH重建，支持高效光线投射
        
        更新策略：
        - 每次传感器更新时调用（在 _update_buffers_impl 中）
        - 通过 refit() 更新BVH加速结构（比rebuild()快10x）
        
        与 env_dynamic_mesh 的区别：
        - combined_mesh: 所有环境共享，全局障碍物
        - env_dynamic_mesh: 每个环境独立，环境特定物体
        """
        # ========================================
        # 前置检查：确保所有必要的数据结构已初始化
        # ========================================
        # 如果任何一个必需的属性为None，则跳过更新
        if (self.all_mesh_view is None or  # XFormPrim视图（用于批量获取变换）
            self.combined_mesh is None or  # Warp网格对象
            self.mesh_instance_indices is None or  # 网格实例索引列表
            self.vertex_counts_per_instance is None or  # 每个实例的顶点数
            self.all_base_points is None):  # 局部坐标系的基础顶点
            return
            
        try:
            # ========================================
            # 第一步：批量获取所有网格实例的当前世界变换
            # ========================================
            # 🔑 关键点：get_world_poses() 返回的是**实时**位姿
            # 如果物体移动了，这里获取的就是移动后的新位置
            # 计算网格实例总数
            num_mesh_instances = len(self.mesh_instance_indices)
            current_poses, current_quats = self.all_mesh_view.get_world_poses(
                torch.arange(num_mesh_instances, device=self.device)
            )
            if isinstance(current_poses, np.ndarray):
                # 将numpy数组转换为torch张量并移到GPU
                current_poses = torch.from_numpy(current_poses).to(device=self.device)
            if isinstance(current_quats, np.ndarray):
                current_quats = torch.from_numpy(current_quats).to(device=self.device)
            
            # ========================================
            # 第三步：扩展变换到顶点级别
            # ========================================
            # 每个网格实例有不同数量的顶点，需要将变换复制到每个顶点
            # 例如：
            #   网格实例0有1000个顶点 -> 其变换复制1000次
            #   网格实例1有500个顶点 -> 其变换复制500次
            
            # repeat_interleave按顶点数重复位置向量
            expanded_positions = torch.repeat_interleave(
                current_poses,  # 输入：[num_mesh_instances, 3]
                self.vertex_counts_per_instance.long(),  # 每个实例重复次数：[1000, 500, 800, ...]
                dim=0  # 在第0维（行）上重复
            )
            # 输出：[总顶点数, 3] 例如：[2300, 3]（1000+500+800）
            # 结果：[pos0, pos0, ...(1000次), pos1, pos1, ...(500次), pos2, pos2, ...(800次)]
            
            # 同样重复四元数
            expanded_quats = torch.repeat_interleave(
                current_quats,  # 输入：[num_mesh_instances, 4]
                self.vertex_counts_per_instance.long(),
                dim=0
            )
            # 输出：[总顶点数, 4]
            
            # ========================================
            # 第四步：应用刚体变换（旋转 + 平移）
            # ========================================
            # 刚体变换公式：world_point = quat * local_point + position
            # 这保证了网格的形状不变，只改变位置和朝向
            
            # 步骤1：应用旋转变换
            # quat_apply实现四元数旋转：v' = q * v * q^(-1)
            rotated_points = quat_apply(
                expanded_quats,  # 每个顶点对应的四元数旋转 [总顶点数, 4]
                self.all_base_points  # 局部坐标系的基础顶点 [总顶点数, 3]
            )
            # 输出：旋转后的顶点（仍在原点为中心） [总顶点数, 3]
            
            # 步骤2：应用平移变换
            transformed_points = rotated_points + expanded_positions
            # 输出：最终的世界坐标顶点 [总顶点数, 3]
            # 现在每个顶点都在其正确的世界位置上
            
            # ========================================
            # 第五步：更新Warp网格并重建BVH加速结构
            # ========================================
            # 将PyTorch张量转换为Warp数组（GPU间零拷贝）
            updated_points_wp = wp.from_torch(
                transformed_points,  # PyTorch张量
                dtype=wp.vec3  # Warp的3D向量类型
            )
            # wp.from_torch创建Warp数组的视图，不复制数据，效率极高
            
            # 更新网格的顶点数据
            self.combined_mesh.points = updated_points_wp
            # combined_mesh是wp.Mesh对象，points属性存储所有顶点坐标
            
            # 重新拟合BVH（边界体积层次结构）
            self.combined_mesh.refit()
            # refit()用于拓扑不变的网格（顶点连接关系不变，只有坐标变化）
            # 比rebuild()快约10倍，因为只更新边界框，不重建整个树结构
            # BVH加速结构使光线投射从O(N)优化到O(log N)
            
        except Exception as e:
            # 捕获所有异常，避免网格更新失败导致传感器崩溃
            omni.log.warn(f"Failed to update combined mesh efficiently: {str(e)}")

    def _update_env_dynamic_mesh_efficiently(self):
        """高效更新环境动态网格 - 每个环境独立的动态物体
        
        该方法专门用于更新每个环境中独立的动态物体（如可抓取的物体、移动障碍物）。
        
        与combined_mesh的区别：
        1. combined_mesh: 全局/静态网格，所有环境共享同一套网格
        2. env_dynamic_mesh: 环境特定网格，每个环境有自己的动态物体实例
        
        典型应用场景：
        - 每个环境中可移动的箱子、球体等物体（如YCB数据集物体）
        - 每个环境独立的动态障碍物
        - 机器人操作任务中的目标物体
        
        技术实现：
        - 与combined_mesh使用相同的向量化策略
        - 支持不同环境中同一物体的不同位姿
        - 使用独立的Warp网格和BVH加速结构
        """
        # ========================================
        # 前置检查：确保环境动态网格系统已初始化
        # ========================================
        # 检查所有必需的数据结构是否存在
        if (
            self.all_env_dynamic_mesh_view is None  # 环境动态物体的XFormPrim视图
            or self.env_dynamic_mesh is None  # Warp网格对象（独立于combined_mesh）
            or self.env_mesh_instance_indices is None  # 环境动态网格实例索引
            or self.env_vertex_counts_per_instance is None  # 每个实例的顶点数
            or self.env_base_points is None  # 环境动态网格的局部坐标顶点
        ):
            # 如果没有配置环境动态网格，直接返回（不是错误）
            return

        try:
            # ========================================
            # 第一步：批量获取所有环境动态网格实例的世界变换
            # ========================================
            # 计算环境动态网格实例总数
            # 例如：4个环境，每个环境3个物体 = 12个实例
            num_instances = len(self.env_mesh_instance_indices)
            
            # 批量获取所有实例的世界位姿
            # 这里获取的是每个环境中每个物体的当前位置和旋转
            current_poses, current_quats = self.all_env_dynamic_mesh_view.get_world_poses(
                torch.arange(num_instances, device=self.device)  # 索引：[0, 1, ..., 11]
            )
            # current_poses: [num_instances, 3] = [12, 3]
            # current_quats: [num_instances, 4] = [12, 4]
            # 实例顺序：env0_obj0, env0_obj1, env0_obj2, env1_obj0, env1_obj1, ...

            # ========================================
            # 第二步：类型转换（numpy -> torch，确保在GPU上）
            # ========================================
            # XFormPrim.get_world_poses可能返回numpy数组
            if isinstance(current_poses, np.ndarray):
                current_poses = torch.from_numpy(current_poses).to(device=self.device)
            if isinstance(current_quats, np.ndarray):
                current_quats = torch.from_numpy(current_quats).to(device=self.device)

            # ========================================
            # 第三步：将实例级变换扩展到顶点级
            # ========================================
            # 每个物体实例有不同数量的顶点
            # 例如：
            #   CrackerBox: 2000个顶点
            #   SugarBox: 1500个顶点
            #   TomatoSoupCan: 1800个顶点
            # 需要将每个实例的变换复制到其所有顶点
            
            # 扩展位置向量
            expanded_positions = torch.repeat_interleave(
                current_poses,  # [num_instances, 3]
                self.env_vertex_counts_per_instance.long(),  # 每个实例的顶点数
                dim=0
            )
            # 输出：[总顶点数, 3]
            # 例如：12个实例，总共63600个顶点 -> [63600, 3]
            
            # 扩展旋转四元数
            expanded_quats = torch.repeat_interleave(
                current_quats,  # [num_instances, 4]
                self.env_vertex_counts_per_instance.long(),
                dim=0
            )
            # 输出：[总顶点数, 4]

            # ========================================
            # 第四步：应用刚体变换到所有顶点
            # ========================================
            # 使用相同的变换公式：world_point = quat * local_point + position
            
            # 旋转 + 平移（一步完成）
            transformed_points = quat_apply(
                expanded_quats,  # 每个顶点的旋转 [总顶点数, 4]
                self.env_base_points  # 环境动态网格的局部坐标顶点 [总顶点数, 3]
            ) + expanded_positions  # 加上平移 [总顶点数, 3]
            # 输出：所有动态物体在世界坐标系中的顶点位置 [总顶点数, 3]
            
            # 这一步将每个环境中的物体从局部坐标系变换到世界坐标系
            # 考虑了物体的旋转和平移（例如箱子被抓取后的新位置）

            # ========================================
            # 第五步：更新Warp网格并重建BVH
            # ========================================
            # 转换为Warp格式（GPU零拷贝）
            updated_points_wp = wp.from_torch(
                transformed_points,  # PyTorch张量
                dtype=wp.vec3  # Warp 3D向量
            )

            # 更新环境动态网格的顶点数据
            self.env_dynamic_mesh.points = updated_points_wp
            # env_dynamic_mesh是独立的wp.Mesh对象，与combined_mesh分离
            # 这样可以分别对静态和动态物体进行光线投射
            
            # 重新拟合BVH加速结构
            self.env_dynamic_mesh.refit()
            # 更新BVH以反映新的顶点位置
            # 使得光线投射能够正确检测到物体的新位置

        except Exception as e:
            # 错误处理：记录警告但不中断传感器更新
            # 即使环境动态网格更新失败，combined_mesh仍可正常工作
            omni.log.warn(f"Failed to update env dynamic mesh efficiently: {str(e)}")
            
            
            
#stage = omni.usd.get_context().get_stage()

#mesh_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/LF_HIP/visuals/mesh_0")
#mesh_prim.GetTypeName() == "Mesh"? 
#Mesh = UsdGeom.Mesh.Get(stage,mesh_prim.GetPath().pathString)
#mesh_geom = UsdGeom.Mesh(mesh_prim)
#mesh_prim.GetChildren()
#mesh_prim.GetParent()