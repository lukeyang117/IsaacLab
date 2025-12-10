"""Configuration for Go2 quadruped with PVCNN-based perception."""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Import Isaac Lab MDP functions
from isaaclab.envs import mdp as isaac_mdp

# Import custom PVCNN MDP components
import go2_pvcnn.mdp as custom_mdp


# Import LiDAR sensor
from isaaclab.sensors import LidarSensorCfg
from isaaclab.sensors.ray_caster.patterns import LivoxPatternCfg

# Import usd root
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Terrain Configuration
##
def create_dynamic_objects_collection_cfg(num_objects: int = 3) -> RigidObjectCollectionCfg:
    """Create RigidObjectCollectionCfg with 3 YCB objects per environment.
    
    创建动态USD物体配置，每个环境3个物体（CrackerBox, SugarBox, TomatoSoupCan）。
    
    Args:
        num_objects: Number of objects per environment (fixed at 3 for YCB objects)
        
    Returns:
        RigidObjectCollectionCfg configured with 3 YCB objects
    """
    # 固定使用3个YCB物体，每个物体对应一个USD文件
    # Fixed 3 YCB objects, each with its own USD file
    ycb_objects = [
        {
            "name": "cracker_box",
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
            "mesh_name": "_03_cracker_box",  # USD内部的mesh名称
        },
        {
            "name": "sugar_box",
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            "mesh_name": "_04_sugar_box",
        },
        {
            "name": "tomato_soup_can",
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            "mesh_name": "_05_tomato_soup_can",
        },
    ]
    
    # Create rigid objects dictionary
    rigid_objects = {}
    
    for i, obj_info in enumerate(ycb_objects):
        # 使用 {ENV_REGEX_NS} 让物体在每个环境中自动复制
        # Use {ENV_REGEX_NS} to automatically replicate objects across environments
        rigid_objects[f"object_{i}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Object_{i}",  # 每个环境: Object_0, Object_1, Object_2
            spawn=sim_utils.UsdFileCfg(
                usd_path=obj_info["usd_path"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                activate_contact_sensors=True,  # 启用接触传感器 - Enable contact sensors
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.5),  # 将通过事件随机化位置 - Will be randomized by events
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
    
    print(f"[create_dynamic_objects_collection_cfg] 创建了 3 个动态物体配置（每个环境）")
    print(f"[create_dynamic_objects_collection_cfg] Object_0: CrackerBox (mesh: {ycb_objects[0]['mesh_name']})")
    print(f"[create_dynamic_objects_collection_cfg] Object_1: SugarBox (mesh: {ycb_objects[1]['mesh_name']})")
    print(f"[create_dynamic_objects_collection_cfg] Object_2: TomatoSoupCan (mesh: {ycb_objects[2]['mesh_name']})")
    
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        #Other terrain types commented out for simpler flat terrain
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


##
# Pre-defined configs
##
from go2_pvcnn.assets import UNITREE_GO2_CFG  # isort: skip

##
# Scene definition
##


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """Configuration for Go2 robot scene with LiDAR and dynamic objects."""

    # ========================================
    # Scene Replication Settings
    # ========================================
    # Enable physics replication for ContactSensor filtering to work properly
    # This is REQUIRED when using filter_prim_paths_expr in ContactSensor
    replicate_physics: bool = True
    
    # Ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Robot
    robot: ArticulationCfg = MISSING

    # Height scanner
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # ========================================
    # Contact Sensors - 测试 PhysX 自动环境分组
    # ========================================
    # 假设验证：
    # 虽然 filter pattern 会匹配所有环境的机器人（{ENV_REGEX_NS}/Robot/*），
    # 但 PhysX 应该会根据 sensor prim 所在的环境自动将数据分组到
    # 正确的环境索引，而不是真的返回所有304个body的数据。
    # 
    # 如果这个假设正确，force_matrix_w 的形状应该是：
    #   [num_envs, 1, 19, 3]  而不是  [num_envs, 1, 304, 3]
    
    # 机器人的接触传感器（检测所有碰撞）
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    
    # 为每个动态物体创建接触传感器
    # ⚠️ 重要：filter_prim_paths_expr 只支持 one-to-many filtering
    # 即：1个sensor body 对多个filter bodies
    # 不能使用 Robot/* 因为这会匹配所有19个Robot bodies导致数量不匹配
    
    # Object_0: CrackerBox - 检测与整个机器人的碰撞
    object_0_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        history_length=3,
        track_air_time=False,
        # 移除 filter - 检测所有碰撞（包括地面和机器人）
        # 或者使用不带通配符的路径
    )
    
    # Object_1: SugarBox
    object_1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object_1",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=[
            # Base
            "{ENV_REGEX_NS}/Robot/base",           # 1. 基座
            
            # Front Left (FL) leg - 4个links
            "{ENV_REGEX_NS}/Robot/FL_hip",         # 2. 左前髋关节link
            "{ENV_REGEX_NS}/Robot/FL_thigh",       # 3. 左前大腿
            "{ENV_REGEX_NS}/Robot/FL_calf",        # 4. 左前小腿
            "{ENV_REGEX_NS}/Robot/FL_foot",        # 5. 左前脚
            
            # Front Right (FR) leg - 4个links
            "{ENV_REGEX_NS}/Robot/FR_hip",         # 6. 右前髋关节link
            "{ENV_REGEX_NS}/Robot/FR_thigh",       # 7. 右前大腿
            "{ENV_REGEX_NS}/Robot/FR_calf",        # 8. 右前小腿
            "{ENV_REGEX_NS}/Robot/FR_foot",        # 9. 右前脚
            
            # Rear Left (RL) leg - 4个links
            "{ENV_REGEX_NS}/Robot/RL_hip",         # 10. 左后髋关节link
            "{ENV_REGEX_NS}/Robot/RL_thigh",       # 11. 左后大腿
            "{ENV_REGEX_NS}/Robot/RL_calf",        # 12. 左后小腿
            "{ENV_REGEX_NS}/Robot/RL_foot",        # 13. 左后脚
            
            # Rear Right (RR) leg - 4个links
            "{ENV_REGEX_NS}/Robot/RR_hip",         # 14. 右后髋关节link
            "{ENV_REGEX_NS}/Robot/RR_thigh",       # 15. 右后大腿
            "{ENV_REGEX_NS}/Robot/RR_calf",        # 16. 右后小腿
            "{ENV_REGEX_NS}/Robot/RR_foot",        # 17. 右后脚
        ],
    )
    
    # Object_2: TomatoSoupCan
    object_2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object_2",
        history_length=3,
        track_air_time=False,
    )

    # LiDAR sensor for PVCNN - Updated to include dynamic objects
    lidar_sensor = LidarSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=LidarSensorCfg.OffsetCfg(pos=(0.3, 0.0, 0.2)),  # Forward and up from base
        attach_yaw_only=False,
        pattern_cfg=LivoxPatternCfg(
            sensor_type="mid360",
            samples=24000,  # Target number of points for PVCNN
            use_simple_grid=True,  # Use simple grid pattern 
            vertical_line_num=62,  # 62 vertical lines
            horizontal_line_num=33,  # 33 horizontal lines = 2046 rays
        ),
        mesh_prim_paths=[
            "/World/ground"                      # 地形
        ],
        dynamic_env_mesh_prim_paths=[
            # 精确的mesh路径（不支持通配符）
            # Exact mesh paths (wildcards not supported)
            # Object_0: CrackerBox
            "{ENV_REGEX_NS}/Object_0/_03_cracker_box",
            # Object_1: SugarBox  
            "{ENV_REGEX_NS}/Object_1/_04_sugar_box",
            # Object_2: TomatoSoupCan
            "{ENV_REGEX_NS}/Object_2/_05_tomato_soup_can",
        ],
        max_distance=10.0,  # Reduce max distance for better ground coverage
        min_range=0.1,  # Allow closer points
        return_pointcloud=False,  # We'll use ray_hits_w instead
        pointcloud_in_world_frame=True,  # Use world frame
        enable_sensor_noise=False,  # Disable noise for now
        random_distance_noise=0.0,
        update_frequency=10.0,  # 10 Hz
        debug_vis=False,  # Disable viz for performance
    )

    # Lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    
    # ========================================
    # Dynamic Objects - 每个环境3个物体
    # ========================================
    # 注意：RigidObjectCollectionCfg会为每个环境复制物体
    # 3个物体 × 512个环境 = 1536个物体实例（总共）
    # Each environment will have 3 objects (CrackerBox, SugarBox, TomatoSoupCan)
    dynamic_objects = create_dynamic_objects_collection_cfg(num_objects=3)


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # Simple velocity command
    base_velocity = isaac_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1, 1)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # PVCNN features from LiDAR
        
        # Note: This will be set dynamically with the PVCNN wrapper instance
        pvcnn_features = ObsTerm(func=custom_mdp.pvcnn_features, params={"sensor_cfg": SceneEntityCfg("lidar_sensor")})

        # Base velocity in base frame
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        
        # Projected gravity
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        # Joint positions and velocities
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # Velocity commands
        velocity_commands = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"})
        
        # Last actions
        actions = ObsTerm(func=isaac_mdp.last_action)


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group (privileged information)."""

        # Height scan
        height_scan = ObsTerm(
            func=isaac_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Define observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for randomization events."""

    # ========================================
    # Dynamic Objects Randomization Events
    # ========================================
    # 在启动时随机化动态物体位置
    # Randomize dynamic objects positions on startup
    randomize_objects_startup = EventTerm(
        func=custom_mdp.randomize_dynamic_objects_on_startup,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("dynamic_objects"),
            "robot_cfg": SceneEntityCfg("robot"),
            "position_range": (-3.0, 3.0, -3.0, 3.0),  # 相对机器人±3米范围 - ±3m range relative to robot
            "height_offset": 0.3,  # 地形上方0.3米 - 0.3m above terrain
        },
    )

    # ========================================
    # Robot Reset Events (执行顺序很重要！先重置机器人，再生成物体)
    # Robot Reset Events (Order matters! Reset robot first, then spawn objects)
    # ========================================
    # Reset all environments when episode ends
    reset_base = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # Reset robot joints
    reset_robot_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    # 每次reset时随机化动态物体位置（必须在机器人重置之后！）
    # Randomize dynamic objects positions on reset (MUST be after robot reset!)
    reset_objects_position = EventTerm(
        func=custom_mdp.reset_dynamic_objects_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("dynamic_objects"),
            "robot_cfg": SceneEntityCfg("robot"),
            "position_range": (-3.0, 3.0, -3.0, 3.0),  # 相对机器人±3米范围 - ±3m range relative to robot
            "height_offset": 0.3,  # 地形上方0.3米 - 0.3m above terrain
        },
    )

    # Push robot
    push_robot = EventTerm(
        func=isaac_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task rewards
    track_lin_vel_xy_exp = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # -- penalties
    lin_vel_z_l2 = RewTerm(func=custom_mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=custom_mdp.flat_orientation_l2, weight=-0.5)
    joint_torques_l2 = RewTerm(func=custom_mdp.joint_torques_l2, weight=-1.0e-5)
    joint_acc_l2 = RewTerm(func=custom_mdp.joint_acc_l2, weight=-2.5e-7)
    joint_vel_l2 = RewTerm(func=custom_mdp.joint_vel_l2, weight=-1.0e-4)
    action_rate_l2 = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.01)
    joint_power = RewTerm(func=custom_mdp.joint_power, weight=-2.0e-5)
    joint_pos_limits = RewTerm(func=custom_mdp.joint_pos_limits, weight=-1.0)
    
    # ========================================
    # 碰撞惩罚系统 - 使用物体级 ContactSensor（相对路径实现环境隔离）
    # ========================================
    # 方案：每个物体有独立的 ContactSensor，使用相对路径 filter 同环境的机器人
    # 这样可以精确获取每个物体与机器人的碰撞信息，且有环境隔离
    
    # 非足端部位与地面碰撞惩罚
    body_terrain_collision = RewTerm(
        func=custom_mdp.body_ground_collision_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 1.0,  # 接触力阈值
        },
    )
    
    # 机器人与动态物体碰撞惩罚（使用物体的 ContactSensor）
    # CrackerBox 碰撞 - 最重惩罚
    collision_crackerbox = RewTerm(
        func=custom_mdp.object_contact_penalty,
        weight=-5.0,  # 直接用 weight 控制惩罚强度
        params={
            "sensor_cfg": SceneEntityCfg("object_0_contact"),
            "threshold": 0.1,
        },
    )
    
    # SugarBox 碰撞 - 中等惩罚
    collision_sugarbox = RewTerm(
        func=custom_mdp.object_contact_penalty,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg("object_1_contact"),
            "threshold": 0.1,
        },
    )
    
    # TomatoSoupCan 碰撞 - 最轻惩罚
    collision_tomatosoupcan = RewTerm(
        func=custom_mdp.object_contact_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("object_2_contact"),
            "threshold": 0.1,
        },
    )

    
    # -- optional: feet air time and sliding
    # feet_air_time_positive_reward = RewTerm(
    #     func=custom_mdp.feet_air_time_positive_reward,
    #     weight=0.125,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )
    # feet_slide = RewTerm(
    #     func=custom_mdp.feet_slide,
    #     weight=-0.1,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot")},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=custom_mdp.bad_orientation,
        params={"limit_angle": 1.0},
    )
    base_height = DoneTerm(
        func=custom_mdp.base_height,
        params={"minimum_height": 0.2},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


##
# Environment configuration
##


@configclass
class Go2PvcnnEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go2 quadruped environment with PVCNN."""

    # Scene settings
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 20.0
        
        # Simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # Note: disable_contact_processing is deprecated and removed
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # Update scene
        self.scene.robot = self._get_robot_cfg()
        
        print(f"[Go2PvcnnEnvCfg] 配置完成：每个环境3个动态物体，共{self.scene.num_envs}个环境")
        print(f"[Go2PvcnnEnvCfg] Total objects in all environments: {3 * self.scene.num_envs}")
        
    def _get_robot_cfg(self) -> ArticulationCfg:
        """Get the robot articulation configuration."""
        # Use the pre-configured Unitree Go2 from our assets
        return UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Go2PvcnnEnvCfg_PLAY(Go2PvcnnEnvCfg):
    """Configuration for playing/evaluation."""
    
    def __post_init__(self):
        # Post-initialization
        super().__post_init__()
        
        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Disable randomization for play
        self.observations.policy.enable_corruption = False
        
        # Remove random pushing for play
        self.events.push_robot = None
        
        print(f"[Go2PvcnnEnvCfg_PLAY] Configured for play mode with {3 * 50} dynamic objects")
