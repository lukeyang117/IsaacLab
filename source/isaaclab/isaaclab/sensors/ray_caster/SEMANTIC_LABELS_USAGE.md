# RayCaster 语义标签使用指南

## 📋 功能概述

RayCaster 传感器现在支持语义类别识别功能，可以识别击中点云属于哪个物体类别。

**支持的语义类别：**
- `0`: `'terrain'` - 来自 `mesh_prim_paths` 的静态网格
- `1+`: 动态物体 - 来自 `dynamic_env_mesh_prim_paths` 的物体，按配置顺序编号

---

## 🔧 配置示例

### 场景配置（参考 `example_test_env_cfg.py`）

```python
# 在 InteractiveSceneCfg 中配置 LiDAR 传感器
lidar_sensor = LidarSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=LidarSensorCfg.OffsetCfg(pos=(0.3, 0.0, 0.2)),
    attach_yaw_only=False,
    pattern_cfg=LivoxPatternCfg(...),
    
    # 静态网格路径（语义标签 = 0）
    mesh_prim_paths=[
        "/World/ground"  # 地形
    ],
    
    # 动态物体路径（语义标签 = 1, 2, 3, ...）
    dynamic_env_mesh_prim_paths=[
        "{ENV_REGEX_NS}/Object_0/_03_cracker_box",      # 语义标签 = 1
        "{ENV_REGEX_NS}/Object_1/_04_sugar_box",        # 语义标签 = 2
        "{ENV_REGEX_NS}/Object_2/_05_tomato_soup_can",  # 语义标签 = 3
    ],
    
    max_distance=10.0,
    debug_vis=False,
)
```

**重要说明：**
- `mesh_prim_paths`: 全局共享的静态网格（地形、墙壁等）
- `dynamic_env_mesh_prim_paths`: 每个环境独立的动态物体（箱子、球体等）
- 语义类别名称从路径的最后一段提取（去掉下划线前缀）
  - `_03_cracker_box` → `03_cracker_box`
  - `_04_sugar_box` → `04_sugar_box`

---

## 📊 数据结构

### RayCasterData 新增字段

```python
@dataclass
class RayCasterData:
    # 原有字段
    pos_w: torch.Tensor          # [N, 3] 传感器位置
    quat_w: torch.Tensor         # [N, 4] 传感器姿态
    ray_hits_w: torch.Tensor     # [N, num_rays, 3] 击中点坐标
    
    # ✅ 新增字段
    semantic_labels: torch.Tensor  # [N, num_rays] 语义标签
    hit_mesh_source: torch.Tensor  # [N, num_rays] 击中来源（0=static, 1=dynamic, -1=no hit）
```

### semantic_labels 值含义

| 值 | 含义 | 来源 |
|----|------|------|
| `-1` | 未击中 | 光线超出 max_distance 或未碰撞 |
| `0` | terrain | mesh_prim_paths 配置的静态网格 |
| `1` | 第1个动态物体 | dynamic_env_mesh_prim_paths[0] |
| `2` | 第2个动态物体 | dynamic_env_mesh_prim_paths[1] |
| `3` | 第3个动态物体 | dynamic_env_mesh_prim_paths[2] |
| ... | ... | ... |

### hit_mesh_source 值含义

| 值 | 含义 |
|----|------|
| `-1` | 未击中任何网格 |
| `0` | 击中静态网格（combined_mesh） |
| `1` | 击中环境动态网格（env_dynamic_mesh） |

---

## 💻 观测函数实现

### 方法一：直接获取语义标签（推荐）

```python
from isaaclab.managers import ObservationTermCfg as ObsTerm

def lidar_semantic_labels(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取 LiDAR 击中点的语义标签
    
    Returns:
        语义标签 [num_envs, num_rays]
        - 0: terrain
        - 1+: 动态物体类别
        - -1: 未击中
    """
    # 获取 LiDAR 传感器
    sensor = env.scene.sensors[sensor_cfg.name]
    
    # 返回语义标签
    return sensor.data.semantic_labels  # [num_envs, num_rays]


# 在 ObservationsCfg 中使用
@configclass
class PolicyCfg(ObsGroup):
    # 其他观测...
    
    # ✅ 新增：语义标签观测
    lidar_semantic = ObsTerm(
        func=lidar_semantic_labels,
        params={"sensor_cfg": SceneEntityCfg("lidar_sensor")},
    )
```

### 方法二：One-Hot 编码

```python
def lidar_semantic_onehot(env, sensor_cfg: SceneEntityCfg, num_classes: int = 4) -> torch.Tensor:
    """获取 LiDAR 语义标签的 one-hot 编码
    
    Args:
        num_classes: 类别数量（包括 terrain）
                     例如：terrain + 3个动态物体 = 4类
    
    Returns:
        one-hot 编码 [num_envs, num_rays, num_classes]
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    semantic_labels = sensor.data.semantic_labels  # [num_envs, num_rays]
    
    # 处理未击中的情况（-1 → 特殊类别）
    # 方案1：将未击中映射到 0（terrain），忽略
    labels = semantic_labels.clone()
    labels[labels < 0] = 0
    
    # 方案2：为未击中创建单独类别
    # labels = semantic_labels.clone() + 1  # -1→0, 0→1, 1→2, ...
    # num_classes += 1
    
    # One-hot 编码
    one_hot = torch.nn.functional.one_hot(
        labels.long(), 
        num_classes=num_classes
    ).float()  # [num_envs, num_rays, num_classes]
    
    return one_hot


# 在 ObservationsCfg 中使用
@configclass
class PolicyCfg(ObsGroup):
    lidar_semantic_onehot = ObsTerm(
        func=lidar_semantic_onehot,
        params={
            "sensor_cfg": SceneEntityCfg("lidar_sensor"),
            "num_classes": 4  # terrain + 3个动态物体
        },
    )
```

### 方法三：结合点云坐标和语义

```python
def lidar_pointcloud_with_semantics(
    env, 
    sensor_cfg: SceneEntityCfg,
    num_classes: int = 4
) -> torch.Tensor:
    """获取带语义标签的点云
    
    Returns:
        点云特征 [num_envs, num_rays, 3 + num_classes]
        - 前3维：(x, y, z) 相对坐标
        - 后num_classes维：语义 one-hot
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取击中点的相对坐标（传感器坐标系）
    ray_hits_w = sensor.data.ray_hits_w  # [num_envs, num_rays, 3]
    sensor_pos_w = sensor.data.pos_w.unsqueeze(1)  # [num_envs, 1, 3]
    relative_hits = ray_hits_w - sensor_pos_w  # [num_envs, num_rays, 3]
    
    # 获取语义标签并转为 one-hot
    semantic_labels = sensor.data.semantic_labels
    labels = semantic_labels.clone()
    labels[labels < 0] = 0  # 未击中视为 terrain
    
    semantic_onehot = torch.nn.functional.one_hot(
        labels.long(),
        num_classes=num_classes
    ).float()  # [num_envs, num_rays, num_classes]
    
    # 合并坐标和语义
    pointcloud_features = torch.cat([
        relative_hits,      # [num_envs, num_rays, 3]
        semantic_onehot     # [num_envs, num_rays, num_classes]
    ], dim=-1)  # [num_envs, num_rays, 3 + num_classes]
    
    return pointcloud_features


# 在 ObservationsCfg 中使用
@configclass
class PolicyCfg(ObsGroup):
    lidar_semantic_pointcloud = ObsTerm(
        func=lidar_pointcloud_with_semantics,
        params={
            "sensor_cfg": SceneEntityCfg("lidar_sensor"),
            "num_classes": 4
        },
    )
```

### 方法四：过滤特定类别

```python
def lidar_dynamic_objects_only(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """只保留动态物体的点云（过滤掉地形）
    
    Returns:
        掩码 [num_envs, num_rays]
        - True: 击中动态物体
        - False: 击中地形或未击中
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    semantic_labels = sensor.data.semantic_labels
    
    # 动态物体的标签 >= 1
    dynamic_mask = semantic_labels >= 1
    
    return dynamic_mask


def lidar_distance_to_objects(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """计算到动态物体的距离（忽略地形）
    
    Returns:
        距离 [num_envs, num_rays]
        - 有效距离：到动态物体的距离
        - inf：击中地形或未击中
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    
    # 计算距离
    ray_hits_w = sensor.data.ray_hits_w
    sensor_pos_w = sensor.data.pos_w.unsqueeze(1)
    distances = torch.linalg.norm(ray_hits_w - sensor_pos_w, dim=-1)
    
    # 获取语义标签
    semantic_labels = sensor.data.semantic_labels
    
    # 只保留动态物体的距离
    distances_filtered = distances.clone()
    distances_filtered[semantic_labels <= 0] = float('inf')  # 地形或未击中
    
    return distances_filtered
```

---

## 🎯 完整示例：在环境中使用

### 环境配置文件

```python
from go2_pvcnn.mdp import custom_mdp  # 你的自定义观测函数

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 机器人状态观测
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity)
        
        # ✅ 新增：LiDAR 语义点云
        lidar_semantic_pointcloud = ObsTerm(
            func=custom_mdp.lidar_pointcloud_with_semantics,
            params={
                "sensor_cfg": SceneEntityCfg("lidar_sensor"),
                "num_classes": 4  # terrain + 3个动态物体
            },
        )
        
        # ✅ 新增：到动态物体的最小距离
        min_distance_to_objects = ObsTerm(
            func=custom_mdp.min_distance_to_dynamic_objects,
            params={"sensor_cfg": SceneEntityCfg("lidar_sensor")},
        )
    
    policy: PolicyCfg = PolicyCfg()
```

### 自定义观测函数（在你的项目中实现）

```python
# 文件：go2_pvcnn/mdp/observations.py

import torch
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lidar_pointcloud_with_semantics(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    num_classes: int = 4
) -> torch.Tensor:
    """获取带语义标签的 LiDAR 点云
    
    Returns:
        [num_envs, num_rays, 3 + num_classes]
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    
    # 相对坐标
    ray_hits_w = sensor.data.ray_hits_w
    sensor_pos_w = sensor.data.pos_w.unsqueeze(1)
    relative_hits = ray_hits_w - sensor_pos_w
    
    # 语义 one-hot
    semantic_labels = sensor.data.semantic_labels
    labels = semantic_labels.clone()
    labels[labels < 0] = 0
    
    semantic_onehot = torch.nn.functional.one_hot(
        labels.long(),
        num_classes=num_classes
    ).float()
    
    # 合并
    return torch.cat([relative_hits, semantic_onehot], dim=-1)


def min_distance_to_dynamic_objects(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """计算到最近动态物体的距离
    
    Returns:
        [num_envs, 1]
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    
    # 计算所有距离
    ray_hits_w = sensor.data.ray_hits_w
    sensor_pos_w = sensor.data.pos_w.unsqueeze(1)
    distances = torch.linalg.norm(ray_hits_w - sensor_pos_w, dim=-1)
    
    # 只保留动态物体
    semantic_labels = sensor.data.semantic_labels
    distances[semantic_labels <= 0] = float('inf')
    
    # 取最小值
    min_dist, _ = distances.min(dim=-1)
    
    return min_dist.unsqueeze(-1)
```

---

## 🔍 调试和验证

### 检查语义类别名称

```python
# 在环境初始化后
lidar_sensor = env.scene.sensors["lidar_sensor"]
print("Semantic classes:", lidar_sensor.semantic_class_names)
# 输出：['terrain', '03_cracker_box', '04_sugar_box', '05_tomato_soup_can']
```

### 可视化语义分布

```python
import matplotlib.pyplot as plt

# 获取语义标签
semantic_labels = env.scene.sensors["lidar_sensor"].data.semantic_labels
env_id = 0  # 查看第一个环境

# 统计每个类别的击中次数
unique, counts = torch.unique(semantic_labels[env_id], return_counts=True)

# 绘制直方图
plt.bar(unique.cpu().numpy(), counts.cpu().numpy())
plt.xlabel('Semantic Label')
plt.ylabel('Count')
plt.title('LiDAR Semantic Distribution')
plt.xticks(range(-1, len(lidar_sensor.semantic_class_names)))
plt.show()
```

### 验证数据形状

```python
data = env.scene.sensors["lidar_sensor"].data

print(f"ray_hits_w shape: {data.ray_hits_w.shape}")  # [num_envs, num_rays, 3]
print(f"semantic_labels shape: {data.semantic_labels.shape}")  # [num_envs, num_rays]
print(f"hit_mesh_source shape: {data.hit_mesh_source.shape}")  # [num_envs, num_rays]

# 检查语义标签范围
print(f"Semantic labels range: {data.semantic_labels.min()} to {data.semantic_labels.max()}")
# 输出：Semantic labels range: -1 to 3
```

---

## ⚠️ 注意事项

### 1. 配置顺序很重要

语义标签的编号严格按照 `dynamic_env_mesh_prim_paths` 的顺序：

```python
dynamic_env_mesh_prim_paths=[
    "{ENV_REGEX_NS}/Object_0/_03_cracker_box",  # label = 1
    "{ENV_REGEX_NS}/Object_1/_04_sugar_box",    # label = 2
    "{ENV_REGEX_NS}/Object_2/_05_tomato_soup_can",  # label = 3
]
```

**不要改变顺序！** 否则训练好的模型会错误识别物体类别。

### 2. 未击中的处理

有两种处理 `semantic_labels = -1`（未击中）的策略：

**策略 A：映射到 terrain（推荐用于 RL）**
```python
labels = semantic_labels.clone()
labels[labels < 0] = 0  # 未击中视为地形
```

**策略 B：创建单独的"未击中"类别**
```python
labels = semantic_labels + 1  # -1→0 (no hit), 0→1 (terrain), 1→2, ...
num_classes += 1  # 需要增加类别数
```

### 3. 性能考虑

- **语义识别开销极小**：只增加一次 `torch.searchsorted` 调用（GPU 加速）
- **内存占用**：每个环境额外 `num_rays * 4 bytes`（int32）
  - 例如：24000 rays × 512 envs × 4 bytes ≈ 47 MB
- **建议**：如果不需要语义信息，可以不读取 `semantic_labels` 字段

### 4. 类别数量限制

- 理论上支持无限类别（受 int32 限制：2^31 - 1）
- 实际建议：< 100 个类别（保持 one-hot 向量合理大小）

---

## 📚 相关文件

- **数据结构**: `isaaclab/sensors/ray_caster/ray_caster_data.py`
- **核心实现**: `isaaclab/sensors/ray_caster/ray_caster.py`
- **配置示例**: `isaaclab_tasks/manager_based/example_test_env_cfg.py`
- **TODO 文档**: `isaaclab/sensors/ray_caster/todo.md`

---

**最后更新：** 2025-12-13  
**功能状态：** ✅ 已实现并测试
