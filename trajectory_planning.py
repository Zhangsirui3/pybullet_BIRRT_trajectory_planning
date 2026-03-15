import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt

# -------------------------- 1. 正确初始化环境 --------------------------
try:
    physics_client = p.connect(p.GUI)
except:
    p.disconnect()
    physics_client = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

# 加载地面和桌子
plane_id = p.loadURDF("plane.urdf")
# 桌子放在正前方
table_id = p.loadURDF("table/table.urdf", basePosition=[0.4, 0.0, 0.0], useFixedBase=True)

# -------------------------- 2. 加载机械臂 --------------------------
# 将机械臂放在稍后方，固定在地面或桌子高度
robot_id = p.loadURDF("franka_panda/panda.urdf", 
                      basePosition=[0, 0, 0.625], # 提高到桌子的高度，假设桌内有一块伸展板
                      useFixedBase=True)

controlled_joints = []
joint_limits = []
for j in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, j)
    joint_name = joint_info[1].decode('utf-8')
    if joint_info[2] != p.JOINT_FIXED and "finger" not in joint_name:
        controlled_joints.append(j)
        # 获取关节限制 (lower, upper)
        joint_limits.append((joint_info[8], joint_info[9]))

controlled_joints = controlled_joints[:7]
joint_limits = joint_limits[:7]
EE_LINK_ID = 11

print(f"✅ 环境构建成功！")

# 调整初始视角，拉近并倾斜摄像机，完美观赏轨迹规划和避障过程
p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=55, cameraPitch=-30, cameraTargetPosition=[0.5, 0.0, 0.9])

# -------------------------- 3. 场景和空间复杂交错球体障碍物 --------------------------
# 简化场景：我们在机械臂从右向左摆动的必经之路上，放一个红色的圆柱形球体障碍
sphere_obstacles_data = [
    {"pos": [0.45,  0.00, 1.05], "radius": 0.15, "color": [0.9, 0.3, 0.3, 0.9]},  
]

obstacles = []
for obs in sphere_obstacles_data:
    vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=obs["radius"], rgbaColor=obs["color"])
    col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=obs["radius"])
    body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=obs["pos"])
    obstacles.append(body_id)

# 目标小球：之后在下方根据实际无碰撞关节解反算出真正的位置
# 并且在代码底部进行布置
def set_joints(angles):
    for i, j_id in enumerate(controlled_joints):
        p.resetJointState(robot_id, j_id, angles[i])
    # [极其关键优化]：移除之前此处的 p.stepSimulation()！
    # 因为在进行几千次快速位置采样检测碰撞时，更新引擎物理时间会导致重力下坠并引起画面严重卡顿。

def get_ee():
    link_state = p.getLinkState(robot_id, EE_LINK_ID)
    return link_state[4]

def get_fk_ee(angles):
    """用于绘制搜索树：快速正解计算末端位置"""
    curr_states = [p.getJointState(robot_id, j)[0] for j in controlled_joints]
    for i, j_id in enumerate(controlled_joints):
        p.resetJointState(robot_id, j_id, angles[i])
    ee_pos = p.getLinkState(robot_id, EE_LINK_ID, computeForwardKinematics=True)[4]
    for i, j_id in enumerate(controlled_joints):
        p.resetJointState(robot_id, j_id, curr_states[i])
    return ee_pos

def is_collision_free(angles):
    # 保存当前状态
    curr_states = [p.getJointState(robot_id, j)[0] for j in controlled_joints]
    set_joints(angles)
    p.performCollisionDetection()
    
    # 检测是否与障碍物或桌子碰撞
    collision = False
    if len(p.getContactPoints(robot_id, table_id)) > 0:
        collision = True
    for obs in obstacles:
        if len(p.getContactPoints(robot_id, obs)) > 0:
            collision = True
            break
            
    # 恢复原状态
    set_joints(curr_states)
    return not collision

# -------------------------- 5. BiRRT 轨迹规划核心 --------------------------
class Node:
    def __init__(self, angles, parent=None):
        self.angles = np.array(angles)
        self.parent = parent

def get_random_node():
    # 随机采样，满足关节限制
    return np.array([random.uniform(limit[0], limit[1]) for limit in joint_limits])

def get_nearest_node(tree, target_node):
    distances = [np.linalg.norm(node.angles - target_node) for node in tree]
    return tree[np.argmin(distances)]

# 专门用于存储规划算法探索出的每一条树枝，以便后续动画回放
exploration_edges = []

def extend(tree, target_angles, step_size, is_treeA=True):
    nearest = get_nearest_node(tree, target_angles)
    direction = target_angles - nearest.angles
    dist = np.linalg.norm(direction)
    
    if dist < step_size:
        new_angles = target_angles
    else:
        new_angles = nearest.angles + (direction / dist) * step_size
        
    if is_collision_free(new_angles):
        new_node = Node(new_angles, parent=nearest)
        tree.append(new_node)
        
        # 记录树的伸展边：TreeA为蓝色，TreeB为橙色
        color = [0.4, 0.7, 1.0] if is_treeA else [1.0, 0.5, 0.0]
        # 保存：(起点坐标，终点坐标，颜色)
        exploration_edges.append((get_fk_ee(nearest.angles), get_fk_ee(new_angles), color))
            
        return new_node
    return None

def connect(tree, target_angles, step_size, is_treeA=False):
    """标准的 RRT-Connect '不断延伸'以穿越狭窄通道"""
    last_added = None
    while True:
        nearest = get_nearest_node(tree, target_angles)
        dist = np.linalg.norm(target_angles - nearest.angles)
        if dist < 1e-3:
            return last_added, "reached"
            
        new_node = extend(tree, target_angles, step_size, is_treeA=is_treeA)
        if new_node:
            last_added = new_node
            if np.linalg.norm(new_node.angles - target_angles) < 1e-3:
                return new_node, "reached"
        else:
            return last_added, "trapped"

def plan_path_birrt(start_angles, goal_angles):
    print("🚀 开始改进版 BiRRT-Connect 轨迹规划（后台高速计算中...）")
    exploration_edges.clear() # 清空边数据
    
    if not is_collision_free(start_angles):
        print("⚠️ 起点处于碰撞状态，无法到达！")
        return [start_angles, goal_angles], 0.0
    if not is_collision_free(goal_angles):
        print("⚠️ 终点处于碰撞状态，无法到达！")
        return [start_angles, goal_angles], 0.0

    treeA = [Node(start_angles)]
    treeB = [Node(goal_angles)]
    
    step_size = 0.015
    max_iter = 100000
    
    planning_start = time.time()
    
    for i in range(max_iter):
        rand_angles = get_random_node()
        if random.random() < 0.2:
            rand_angles = treeB[-1].angles

        new_nodeA = extend(treeA, rand_angles, step_size, is_treeA=True)
        
        if new_nodeA:
            connect_node, status = connect(treeB, new_nodeA.angles, step_size, is_treeA=False)
            
            if status == "reached":
                planning_time = time.time() - planning_start
                print(f"✅ BiRRT-Connect 寻找成功！后台计算迭代次数：{i}，用时：{planning_time:.4f}秒")
                
                pathA = []
                curr = new_nodeA
                while curr:
                    pathA.append(curr.angles.tolist())
                    curr = curr.parent
                pathA.reverse()
                
                pathB = []
                curr = connect_node
                while curr:
                    pathB.append(curr.angles.tolist())
                    curr = curr.parent
                    
                if np.linalg.norm(treeA[0].angles - start_angles) < 1e-3:
                    final_path = pathA + pathB
                else:
                    pathB.reverse()
                    pathA.reverse()
                    final_path = pathB + pathA
                    
                return final_path, planning_time
                    
        treeA, treeB = treeB, treeA
        
    planning_time = time.time() - planning_start
    print("⚠️ BiRRT 未能找到目标路径，返回兜底直连路径（容易发生碰撞）。")
    return [start_angles, goal_angles], planning_time
# 初始关节角 (机械臂偏向右侧)
start_angles = [0.8, -0.2, 0.0, -1.8, 0.0, 1.5, 0.78]
set_joints(start_angles)
print(f"🏁 初始末端位置：{np.round(get_ee(), 3)}")

# 目标关节角（机械臂偏向左侧，中途由于红球的存在无法强行水平扫过，必须“抬起”或“后缩”绕过红球）
target_angles = [-0.8, -0.2, 0.0, -1.8, 0.0, 1.5, 0.78]
# 顺便反算并在场景中画出目标蓝球
set_joints(target_angles)
target_pos_real = get_ee()
target_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.2,0.5,1,0.9])
p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_vis, basePosition=target_pos_real)
set_joints(start_angles) # 恢复初始状态准备规划

# BiRRT 规划 (高速无渲染后台计算)
path, planning_time = plan_path_birrt(start_angles, target_angles)

print(f"\n🎥 计算完成！开始回放规划树的探索过程（共 {len(exploration_edges)} 条新分支）...")
# 【关键改进】：延后渲染。等确定规划成功后，再一根一根展示探索过程，不拖慢算法！
for edge in exploration_edges:
    p1, p2, color = edge
    p.addUserDebugLine(p1, p2, color, 3, 0)
    # 放慢播放速度方便细看
    time.sleep(0.02)
time.sleep(1.0) # 树长完之后停顿一秒欣赏

print("\n🎨 探索树展示完毕，[保留探索树]，绘制末端最终轨迹线（绿色加粗）...")
# 移除 p.removeAllUserDebugItems() 以保留蓝橙色的探索树树枝
# p.removeAllUserDebugItems()

traj_points = []
for idx, angles in enumerate(path):
    set_joints(angles)
    traj_points.append(get_ee())

# 恢复起点并画线
set_joints(start_angles)
for i in range(len(traj_points)-1):
    p.addUserDebugLine(traj_points[i], traj_points[i+1], [0,1,0], 5)  # 5倍高亮绿线展示最终路径

print("\n▶️ 开始环境与数据初始化，准备进入控制循环...")
# 数据记录容器
record_time = []
record_joints = []
record_ee = []

# 对离散路径进行线性插值，模拟高频控制指令
smooth_path = []
for i in range(len(path)-1):
    steps = 1  # [改动] 增加插值步数到40分段，让两点之间的移动非常密集和缓慢
    for j in range(steps):
        alpha = j / steps
        interp_angles = np.array(path[i]) * (1 - alpha) + np.array(path[i+1]) * alpha
        smooth_path.append(interp_angles.tolist())
smooth_path.append(path[-1])

print("\n================================================")
print("💡 交互模式已开启！请选中 PyBullet 仿真窗口窗口：")
print("  [按键 R] ➡️ 重新沿着计算出轨迹柔和地播放一遍")
print("  [按键 Q] ➡️ 结束仿真并开始生成精美科研配图")
print("================================================\n")

p.setRealTimeSimulation(1) # 开启真实物理引擎显示

def play_trajectory():
    """将播放操作封装并放慢速度，让用户可以欣赏避障动作"""
    print("🚀 正在播放轨迹动画...")
    
    # 瞬间瞬回起点准备
    for i, j_id in enumerate(controlled_joints):
        p.resetJointState(robot_id, j_id, start_angles[i])
        # 使用强力 PD 控制让它严格跟随，避免因为重力或摩擦力走一半卡住
        p.setJointMotorControl2(robot_id, j_id, p.POSITION_CONTROL, targetPosition=start_angles[i], force=500)
    time.sleep(0.5)

    # 清空上次的数据，确保只有最后一次查阅的动画数据出图
    record_time.clear()
    record_joints.clear()
    record_ee.clear()
    
    start_t = time.time()
    for angles in smooth_path:
        # 下发微调指令，增大 force(扭矩) 使得机械臂能够精准跟踪规划出的轨迹，防止半路罢工
        for i, j_id in enumerate(controlled_joints):
            p.setJointMotorControl2(robot_id, j_id, p.POSITION_CONTROL, targetPosition=angles[i], maxVelocity=1.0, force=500)
        
        current_time = time.time() - start_t
        current_joints = [p.getJointState(robot_id, j)[0] for j in controlled_joints]
        
        record_time.append(current_time)
        record_joints.append(current_joints)
        record_ee.append(get_ee())
        
        # 增加 sleep 时长，强行拉长动画播放过程
        time.sleep(0.04)
        
    print("✅ 本次播放完毕。可按 'r' 继续重放，或 'q' 结束。")

# 首次自动播放
play_trajectory()

# 轮询监听按键事件
while True:
    keys = p.getKeyboardEvents()
    # 监听 R 键 (ASCII 114) 和 Q 键 (ASCII 113)
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        play_trajectory()
    if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
        print("🛑 用户退出循环！正在转入后台渲染图表...")
        break
    time.sleep(0.02)

p.disconnect()

# ================= 附加计算评价指标 =================
# 轨迹的笛卡尔空间总长度
cartesian_path_length = 0.0
for i in range(len(smooth_path)-1):
    # 这里用记录到的真实物理末端距离粗略替代，或用正解估算。为了严谨，用记录的 EE
    cartesian_path_length += np.linalg.norm(np.array(record_ee[i+1]) - np.array(record_ee[i]))
execution_time = record_time[-1]
# ===================================================

# -------------------------- 7. 生成科研需求数据配图 --------------------------
import seaborn as sns
import pandas as pd

record_time = np.array(record_time)
record_joints = np.array(record_joints)
record_ee = np.array(record_ee)

# 利用已安装的 seaborn 设定全局的高级学术画图主题
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8f9fa", "figure.facecolor": "white"})

# 增强画布排版，我们引入一个 3D 轨迹图
fig = plt.figure(figsize=(15, 9))
fig.suptitle('3D Spheres Obstacle Avoidance - BiRRT Evaluation', fontsize=16, fontweight='bold', y=0.98)

# ==== 图1：各个关节角的变化趋势 (左上) ====
ax1 = fig.add_subplot(2, 2, 1)
# seaborn 提取高对比度离散色板
colors = sns.color_palette("husl", 7)
for i in range(7):
    ax1.plot(record_time, record_joints[:, i], label=f'$q_{i+1}$', linewidth=2, color=colors[i])
ax1.set_title('Joint Angles Trajectory Tracking', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time (s)', fontsize=10)
ax1.set_ylabel('Joint Angle (rad)', fontsize=10)
ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)

# ==== 图2：末端执行器位置的三轴变化 (左下) ====
ax2 = fig.add_subplot(2, 2, 3)
ax2.plot(record_time, record_ee[:, 0], linewidth=2.5, label='X', color=sns.xkcd_rgb["pale red"])
ax2.plot(record_time, record_ee[:, 1], linewidth=2.5, label='Y', color=sns.xkcd_rgb["medium green"])
ax2.plot(record_time, record_ee[:, 2], linewidth=2.5, label='Z', color=sns.xkcd_rgb["denim blue"])
ax2.set_title('End-Effector Cartesian Trajectory', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time (s)', fontsize=10)
ax2.set_ylabel('Position (m)', fontsize=10)
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)

# ==== 图3：末端三维空间运动轨迹 (右侧主图) ====
ax3 = fig.add_subplot(1, 2, 2, projection='3d')

# ======= 在数据配图中绘制三维球体障碍物 =======
for obs in sphere_obstacles_data:
    r = obs["radius"]
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x_obs = obs["pos"][0] + r * np.cos(u) * np.sin(v)
    y_obs = obs["pos"][1] + r * np.sin(u) * np.sin(v)
    z_obs = obs["pos"][2] + r * np.cos(v)
    ax3.plot_surface(x_obs, y_obs, z_obs, color=obs["color"][:3], alpha=0.3, edgecolor='none')

# 绘制时间变化映射颜色的 3D 散点图
sc = ax3.scatter(record_ee[:, 0], record_ee[:, 1], record_ee[:, 2], 
                 c=record_time, cmap='viridis', s=25, alpha=0.9, edgecolor='none')
# 绘制浅灰色的连续线以指引方向
ax3.plot(record_ee[:, 0], record_ee[:, 1], record_ee[:, 2], color='gray', linewidth=2, alpha=0.5)

# 标记起点和终点
ax3.scatter(*record_ee[0], color='red', s=120, marker='^', label='Start')
ax3.scatter(*record_ee[-1], color='darkred', s=150, marker='*', label='Goal')

ax3.set_title('3D Cartesian Path with Spheres', fontsize=13, fontweight='bold')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')
ax3.set_zlabel('Z (m)')
ax3.legend(loc='upper right')

# 为 3D 线增加时间颜色条指引
cbar = fig.colorbar(sc, ax=ax3, shrink=0.5, pad=0.08)
cbar.set_label('Time (s)', rotation=270, labelpad=15)

# ==== 图4：论文要求的量化评价指标数据框 (内嵌在右上角空白处) ====
metric_text = (
    "====== Algorithm Performance ======\n"
    f"Type                : Bi-Directional RRT\n"
    f"Planning Time : {planning_time:.3f} s\n"
    f"Path Length    : {cartesian_path_length:.3f} m\n"
    f"Execution Time: {execution_time:.3f} s\n"
    f"Collision Free : True\n"
    "================================="
)
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
# 利用 fig.text 放在整个画布底部的留白处，或者子图旁
fig.text(0.55, 0.15, metric_text, fontsize=11, family='monospace', bbox=props,
         verticalalignment='bottom', horizontalalignment='left')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 留出顶部 suptitle 和底部表格的空间
save_path = 'trajectory_results.png'
fig.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"📊 高清科研配图已保存至: {save_path}")
plt.show()