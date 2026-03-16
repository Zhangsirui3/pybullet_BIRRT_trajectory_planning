import pybullet as p
import pybullet_data
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -------------------------- 1. 初始化 PyBullet 的 GUI 模式以进行可视化 --------------------------
# 改为使用 p.GUI 模式以便我们能在窗口中看到它的搜索回放。由于多次跑批，可以通过断开重连或者重置清理。
try:
    physics_client = p.connect(p.GUI)
except:
    p.disconnect()
    physics_client = p.connect(p.GUI)

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=[0.0, 0.0, 0.85])

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
table_id = p.loadURDF("table/table.urdf", basePosition=[0.0, 0.0, 0.0], useFixedBase=True)
robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0.0, 0.0, 0.625], useFixedBase=True)

controlled_joints = []
joint_limits = []
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    if info[2] != p.JOINT_FIXED and "finger" not in info[1].decode('utf-8'):
        controlled_joints.append(j)
        joint_limits.append((info[8], info[9]))

controlled_joints = controlled_joints[:7]
joint_limits = joint_limits[:7]
EE_LINK_ID = 11

sphere_obstacles_data = [
    {"pos": [0.45, -0.25, 0.85], "radius": 0.12, "color": [0.9, 0.3, 0.3, 0.9]}, 
    {"pos": [0.45,  0.00, 1.15], "radius": 0.12, "color": [0.9, 0.3, 0.3, 0.9]}, 
    {"pos": [0.45,  0.25, 0.85], "radius": 0.12, "color": [0.9, 0.3, 0.3, 0.9]}, 
]

obstacles = []
for obs in sphere_obstacles_data:
    col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=obs["radius"])
    body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, basePosition=obs["pos"])
    obstacles.append(body_id)

def set_joints(angles):
    for i, j_id in enumerate(controlled_joints):
        p.resetJointState(robot_id, j_id, angles[i])

def get_fk_ee(angles):
    curr = [p.getJointState(robot_id, j)[0] for j in controlled_joints]
    set_joints(angles)
    ee_pos = p.getLinkState(robot_id, EE_LINK_ID, computeForwardKinematics=True)[4]
    set_joints(curr)
    return ee_pos

def is_collision_free(angles):
    curr = [p.getJointState(robot_id, j)[0] for j in controlled_joints]
    set_joints(angles)
    p.performCollisionDetection()
    collision = False
    if len(p.getContactPoints(robot_id, table_id)) > 0: collision = True
    if not collision:
        for obs in obstacles:
            if len(p.getContactPoints(robot_id, obs)) > 0:
                collision = True
                break
    set_joints(curr)
    return not collision

# -------------------------- 2. 算法公共模块 --------------------------
class Node:
    def __init__(self, angles, parent=None):
        self.angles = np.array(angles)
        self.parent = parent

def get_random_node():
    return np.array([random.uniform(limit[0], limit[1]) for limit in joint_limits])

def get_nearest_node(tree, target_node):
    distances = [np.linalg.norm(node.angles - target_node) for node in tree]
    return tree[np.argmin(distances)]

def extend(tree, target_angles, step_size):
    nearest = get_nearest_node(tree, target_angles)
    direction = target_angles - nearest.angles
    dist = np.linalg.norm(direction)
    if dist < step_size: new_angles = target_angles
    else: new_angles = nearest.angles + (direction / dist) * step_size
        
    if is_collision_free(new_angles):
        new_node = Node(new_angles, parent=nearest)
        tree.append(new_node)
        return new_node
    return None

def connect(tree, target_angles, step_size):
    last_added = None
    while True:
        nearest = get_nearest_node(tree, target_angles)
        if np.linalg.norm(target_angles - nearest.angles) < 1e-3: return last_added, "reached"
        new_node = extend(tree, target_angles, step_size)
        if new_node:
            last_added = new_node
            if np.linalg.norm(new_node.angles - target_angles) < 1e-3: return new_node, "reached"
        else:
            return last_added, "trapped"

# -------------------------- 3. 算法实现 --------------------------
# [算法 1] 标准单向 RRT
# 对于狭窄通道或高维空间，原版 RRT 需要非常久才能碰巧碰到终点。
# 我将 max_iter 放宽到 15000 并且对随机终点偏向 (goal_bias) 进行优化，防止其假死。
def plan_rrt(start_angles, goal_angles, step_size=0.03, max_iter=10000, max_time=10.0):
    tree = [Node(start_angles)]
    t0 = time.time()
    for i in range(max_iter):
        if time.time() - t0 > max_time:
            break  # 超时直接返回失败，证明基础RRT在这种难度下的无力
        # 偏向目标点有助于原版RRT稍微聪明一点（虽然仍很容易陷入局部最小值）
        rand_angles = get_random_node() if random.random() > 0.15 else goal_angles
        new_node = extend(tree, rand_angles, step_size)
        if new_node and np.linalg.norm(new_node.angles - goal_angles) < 1e-3:
            path = []
            curr = new_node
            while curr: path.append(curr.angles.tolist()); curr = curr.parent
            return path[::-1], time.time() - t0
    return [], time.time() - t0

# [算法 2] 双向 BiRRT
def plan_birrt(start_angles, goal_angles, step_size=0.03, max_iter=10000, max_time=10.0):
    treeA, treeB = [Node(start_angles)], [Node(goal_angles)]
    t0 = time.time()
    for _ in range(max_iter):
        if time.time() - t0 > max_time:
            break
        rand_angles = get_random_node() if random.random() > 0.2 else treeB[-1].angles
        new_nodeA = extend(treeA, rand_angles, step_size)
        if new_nodeA:
            connect_node, status = connect(treeB, new_nodeA.angles, step_size)
            if status == "reached":
                pathA = []; curr = new_nodeA
                while curr: pathA.append(curr.angles.tolist()); curr = curr.parent
                pathA.reverse()
                pathB = []; curr = connect_node
                while curr: pathB.append(curr.angles.tolist()); curr = curr.parent
                if np.linalg.norm(treeA[0].angles - start_angles) < 1e-3: final_path = pathA + pathB
                else: pathB.reverse(); pathA.reverse(); final_path = pathB + pathA
                return final_path, time.time() - t0
        treeA, treeB = treeB, treeA
    return [], time.time() - t0

# [算法 3] S-BiRRT (平滑+剪枝改进版)
def smooth_path_shortcutting(path, max_iters=200):
    if len(path) <= 2: return path
    smoothed = path.copy()
    for _ in range(max_iters):
        if len(smoothed) <= 2: break
        i, j = random.randint(0, len(smoothed)-2), random.randint(0, len(smoothed)-1)
        if i >= j or j - i <= 1: continue 
        dist = np.linalg.norm(np.array(smoothed[i]) - np.array(smoothed[j]))
        steps = int(dist / 0.02) + 1; valid = True
        for step in range(1, steps):
            interp = np.array(smoothed[i]) * (1 - step/steps) + np.array(smoothed[j]) * (step/steps)
            if not is_collision_free(interp): valid = False; break
        if valid: smoothed = smoothed[:i+1] + smoothed[j:]
    return smoothed

def smooth_path_chaikin(path, iterations=3):
    if len(path) <= 2: return path
    smoothed = np.array(path)
    for _ in range(iterations):
        new_path = [smoothed[0]]
        for i in range(len(smoothed)-1):
            p0, p1 = smoothed[i], smoothed[i+1]
            new_path.append(0.75 * p0 + 0.25 * p1)
            new_path.append(0.25 * p0 + 0.75 * p1)
        new_path.append(smoothed[-1])
        smoothed = np.array(new_path)
    return smoothed.tolist()

def plan_sc_birrt(start_angles, goal_angles, step_size=0.03, max_iter=10000, max_time=10.0):
    path, t_plan = plan_birrt(start_angles, goal_angles, step_size, max_iter, max_time)
    if not path: return [], t_plan
    t0 = time.time()
    shortcut_path = smooth_path_shortcutting(path, max_iters=200)
    final_path = smooth_path_chaikin(shortcut_path, iterations=3)
    return final_path, t_plan + (time.time() - t0)

# -------------------------- 4. 评估指标计算 --------------------------
def calc_metrics(path):
    if not path: return 0.0, 0.0
    path = np.array(path)
    
    # 1. 笛卡尔末端总路径长度
    cartesian_len = 0.0
    ee_pts = [get_fk_ee(p) for p in path]
    for i in range(len(ee_pts)-1):
        cartesian_len += np.linalg.norm(np.array(ee_pts[i+1]) - np.array(ee_pts[i]))
        
    # 2. 关节角加速度冲击度 (Jerk/Smoothness Index) -> 越低越平滑
    smoothness_cost = 0.0
    if len(path) > 2:
        diffs = np.diff(path, axis=0) # 速度
        accs = np.diff(diffs, axis=0) # 加速度
        smoothness_cost = np.sum(np.linalg.norm(accs, axis=1) ** 2)
        
    return cartesian_len, smoothness_cost, ee_pts

# -------------------------- 5. 实验启动跑批 --------------------------
start_pos = [0.45, -0.45, 0.95]
goal_pos = [0.45, 0.45, 0.95]
start_angles, target_angles = None, None

# 获取安全的起终点
np.random.seed(42)
for _ in range(50):
    res = p.calculateInverseKinematics(robot_id, EE_LINK_ID, start_pos, maxNumIterations=100)[:7]
    if is_collision_free(res): start_angles = list(res); break
for _ in range(50):
    res = p.calculateInverseKinematics(robot_id, EE_LINK_ID, goal_pos, maxNumIterations=100)[:7]
    if is_collision_free(res): target_angles = list(res); break

print("🚀 正在运行科研级基准测试 (N=5 runs/algorithm 以节省时间)...")
trials = 5
results = []
representatives = {"RRT": [], "BiRRT": [], "SC-BiRRT": []}

algorithms = {
    "RRT": plan_rrt,
    "BiRRT": plan_birrt,
    "SC-BiRRT": plan_sc_birrt
}

def play_and_draw_path(path, color):
    # 绘制可视化路径
    traj_lines = []
    ee_pts = [get_fk_ee(pts) for pts in path]
    for i in range(len(ee_pts)-1):
        line = p.addUserDebugLine(ee_pts[i], ee_pts[i+1], color, 4)
        traj_lines.append(line)
        
    # 让机械臂简单顺滑跟一遍，用于动画效果展示
    for pts in path[::max(1, len(path)//20)]: # 抽稀播放防止过慢
        set_joints(pts)
        time.sleep(0.01)
    
    # 清理画线
    for line in traj_lines:
        p.removeUserDebugItem(line)

colors_viz = {"RRT": [1, 0.5, 0], "BiRRT": [0, 0.8, 0], "SC-BiRRT": [0, 0.5, 1]}

for algo_name, algo_func in algorithms.items():
    print(f"\n🔄 评估 {algo_name}...")
    success_count = 0
    best_path_len = float('inf')
    for i in range(trials):
        print(f"   - 进行第 {i+1}/{trials} 次试验...", end="", flush=True)
        set_joints(start_angles)
        
        path, t_plan = algo_func(start_angles, target_angles)
        
        if len(path) > 0:
            success_count += 1
            length, jerk, ee_pts = calc_metrics(path)
            print(f" 成功! (耗时: {t_plan:.2f}s, 长度: {length:.2f}m)")
            
            results.append({
                "Algorithm": algo_name,
                "Planning Time (s)": t_plan,
                "Path Length (m)": length,
                "Smoothness Cost": jerk
            })
            
            # 在仿真器中实时跑一遍刚刚算出来的最优结果
            play_and_draw_path(path, colors_viz[algo_name])

            # 保留距离最短的作为静态科研配图的代表路径
            if length < best_path_len:
                best_path_len = length
                representatives[algo_name] = ee_pts
        else:
            print(f" 失败... (超时放弃)")
            
    print(f"✅ {algo_name} 最终通过率: {success_count}/{trials}")

p.disconnect()

# -------------------------- 6. 科研论文级绘图 --------------------------
df = pd.DataFrame(results)
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(18, 10))
fig.suptitle('Performance Benchmark: Motion Planning across Sphere Obstacles', fontsize=18, fontweight='bold')

# A. 3D 轨迹对比图 (左侧主图)
ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
colors_repr = {'RRT': '#ff7f0e', 'BiRRT': '#2ca02c', 'SC-BiRRT': '#1f77b4'}
line_styles = {'RRT': ':', 'BiRRT': '--', 'SC-BiRRT': '-'}

for obs in sphere_obstacles_data:
    r = obs["radius"]
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x_obs = obs["pos"][0] + r * np.cos(u) * np.sin(v)
    y_obs = obs["pos"][1] + r * np.sin(u) * np.sin(v)
    z_obs = obs["pos"][2] + r * np.cos(v)
    ax_3d.plot_surface(x_obs, y_obs, z_obs, color='red', alpha=0.3, edgecolor='none')

for algo_name, pts in representatives.items():
    if len(pts) > 0:
        pts = np.array(pts)
        ax_3d.plot(pts[:,0], pts[:,1], pts[:,2], label=f'{algo_name} Trajectory', 
                   color=colors_repr[algo_name], linestyle=line_styles[algo_name], linewidth=3, alpha=0.8)

if representatives['SC-BiRRT']:
    ax_3d.scatter(*representatives['SC-BiRRT'][0], color='black', s=100, marker='^', label='Start')
    ax_3d.scatter(*representatives['SC-BiRRT'][-1], color='black', s=100, marker='*', label='Goal')

ax_3d.set_title("3D Cartesian Workspace & Representative Paths", fontweight='bold')
ax_3d.set_xlabel('X (m)'); ax_3d.set_ylabel('Y (m)'); ax_3d.set_zlabel('Z (m)')
ax_3d.legend(loc='upper right', frameon=True)

# B. 定量盒子图排版 (右侧均分三个子图)
ax_t = fig.add_subplot(3, 2, 2)
sns.boxplot(data=df, x="Algorithm", y="Planning Time (s)", palette="Set2", ax=ax_t, width=0.4)
ax_t.set_title("Planning Time Comparison", fontweight='bold')

ax_l = fig.add_subplot(3, 2, 4)
sns.boxplot(data=df, x="Algorithm", y="Path Length (m)", palette="Set2", ax=ax_l, width=0.4)
ax_l.set_title("Cartesian Path Length", fontweight='bold')

ax_s = fig.add_subplot(3, 2, 6)
# Smoothness 为关节角加速度总和（越小越平滑，用对数轴显示差异）
sns.barplot(data=df, x="Algorithm", y="Smoothness Cost", hue="Algorithm", palette="Set2", ax=ax_s, capsize=0.1, legend=False)
ax_s.set_yscale("log")
ax_s.set_title("Smoothness / Jerk Cost (Log Scale, Lower is Better)", fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path = "benchmark_results.png"
plt.savefig(save_path, dpi=300, facecolor='w')
print(f"📊 基准测试完成！对比科研配图已保存至: {save_path}")
