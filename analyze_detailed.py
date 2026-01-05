#!/usr/bin/env python3
import json
import numpy as np
import sys
import os

# Find latest log
log_dir = 'logs'
logs = sorted([f for f in os.listdir(log_dir) if f.endswith('.json')])
if not logs:
    print("No logs found")
    sys.exit(1)
    
log_file = os.path.join(log_dir, logs[-1])
print(f"Analyzing: {log_file}")

with open(log_file, 'r') as f:
    data = json.load(f)

print('=== 基本统计 ===')
trajectory = data['trajectory']
controls = data['controls']
metadata = data['metadata']
print(f'总步数: {len(trajectory)}')
goal = metadata["goal"]
print(f'目标位置: {goal}')

# 分析轨迹
positions = [(s['x'], s['y']) for s in trajectory]
headings = [s['psi'] for s in trajectory]
velocities = [s['v'] for s in trajectory]
steerings = [c['control']['steering'] for c in controls]
accelerations = [c['control']['acceleration'] for c in controls]
solver_status = [c['solver']['status'] for c in controls]
distances_to_goal = [c['distances']['to_goal'] for c in controls]
distances_to_obs = [c['distances']['to_nearest_obstacle'] for c in controls]

# 起点和终点
print(f'\n起点: ({positions[0][0]:.2f}, {positions[0][1]:.2f})')
print(f'终点: ({positions[-1][0]:.2f}, {positions[-1][1]:.2f})')
final_dist = np.sqrt((positions[-1][0]-goal[0])**2 + (positions[-1][1]-goal[1])**2)
print(f'最终距离目标: {final_dist:.2f}m')

# 速度统计
print(f'\n=== 速度统计 ===')
print(f'平均速度: {np.mean(velocities):.2f} m/s')
print(f'最小速度: {np.min(velocities):.2f} m/s')
print(f'最大速度: {np.max(velocities):.2f} m/s')

# 转向统计
print(f'\n=== 转向统计 ===')
print(f'平均转向角: {np.degrees(np.mean(steerings)):.2f}°')
print(f'转向角标准差: {np.degrees(np.std(steerings)):.2f}°')
print(f'最大左转: {np.degrees(np.min(steerings)):.2f}°')
print(f'最大右转: {np.degrees(np.max(steerings)):.2f}°')

# 统计转向方向
left_turns = sum(1 for s in steerings if s < -0.01)
right_turns = sum(1 for s in steerings if s > 0.01)
straight = len(steerings) - left_turns - right_turns
print(f'左转次数: {left_turns}, 右转次数: {right_turns}, 直行: {straight}')

# 航向变化
heading_changes = [headings[i+1] - headings[i] for i in range(len(headings)-1)]
total_heading_change = sum(abs(h) for h in heading_changes)
print(f'\n=== 航向统计 ===')
print(f'总航向变化: {np.degrees(total_heading_change):.1f}° (绕圈={total_heading_change/(2*np.pi):.1f}圈)')
print(f'初始航向: {np.degrees(headings[0]):.1f}°')
print(f'最终航向: {np.degrees(headings[-1]):.1f}°')

# 目标方向 vs 实际航向
goal_angles = [np.arctan2(goal[1]-p[1], goal[0]-p[0]) for p in positions]
heading_errors = []
for i in range(len(positions)):
    err = goal_angles[i] - headings[i]
    err = np.arctan2(np.sin(err), np.cos(err))
    heading_errors.append(err)
print(f'平均航向误差: {np.degrees(np.mean(np.abs(heading_errors))):.1f}°')

# solver状态
success = sum(1 for s in solver_status if s == 0)
unique_status = set(solver_status)
print(f'\n=== Solver状态 ===')
print(f'成功率: {success}/{len(solver_status)} ({100*success/len(solver_status):.1f}%)')
print(f'状态类型: {unique_status}')

# 分析MPC预测
print(f'\n=== MPC预测分析 (前10步) ===')
mpc_predictions = data.get('mpc_predictions', [])
for i in range(min(10, len(controls))):
    c = controls[i]
    x, y = trajectory[i]['x'], trajectory[i]['y']
    psi = trajectory[i]['psi']
    v = trajectory[i]['v']
    steer = c['control']['steering']
    accel = c['control']['acceleration']
    goal_ang = np.arctan2(goal[1]-y, goal[0]-x)
    h_err = np.arctan2(np.sin(goal_ang-psi), np.cos(goal_ang-psi))
    dist_to_goal = c['distances']['to_goal']
    dist_to_obs = c['distances']['to_nearest_obstacle']
    
    # Check MPC prediction endpoint if available
    if i < len(mpc_predictions) and mpc_predictions[i]:
        mpc_pred = mpc_predictions[i]
        if 'predicted_x' in mpc_pred and len(mpc_pred['predicted_x']) > 0:
            pred_end_x = mpc_pred['predicted_x'][-1]
            pred_end_y = mpc_pred['predicted_y'][-1]
            pred_dist = np.sqrt((goal[0]-pred_end_x)**2 + (goal[1]-pred_end_y)**2)
            print(f'Step {i}: pos=({x:.1f},{y:.1f}) dist_goal={dist_to_goal:.1f}m dist_obs={dist_to_obs:.1f}m v={v:.2f} h_err={np.degrees(h_err):.0f}° steer={np.degrees(steer):.1f}° accel={accel:.2f} | MPC_end=({pred_end_x:.1f},{pred_end_y:.1f})')
        else:
            print(f'Step {i}: pos=({x:.1f},{y:.1f}) dist_goal={dist_to_goal:.1f}m dist_obs={dist_to_obs:.1f}m v={v:.2f} h_err={np.degrees(h_err):.0f}° steer={np.degrees(steer):.1f}° accel={accel:.2f}')
    else:
        print(f'Step {i}: pos=({x:.1f},{y:.1f}) dist_goal={dist_to_goal:.1f}m dist_obs={dist_to_obs:.1f}m v={v:.2f} h_err={np.degrees(h_err):.0f}° steer={np.degrees(steer):.1f}° accel={accel:.2f}')

# 检查是否有障碍物信息
print(f'\n=== 障碍物信息 ===')
static_obs = metadata.get('static_obstacles', [])
for i, obs in enumerate(static_obs):
    print(f'静态障碍物 {i}: pos={obs["position"]}, radius={obs["radius"]}')

# 计算与障碍物的最小距离
print(f'最小障碍物距离: {min(distances_to_obs):.2f}m')
print(f'平均障碍物距离: {np.mean(distances_to_obs):.2f}m')

# 统计在不同距离范围的步数
close_steps = sum(1 for d in distances_to_obs if d < 2.0)
print(f'距离障碍物<2m的步数: {close_steps} ({100*close_steps/len(distances_to_obs):.1f}%)')

# 分析转向模式
print(f'\n=== 转向模式分析 ===')
# 检查是否持续向一个方向转
steer_sign_changes = 0
for i in range(1, len(steerings)):
    if steerings[i] * steerings[i-1] < 0:
        steer_sign_changes += 1
print(f'转向方向切换次数: {steer_sign_changes}')

# 连续同向转向的最长序列
max_same_dir = 0
current_same = 1
for i in range(1, len(steerings)):
    if steerings[i] * steerings[i-1] > 0:
        current_same += 1
        max_same_dir = max(max_same_dir, current_same)
    else:
        current_same = 1
print(f'最长连续同向转向: {max_same_dir}步')

# 分析是否在绕圈
print(f'\n=== 绕圈检测 ===')
# 检查位置是否回到原点附近
for i in range(100, len(positions), 100):
    dist_from_start = np.sqrt((positions[i][0]-positions[0][0])**2 + (positions[i][1]-positions[0][1])**2)
    dist_to_goal_i = np.sqrt((positions[i][0]-goal[0])**2 + (positions[i][1]-goal[1])**2)
    print(f'Step {i}: 距起点 {dist_from_start:.1f}m, 距目标 {dist_to_goal_i:.1f}m')
