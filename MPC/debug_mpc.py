#!/usr/bin/env python3
import numpy as np
from MPC_core import MPC
from Obstacles import static_obstacles, dynamic_obstacles

# 初始化MPC
cur_dyn_obs_MPC = []
for d_obs in dynamic_obstacles:
    cur_dyn_obs_MPC.append({
        'trajectory': d_obs['trajectory'][0],
        'radius': d_obs['radius']
    })

mpc = MPC(static_obstacles, cur_dyn_obs_MPC, horizon=40, dt=0.25)

# 初始状态: 起点(0,0), 速度1m/s, 朝向45度(指向目标)
cur_state = np.array([0.0, 0.0, 1.0, np.radians(45)])

# 求解
control, status, debug = mpc.solve(cur_state, debug=True)

print('=== 完整预测轨迹 ===')
for i in range(mpc.N + 1):
    state = mpc.solver.get(i, 'x')
    if i < mpc.N:
        ctrl = mpc.solver.get(i, 'u')
        print(f'Step {i}: x={state[0]:.2f}, y={state[1]:.2f}, v={state[2]:.2f}, psi={np.degrees(state[3]):.1f}° | steer={np.degrees(ctrl[0]):.1f}°, accel={ctrl[1]:.2f}')
    else:
        print(f'Step {i}: x={state[0]:.2f}, y={state[1]:.2f}, v={state[2]:.2f}, psi={np.degrees(state[3]):.1f}° (terminal)')

print(f'\n=== 分析 ===')
print(f'Solver status: {status}')
print(f'Initial control: steer={np.degrees(control[0]):.1f}°, accel={control[1]:.2f}')

# 计算预测轨迹的总距离
total_dist = 0
for i in range(mpc.N):
    s1 = mpc.solver.get(i, 'x')
    s2 = mpc.solver.get(i+1, 'x')
    total_dist += np.sqrt((s2[0]-s1[0])**2 + (s2[1]-s1[1])**2)
print(f'Predicted total distance: {total_dist:.2f}m in {mpc.N*mpc.dt}s')

# 检查预测终点距目标的距离
final_state = mpc.solver.get(mpc.N, 'x')
goal = [20, 20]
final_dist_to_goal = np.sqrt((final_state[0]-goal[0])**2 + (final_state[1]-goal[1])**2)
print(f'Predicted final pos: ({final_state[0]:.2f}, {final_state[1]:.2f})')
print(f'Predicted final dist to goal: {final_dist_to_goal:.2f}m')

# 计算如果直线行驶会怎样
ideal_dist = 1.0 * 10  # v=1m/s for 10s = 10m
ideal_end = np.array([0, 0]) + ideal_dist * np.array([np.cos(np.radians(45)), np.sin(np.radians(45))])
print(f'\nIdeal straight line end: ({ideal_end[0]:.2f}, {ideal_end[1]:.2f})')
print(f'Ideal dist to goal: {np.sqrt((ideal_end[0]-20)**2 + (ideal_end[1]-20)**2):.2f}m')
