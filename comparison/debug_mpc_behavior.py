"""
简单的MPC行为调试测试
直接从起点模拟MPC控制，观察车辆轨迹
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MPC'))

import numpy as np
import math

from shared_config import (
    MAP_SIZE, START_POS, GOAL_POS, START_YAW,
    STATIC_OBSTACLES, VEHICLE_RADIUS,
    MAX_SPEED, MIN_SPEED, VEHICLE_LENGTH
)

# Patch MPC Config before importing
import MPC.Config as mpc_cfg
mpc_cfg.goal = GOAL_POS
mpc_cfg.vehicle_length = VEHICLE_LENGTH
mpc_cfg.vehicle_width = 1.0
mpc_cfg.vehicle_radius = VEHICLE_RADIUS
mpc_cfg.lf = VEHICLE_LENGTH * 0.5
mpc_cfg.lr = VEHICLE_LENGTH * 0.5

from MPC.MPC_RRT import create_mpc_rrt_controller


def get_mpc_obstacles():
    obstacles = []
    for obs in STATIC_OBSTACLES:
        obstacles.append({
            'position': obs['position'],
            'type': 'circle',
            'radius': obs['radius']
        })
    return obstacles


def vehicle_update(state, control, dt):
    """Kinematic bicycle model update."""
    x, y, v, psi = state
    delta, a = control
    LF = LR = VEHICLE_LENGTH * 0.5
    
    x_new = x + v * math.cos(psi) * dt
    y_new = y + v * math.sin(psi) * dt
    v_new = np.clip(v + a * dt, MIN_SPEED, MAX_SPEED)
    
    beta = math.atan((LR / (LF + LR)) * math.tan(delta))
    psi_new = psi + (v / LR) * math.sin(beta) * dt
    
    return np.array([x_new, y_new, v_new, psi_new])


def main():
    print("=" * 70)
    print("MPC行为调试测试 - 从起点直接模拟")
    print("=" * 70)
    
    static_obstacles = get_mpc_obstacles()
    
    print("\n障碍物:")
    for i, obs in enumerate(static_obstacles):
        print(f"  obs{i}: ({obs['position'][0]:.1f}, {obs['position'][1]:.1f}) r={obs['radius']:.1f}")
    
    # 初始化
    state = np.array([START_POS[0], START_POS[1], 1.0, START_YAW])
    dt = 0.05
    
    print(f"\n初始状态: pos=({state[0]:.2f}, {state[1]:.2f}) v={state[2]:.2f} yaw={np.degrees(state[3]):.1f}°")
    
    # 创建控制器
    print("\n创建MPC+RRT控制器...")
    controller = create_mpc_rrt_controller(
        start_state=state.tolist(),
        goal=GOAL_POS,
        static_obstacles=static_obstacles,
        map_size=MAP_SIZE,
        horizon=20,
        dt=0.2
    )
    
    if not controller.planning_success:
        print("RRT规划失败!")
        return
    
    print(f"RRT路径点数: {len(controller.waypoints)}")
    print(f"预编译solver数: {len(controller.solver_pool)}")
    
    # 打印前几个waypoints
    print("\n前5个RRT路径点:")
    for i, wp in enumerate(controller.waypoints[:5]):
        print(f"  WP{i}: ({wp.x:.2f}, {wp.y:.2f})")
    
    # 模拟运行
    print("\n" + "-" * 70)
    print("模拟运行 (前500步):")
    print("-" * 70)
    
    trajectory = []
    for step in range(500):
        # 获取avoidance_hint来调试
        hint = controller._get_avoidance_hint(state.tolist())
        
        # MPC求解
        control, status, debug = controller.solve(state.tolist(), debug=False)
        
        solver_idx = debug.get('solver_idx', -1)
        target = debug.get('mpc_target', (0, 0))
        
        # 每10步或有重要事件时打印
        if step % 10 == 0 or step < 5:
            hint_str = f"({hint[0]:.1f},{hint[1]:.1f})" if hint else "None"
            target_str = f"({target[0]:.1f},{target[1]:.1f})"
            print(f"Step {step:3d}: pos=({state[0]:6.2f},{state[1]:6.2f}) v={state[2]:.2f} | "
                  f"solver={solver_idx} target={target_str} hint={hint_str} | "
                  f"steer={control[0]:.3f} accel={control[1]:.3f}")
        
        trajectory.append({
            'step': step,
            'x': state[0], 'y': state[1], 'v': state[2],
            'solver': solver_idx,
            'hint': hint
        })
        
        # 更新状态
        state = vehicle_update(state, control, dt)
        
        # 检查是否到达终点
        if math.hypot(state[0] - GOAL_POS[0], state[1] - GOAL_POS[1]) < 1.0:
            print(f"\n*** 到达终点! step={step} ***")
            break
        
        # 检查是否出界
        if state[0] < 0 or state[1] < 0 or state[0] > MAP_SIZE or state[1] > MAP_SIZE:
            print(f"\n*** 出界! step={step} pos=({state[0]:.2f}, {state[1]:.2f}) ***")
            break
    
    print("-" * 70)
    print(f"最终位置: ({state[0]:.2f}, {state[1]:.2f})")
    print(f"距离终点: {math.hypot(state[0] - GOAL_POS[0], state[1] - GOAL_POS[1]):.2f}m")


if __name__ == "__main__":
    main()
