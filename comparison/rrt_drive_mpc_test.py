"""
RRT驱动的MPC快速测试脚本

测试策略：
1. 使用RRT规划的路径点作为"理想轨迹"
2. 车辆直接沿着RRT路径点移动（模拟完美路径跟踪）
3. 在每个位置调用MPC，记录MPC给出的控制指令
4. 快速走完整个流程，观察MPC在每个位置的反应行为

这样可以快速发现MPC在哪些位置有问题，而不需要等待实际仿真。
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MPC'))

import numpy as np
import math
import json
from datetime import datetime

from shared_config import (
    MAP_SIZE, START_POS, GOAL_POS, START_YAW,
    STATIC_OBSTACLES, VEHICLE_RADIUS, GOAL_TOLERANCE,
    MAX_SPEED, MIN_SPEED, VEHICLE_LENGTH, VEHICLE_WIDTH
)

# Patch MPC Config before importing MPC_RRT
import MPC.Config as mpc_cfg
mpc_cfg.goal = GOAL_POS
mpc_cfg.vehicle_length = VEHICLE_LENGTH
mpc_cfg.vehicle_width = VEHICLE_WIDTH
mpc_cfg.vehicle_radius = VEHICLE_RADIUS
mpc_cfg.lf = VEHICLE_LENGTH * 0.5
mpc_cfg.lr = VEHICLE_LENGTH * 0.5

from MPC.MPC_RRT import MPC_RRT_Controller, create_mpc_rrt_controller


def get_mpc_obstacles():
    """Convert shared obstacles to MPC format."""
    obstacles = []
    for obs in STATIC_OBSTACLES:
        obstacles.append({
            'position': obs['position'],
            'type': 'circle',
            'radius': obs['radius']
        })
    return obstacles


def compute_min_obstacle_distance(pos, obstacles):
    """Compute minimum distance to any obstacle."""
    min_dist = float('inf')
    closest_idx = -1
    for i, obs in enumerate(obstacles):
        ox, oy = obs['position']
        dist = math.hypot(pos[0] - ox, pos[1] - oy) - obs['radius'] - VEHICLE_RADIUS
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    return min_dist, closest_idx


def interpolate_path(waypoints, step_size=0.5):
    """
    沿着waypoints插值，生成更密集的路径点
    """
    if len(waypoints) < 2:
        return waypoints
    
    dense_path = []
    for i in range(len(waypoints) - 1):
        wp1 = waypoints[i]
        wp2 = waypoints[i + 1]
        
        dx = wp2.x - wp1.x
        dy = wp2.y - wp1.y
        dist = math.hypot(dx, dy)
        
        if dist < 0.01:
            continue
        
        num_steps = max(1, int(dist / step_size))
        for j in range(num_steps):
            t = j / num_steps
            x = wp1.x + t * dx
            y = wp1.y + t * dy
            yaw = math.atan2(dy, dx)
            dense_path.append({'x': x, 'y': y, 'yaw': yaw})
    
    # Add final point
    final = waypoints[-1]
    dense_path.append({'x': final.x, 'y': final.y, 'yaw': final.yaw})
    
    return dense_path


def run_rrt_driven_mpc_test():
    """
    使用RRT路径直接驱动，测试MPC在每个位置的反应
    """
    print("=" * 70)
    print("RRT驱动的MPC快速测试")
    print("=" * 70)
    print(f"地图: {MAP_SIZE}x{MAP_SIZE}")
    print(f"起点: {START_POS}, 终点: {GOAL_POS}")
    print(f"障碍物: {len(STATIC_OBSTACLES)} 个")
    print("=" * 70)
    
    static_obstacles = get_mpc_obstacles()
    
    # 打印障碍物位置
    print("\n障碍物列表:")
    for i, obs in enumerate(static_obstacles):
        print(f"  obs{i}: ({obs['position'][0]:.1f}, {obs['position'][1]:.1f}) r={obs['radius']:.1f}")
    
    # 初始化MPC+RRT控制器
    initial_state = [START_POS[0], START_POS[1], 1.0, START_YAW]
    
    print("\n[1] 规划RRT路径...")
    controller = create_mpc_rrt_controller(
        start_state=initial_state,
        goal=GOAL_POS,
        static_obstacles=static_obstacles,
        map_size=MAP_SIZE,
        horizon=20,
        dt=0.2,
        waypoint_tolerance=2.0,
        lookahead_distance=4.0,
        replan_on_deviation=15.0
    )
    
    stats = controller.get_planning_stats()
    print(f"  RRT规划: {'成功' if stats['planning_success'] else '失败'}")
    print(f"  路径点数: {stats['num_waypoints']}")
    print(f"  路径长度: {stats['path_length']:.2f}m")
    
    if not stats['planning_success']:
        print("RRT规划失败!")
        return
    
    # 打印waypoints
    print("\n[2] RRT路径点:")
    for i, wp in enumerate(controller.waypoints):
        print(f"  WP{i}: ({wp.x:.2f}, {wp.y:.2f})")
    
    # 打印预编译的solver targets
    print("\n[3] 预编译的Solver目标:")
    for i, (target, wp_idx) in enumerate(zip(controller.solver_targets, controller.solver_waypoint_indices)):
        print(f"  Solver{i}: waypoint{wp_idx} -> ({target[0]:.2f}, {target[1]:.2f})")
    
    # 插值生成密集路径
    dense_path = interpolate_path(controller.waypoints, step_size=1.0)
    print(f"\n[4] 插值后路径点数: {len(dense_path)}")
    
    # 沿着RRT路径测试MPC反应
    print("\n[5] 沿RRT路径测试MPC反应...")
    print("-" * 100)
    print(f"{'Step':>4} | {'Position':^15} | {'Solver':>6} | {'Target':^15} | {'Steer':>8} | {'Accel':>8} | {'Status':>6} | {'ObsDist':>8} | {'Notes'}")
    print("-" * 100)
    
    test_log = []
    velocity = 2.0  # 假设速度
    
    for i, point in enumerate(dense_path):
        x, y, yaw = point['x'], point['y'], point['yaw']
        
        # 构造状态
        cur_state = [x, y, velocity, yaw]
        
        # 调用MPC（开启debug拿到完整参数）
        try:
            control, status, debug_info = controller.solve(cur_state, debug=True)
        except Exception as e:
            print(f"Step {i}: MPC solve error: {e}")
            control = np.array([0.0, 0.0])
            status = -1
            debug_info = {}
        
        # 计算障碍物距离
        min_obs_dist, closest_idx = compute_min_obstacle_distance((x, y), static_obstacles)

        # RRT与MPC目标的差异
        mpc_target = debug_info.get('mpc_target', (0, 0))
        delta_to_mpc_target = math.hypot(x - mpc_target[0], y - mpc_target[1]) if mpc_target else None
        solver_idx = debug_info.get('solver_idx', -1)

        # 记录更多MPC内部信息
        mpc_debug = debug_info.get('mpc_debug', {})
        avoidance_hint = mpc_debug.get('avoidance_hint') if isinstance(mpc_debug, dict) else None

        notes = []
        if min_obs_dist < 2.0:
            notes.append(f"CLOSE_OBS({closest_idx})")
        if status != 0:
            notes.append("FAIL")
        if abs(control[0]) > 0.3:
            notes.append("BIG_STEER")
        if control[1] < -1.0:
            notes.append("BRAKE")
        if delta_to_mpc_target and delta_to_mpc_target > 3.0:
            notes.append(f"RRT_MPC_DIVERGE({delta_to_mpc_target:.1f}m)")
        
        # 最近RRT waypoint索引（用于定位分歧点）
        nearest_wp_idx = 0
        nearest_wp_dist = float('inf')
        for idx, wp in enumerate(controller.waypoints):
            d = math.hypot(x - wp.x, y - wp.y)
            if d < nearest_wp_dist:
                nearest_wp_dist = d
                nearest_wp_idx = idx
        
        log_entry = {
            'step': i,
            'rrt_pos': {'x': x, 'y': y, 'yaw': yaw},
            'nearest_wp_idx': nearest_wp_idx,
            'nearest_wp_dist': nearest_wp_dist,
            'solver_idx': solver_idx,
            'mpc_target': mpc_target,
            'delta_to_mpc_target': delta_to_mpc_target,
            'avoidance_hint': avoidance_hint,
            'steer': float(control[0]),
            'accel': float(control[1]),
            'status': status,
            'min_obs_dist': min_obs_dist,
            'closest_obs': closest_idx,
            'mpc_debug': mpc_debug
        }
        test_log.append(log_entry)
        
        # 打印关键信息（每隔几步或有问题时打印）
        should_print = (i % 5 == 0) or (min_obs_dist < 2.0) or (status != 0) or (notes)
        
        if should_print:
            target_str = f"({mpc_target[0]:.1f},{mpc_target[1]:.1f})" if mpc_target else "N/A"
            notes_str = ",".join(notes) if notes else ""
            print(f"{i:4d} | ({x:6.2f},{y:6.2f}) | {solver_idx:6d} | {target_str:^15} | {control[0]:8.4f} | {control[1]:8.4f} | {status:6d} | {min_obs_dist:8.3f} | {notes_str}")
    
    print("-" * 100)
    
    # 统计分析
    print("\n[6] 统计分析:")
    
    # solver切换统计
    solver_switches = []
    prev_solver = -1
    for entry in test_log:
        if entry['solver_idx'] != prev_solver:
            pos = entry.get('rrt_pos', {})
            solver_switches.append((entry['step'], entry['solver_idx'], pos.get('x', 0.0), pos.get('y', 0.0)))
            prev_solver = entry['solver_idx']
    
    print("\n  Solver切换记录:")
    for step, solver, x, y in solver_switches:
        print(f"    Step {step}: -> Solver{solver} at ({x:.2f}, {y:.2f})")
    
    # 障碍物接近统计
    close_obs_steps = [e for e in test_log if e['min_obs_dist'] < 2.0]
    print(f"\n  接近障碍物的步数: {len(close_obs_steps)}/{len(test_log)}")
    if close_obs_steps:
        print("  详细:")
        for e in close_obs_steps[:10]:  # 只显示前10个
            pos = e.get('rrt_pos', {})
            print(f"    Step {e['step']}: pos=({pos.get('x',0.0):.2f},{pos.get('y',0.0):.2f}) obs_dist={e['min_obs_dist']:.3f} obs_idx={e['closest_obs']}")
    
    # 失败统计
    fail_steps = [e for e in test_log if e['status'] != 0]
    print(f"\n  MPC求解失败步数: {len(fail_steps)}/{len(test_log)}")
    
    # 控制指令统计
    steers = [e['steer'] for e in test_log]
    accels = [e['accel'] for e in test_log]
    print(f"\n  转向统计: min={min(steers):.4f}, max={max(steers):.4f}, mean={np.mean(steers):.4f}")
    print(f"  加速统计: min={min(accels):.4f}, max={max(accels):.4f}, mean={np.mean(accels):.4f}")
    
    # 保存日志
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(log_dir, f"rrt_drive_test_{timestamp}.json")
    
    log_data = {
        'metadata': {
            'test_type': 'rrt_driven_mpc_test',
            'timestamp': timestamp,
            'map_size': MAP_SIZE,
            'start': list(START_POS),
            'goal': list(GOAL_POS),
            'num_obstacles': len(STATIC_OBSTACLES),
            'rrt_waypoints': [(wp.x, wp.y) for wp in controller.waypoints],
            'solver_targets': controller.solver_targets
        },
        'path_points': len(dense_path),
        'solver_switches': solver_switches,
        'close_obstacle_count': len(close_obs_steps),
        'failure_count': len(fail_steps),
        'test_log': test_log
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n[7] 日志已保存: {log_file}")
    print("=" * 70)
    
    return test_log


def analyze_mpc_behavior_at_obstacles():
    """
    专门分析MPC在障碍物附近的行为
    """
    print("\n" + "=" * 70)
    print("障碍物附近MPC行为分析")
    print("=" * 70)
    
    static_obstacles = get_mpc_obstacles()
    
    # 初始化控制器
    initial_state = [START_POS[0], START_POS[1], 1.0, START_YAW]
    controller = create_mpc_rrt_controller(
        start_state=initial_state,
        goal=GOAL_POS,
        static_obstacles=static_obstacles,
        map_size=MAP_SIZE,
        horizon=20,
        dt=0.2
    )
    
    if not controller.planning_success:
        print("RRT规划失败!")
        return
    
    # 找到路径上靠近障碍物的点
    print("\n分析RRT路径上每个点与障碍物的关系:")
    print("-" * 80)
    
    for i, wp in enumerate(controller.waypoints):
        min_dist, closest_idx = compute_min_obstacle_distance((wp.x, wp.y), static_obstacles)
        
        if min_dist < 5.0:  # 5m内有障碍物
            obs = static_obstacles[closest_idx]
            
            # 测试MPC在这个位置的反应
            yaw = wp.yaw if hasattr(wp, 'yaw') else 0
            state = [wp.x, wp.y, 2.0, yaw]
            
            control, status, debug = controller.solve(state, debug=False)
            
            print(f"WP{i}: ({wp.x:.2f}, {wp.y:.2f})")
            print(f"  最近障碍物: obs{closest_idx} at ({obs['position'][0]:.1f}, {obs['position'][1]:.1f}), dist={min_dist:.3f}m")
            print(f"  MPC控制: steer={control[0]:.4f}, accel={control[1]:.4f}, status={status}")
            print(f"  当前solver: {debug.get('solver_idx', -1)}, target={debug.get('mpc_target', 'N/A')}")
            print()


if __name__ == "__main__":
    # 运行主测试
    test_log = run_rrt_driven_mpc_test()
    
    # 运行障碍物分析
    analyze_mpc_behavior_at_obstacles()
