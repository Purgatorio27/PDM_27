
import os
import json
import matplotlib.pyplot as plt
import glob
import numpy as np

def analyze():
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    
    # Find latest logs
    # exclude rrt from mpc search
    all_mpc_logs = glob.glob(os.path.join(log_dir, 'dynamic_mpc_*.json'))
    mpc_logs = [f for f in all_mpc_logs if '_rrt_' not in os.path.basename(f)]
    
    mpc_rrt_logs = glob.glob(os.path.join(log_dir, 'dynamic_mpc_rrt_*.json'))
    
    if not mpc_logs or not mpc_rrt_logs:
        print("Could not find both logs.")
        return
        
    latest_mpc = max(mpc_logs, key=os.path.getctime)
    latest_rrt = max(mpc_rrt_logs, key=os.path.getctime)
    
    print(f"Analyzing:\n  MPC: {os.path.basename(latest_mpc)}\n  MPC+RRT: {os.path.basename(latest_rrt)}")
    
    with open(latest_mpc) as f: mpc_data = json.load(f)
    with open(latest_rrt) as f: rrt_data = json.load(f)
    
    # print summary
    print("\n" + "="*60)
    print(f"{'METRIC':<25} {'MPC':<15} {'MPC+RRT*':<15}")
    print("="*60)
    
    def get_stats(data):
        steps = len(data['trajectory'])
        time = steps * 0.05 # DT
        path_len = 0
        traj = data['trajectory']
        for i in range(1, len(traj)):
            path_len += np.hypot(traj[i]['x'] - traj[i-1]['x'], traj[i]['y'] - traj[i-1]['y'])
        
        collisions = data.get('collisions', 0)
        success = False
        if traj:
            final_pos = (traj[-1]['x'], traj[-1]['y'])
            goal = data['metadata']['goal']
            if np.hypot(final_pos[0]-goal[0], final_pos[1]-goal[1]) < 1.5: # generous acceptance
                success = True
                
        return time, path_len, collisions, success

    m_time, m_len, m_col, m_succ = get_stats(mpc_data)
    r_time, r_len, r_col, r_succ = get_stats(rrt_data)
    
    print(f"{'Success':<25} {str(m_succ):<15} {str(r_succ):<15}")
    print(f"{'Time (s)':<25} {m_time:<15.2f} {r_time:<15.2f}")
    print(f"{'Path Length (m)':<25} {m_len:<15.2f} {r_len:<15.2f}")
    print(f"{'Collisions':<25} {m_col:<15} {r_col:<15}")
    print("="*60)
    
    # Plot Trajectories
    plt.figure(figsize=(10, 10))
    
    # Plot MPC
    mx = [p['x'] for p in mpc_data['trajectory']]
    my = [p['y'] for p in mpc_data['trajectory']]
    plt.plot(mx, my, 'r-', label='Pure MPC', linewidth=2)
    
    # Plot MPC+RRT
    rx = [p['x'] for p in rrt_data['trajectory']]
    ry = [p['y'] for p in rrt_data['trajectory']]
    plt.plot(rx, ry, 'b-', label='MPC + RRT*', linewidth=2)
    
    # Start/Goal
    start = mpc_data['metadata'].get('start', [2,2]) # fallback
    goal = mpc_data['metadata']['goal']
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'mx', markersize=10, label='Goal')
    
    # Plot Static Obstacles (from Config, approximating since not in log metadata fully)
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from shared_config import STATIC_OBSTACLES, MAP_SIZE
    for obs in STATIC_OBSTACLES:
        circle = plt.Circle(obs['position'], obs['radius'], color='k', fill=True, alpha=0.3)
        plt.gca().add_patch(circle)
        
    plt.xlim(0, MAP_SIZE)
    plt.ylim(0, MAP_SIZE)
    plt.title(f"Trajectory Comparison (Dynamic Obstacles)\nMPC Collisions: {m_col} | MPC+RRT Collisions: {r_col}")
    plt.legend()
    plt.grid(True)
    
    out_path = os.path.join(os.path.dirname(__file__), 'plots', 'dynamic_comparison.png')
    plt.savefig(out_path)
    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    analyze()
