
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUMMARY_FILE = 'benchmark_summary.json'
LOG_DIR = 'logs'
OUTPUT_DIR = 'benchmark_plots'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_log_data(log_file_path):
    # If path from summary is absolute or relative, handle it.
    # The summary has 'logs/...' usually.
    if not os.path.exists(log_file_path):
        # Try prepending LOG_DIR if just name
        candidate = os.path.join(LOG_DIR, os.path.basename(log_file_path))
        if os.path.exists(candidate):
            log_file_path = candidate
        else:
            print(f"Warning: Log file {log_file_path} not found.")
            return None
            
    try:
        with open(log_file_path, 'r') as f:
            data = json.load(f)
            
        # Try to adapt to different log structures
        if 'summary' in data:
            summary = data['summary']
            collisions = summary.get('collision_count', 0)
            final_dist = summary.get('final_distance_to_goal', 999.0)
            success = final_dist < 2.0
            
            if 'total_time_s' in summary:
                time_taken = summary['total_time_s']
            elif 'total_execution_time_s' in summary:
                time_taken = summary['total_execution_time_s']
            else:
                time_taken = 0.0
        else:
            # Fallback for dynamic scripts (simple structure)
            collisions = data.get('collisions', 0)
            
            trajectory = data.get('trajectory', [])
            if trajectory:
                # Get Goal from metadata
                goal = data.get('metadata', {}).get('goal', None)
                if goal is None:
                    # Fallback if metadata missing (unlikely)
                    print(f"Warning: Goal missing in {log_file_path}")
                    final_dist = 999.0
                else:
                    last_pt = trajectory[-1]
                    # Check format
                    if isinstance(last_pt, dict):
                        dx = last_pt['x'] - goal[0]
                        dy = last_pt['y'] - goal[1]
                    else:
                        dx = last_pt[0] - goal[0]
                        dy = last_pt[1] - goal[1]
                    final_dist = np.hypot(dx, dy)
                
                # Assume 0.05s DT
                time_taken = len(trajectory) * 0.05
            else:
                final_dist = 999.0
                time_taken = 0.0
            
            success = final_dist < 2.0 # use 2.0m threshold

        return {
            'collisions': collisions,
            'success': success,
            'time': time_taken,
            'd_goal': final_dist
        }
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return None

def main():
    if not os.path.exists(SUMMARY_FILE):
        print(f"Summary file {SUMMARY_FILE} not found.")
        return

    with open(SUMMARY_FILE, 'r') as f:
        benchmark_runs = json.load(f)

    results = []
    
    for run in benchmark_runs:
        log_file = run['log_file']
        metrics = load_log_data(log_file)
        
        if metrics:
            entry = {
                'Map Size': run['map_size'],
                'Scenario': run['scenario'],
                'Algorithm': run['algorithm'],
                'Collisions': metrics['collisions'],
                'Success': metrics['success'],
                'Time (s)': metrics['time'],
                'Dist to Goal': metrics['d_goal']
            }
            results.append(entry)

    df = pd.DataFrame(results)
    
    # --- Plotting ---
    
    # 1. Collisions by Map Size and Algorithm
    # Separate Static and Dynamic
    
    scenarios = ['static', 'dynamic']
    
    for scenario in scenarios:
        subset = df[df['Scenario'] == scenario]
        if subset.empty:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Group bar chart
        # X axis: Map Size
        # Hue: Algorithm
        
        map_sizes = sorted(subset['Map Size'].unique())
        algos = sorted(subset['Algorithm'].unique())
        
        x = np.arange(len(map_sizes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, algo in enumerate(algos):
            algo_data = subset[subset['Algorithm'] == algo]
            # Ensure order matches map_sizes
            counts = []
            for m in map_sizes:
                val = algo_data[algo_data['Map Size'] == m]['Collisions'].values
                counts.append(val[0] if len(val) > 0 else 0)
            
            offset = (i - len(algos)/2) * width + width/2
            rects = ax.bar(x + offset, counts, width, label=algo)
            ax.bar_label(rects, padding=3)

        ax.set_ylabel('Collision Count')
        ax.set_title(f'Collisions by Map Size ({scenario.capitalize()})')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(m)}x{int(m)}' for m in map_sizes])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'collisions_{scenario}.png'))
        plt.close()

    # 2. Success Rate (Just Table or Bar)
    # Since we have 1 run per case, it's 0 or 1.
    # Bar chart of Success (1/0)
    for scenario in scenarios:
        subset = df[df['Scenario'] == scenario]
        if subset.empty:
            continue
            
        map_sizes = sorted(subset['Map Size'].unique())
        algos = sorted(subset['Algorithm'].unique())
        x = np.arange(len(map_sizes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, algo in enumerate(algos):
            algo_data = subset[subset['Algorithm'] == algo]
            vals = []
            for m in map_sizes:
                val = algo_data[algo_data['Map Size'] == m]['Success'].values
                vals.append(1 if (len(val) > 0 and val[0]) else 0)
            
            offset = (i - len(algos)/2) * width + width/2
            rects = ax.bar(x + offset, vals, width, label=algo)
            
        ax.set_ylabel('Success (1=Yes, 0=No)')
        ax.set_title(f'Success Status ({scenario.capitalize()})')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(m)}x{int(m)}' for m in map_sizes])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'success_{scenario}.png'))
        plt.close()

    # 3. Time Taken
    for scenario in scenarios:
        subset = df[df['Scenario'] == scenario]
        if subset.empty:
            continue
            
        map_sizes = sorted(subset['Map Size'].unique())
        algos = sorted(subset['Algorithm'].unique())
        x = np.arange(len(map_sizes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, algo in enumerate(algos):
            algo_data = subset[subset['Algorithm'] == algo]
            times = []
            for m in map_sizes:
                val = algo_data[algo_data['Map Size'] == m]['Time (s)'].values
                times.append(val[0] if len(val) > 0 else 0)
            
            offset = (i - len(algos)/2) * width + width/2
            rects = ax.bar(x + offset, times, width, label=algo)
            ax.bar_label(rects, fmt='%.1f', padding=3)
            
        ax.set_ylabel('Time (s)')
        ax.set_title(f'Execution Time ({scenario.capitalize()})')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(m)}x{int(m)}' for m in map_sizes])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'time_{scenario}.png'))
        plt.close()

    print(f"Plots saved to {OUTPUT_DIR}/")
    print(df)
    df.to_csv(os.path.join(OUTPUT_DIR, 'summary_table.csv'), index=False)

if __name__ == "__main__":
    main()
