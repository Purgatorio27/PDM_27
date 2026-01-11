#!/usr/bin/env python3
"""
Master runner script to execute all three comparison simulations:
1. Pure RRT*
2. Pure MPC
3. MPC + RRT*

Runs in headless mode and saves logs to logs/ directory.
"""

import subprocess
import os
import sys
import time
from datetime import datetime

def run_experiment(script_name, description):
    print(f"\n{'='*60}")
    print(f"Running Experiment: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Construct command: python3 script_name --headless
        cmd = [sys.executable, script_name, '--headless']
        
        # Run command and capture output
        process = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) # Run from comparison dir
        )
        
        print("STDOUT:", process.stdout[-500:]) # Print last 500 chars
        print(f"✅ Success! Duration: {time.time() - start_time:.2f}s")
        return True, process.stdout
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        print("STDERR:", e.stderr)
        print("STDOUT:", e.stdout)
        return False, e.stderr

def main():
    print(f"Starting Benchmark Suite at {datetime.now()}")
    
    scripts = [
        ("rrt_comparison.py", "Pure RRT* Planner"),
        ("mpc_comparison.py", "Pure MPC Controller"),
        ("mpc_rrt_comparison.py", "Integrated MPC + RRT*")
    ]
    
    results = {}
    
    for script, desc in scripts:
        success, output = run_experiment(script, desc)
        results[script] = success
        
        # Parse output for log file location (heuristic)
        for line in output.split('\n'):
            if "Log Saved" in line or ".json" in line:
                print(f"Found Log info: {line.strip()}")
                
        time.sleep(1) # Cooldown
        
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    all_passed = True
    for script, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{script:<25}: {status}")
        if not success:
            all_passed = False
            
    if all_passed:
        print("\nAll experiments completed successfully.")
        print("Run 'python3 analyze_all_results.py' to generate report.")
        print("Run 'python3 visualize_trajectories.py' to see plots.")
    else:
        print("\nSome experiments failed.")

if __name__ == "__main__":
    main()
