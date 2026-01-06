import pybullet as p
import time
import sys
from environment import MazeEnvironment

def main():
    difficulty = "Intermediate"  # Choose: "Simple", "Intermediate", "Advanced" ("Expert" and "Master" exist but are too difficult)

    if len(sys.argv) > 1:
        difficulty = sys.argv[1]

    print(f"--- Loading {difficulty} Maze (25x25) ---")
    env = MazeEnvironment(difficulty=difficulty)
    
    start_pose = env.config['start']
    goal_pos = env.config['goal']

    robot_1 = env.spawn_car(start_pose, robot_idx=1)
    env.draw_goal(goal_pos)

    # Keep simulation open
    try:
        while True:
            p.stepSimulation()
            time.sleep(0.01)
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    main()