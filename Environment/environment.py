import pybullet as p
import pybullet_data
from maze_layouts import MAZES, MAZE_CONFIGS

class MazeEnvironment:
    def __init__(self, difficulty="Simple"):
        if not p.isConnected(): 
            p.connect(p.GUI)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0], globalScaling=2.5)
        
        p.resetDebugVisualizerCamera(cameraDistance=28.0, cameraYaw=0, 
                                     cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])
        
        self.colors = {
            'wall': [0.2, 0.2, 0.2, 1],
            'robot_1': [0.1, 0.3, 0.9, 1],
            'robot_2': [0.9, 0.3, 0.1, 1],
            'goal': [0, 1, 0, 0.4]
        }
        
        self.obstacles = []

        self._build_outer_boundary(size=25.0)
        self._build_maze(difficulty)
        
        self.config = MAZE_CONFIGS.get(difficulty)

    def _build_outer_boundary(self, size):
        half_size = size / 2.0
        thickness = 0.3

        # [pos_x, pos_y, pos_z], [dim_x, dim_y, dim_z]
        walls = [
            [[0, half_size, 0.5], [size, thickness, 1.0]],
            [[0, -half_size, 0.5], [size, thickness, 1.0]],
            [[half_size, 0, 0.5], [thickness, size, 1.0]],
            [[-half_size, 0, 0.5], [thickness, size, 1.0]]
        ]
        for pos, dim in walls:
            self._create_box(pos, dim, self.colors['wall'])

    def _build_maze(self, difficulty):
        self.maze_data = MAZES.get(difficulty)
        
        for wall in self.maze_data:
            x, y, w, h = wall
            self._create_box([x, y, 0.5], [w, h, 1.0], self.colors['wall'])

    def _create_box(self, pos, dim, color):
        half_ext = [dim[0]/2, dim[1]/2, dim[2]/2]
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=color)
        return p.createMultiBody(0, col_id, vis_id, pos)

    def draw_goal(self, goal_pos):
        p.createMultiBody(0, -1, 
            p.createVisualShape(p.GEOM_CYLINDER, radius=0.6, length=0.01, rgbaColor=self.colors['goal']), 
            [goal_pos[0], goal_pos[1], 0.02])

    def spawn_car(self, start_pose, robot_idx=1):
        # start_pose is (x, y, yaw)
        color = self.colors['robot_1'] if robot_idx == 1 else self.colors['robot_2']  # maybe useful if we want to spawn mpc and rrt together
        return p.createMultiBody(1.0, 
            p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.2, 0.1]), 
            p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.2, 0.1], rgbaColor=color),
            [start_pose[0], start_pose[1], 0.15],
            p.getQuaternionFromEuler([0, 0, start_pose[2]]))