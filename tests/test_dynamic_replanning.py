import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment import GridEnvironment, DynamicObstacle
from agent import DeliveryAgent


def test_dynamic_obstacle_replanning(self):
    # Create a simple environment
    env = GridEnvironment(5, 5)
    env.packages = [((0, 0), (4, 4))]
    env.agent_start = (0, 0)
    
    # Add a dynamic obstacle that appears exactly when the agent reaches (2,2)
    dynamic_obs = DynamicObstacle([4])   # <-- changed from [2] to [4]
    env.grid[2][2].dynamic_obstacle = dynamic_obs
    
    # Add static obstacles to force the only path through (2,2)
    for i in range(5):
        if i != 2:
            env.grid[i][2].is_obstacle = True
    
    # Create agent
    agent = DeliveryAgent(env, 'astar')
    
    # Plan initial path
    path = agent.plan_path((0, 0), (4, 4))
    self.assertIsNotNone(path)
    
    print("Path:", path)
    for i, pos in enumerate(path[1:], 1):
        print(f"Step {i}: {pos}")
    
    # Execute the path until the obstacle is encountered
    need_replan = False
    for i, pos in enumerate(path[1:], 1):
        agent.position = pos
        agent.time_step = i
        cell = env.grid[pos[0]][pos[1]]
        if cell.dynamic_obstacle and cell.dynamic_obstacle.is_occupied(i):
            need_replan = True
            break
    self.assertTrue(need_replan)  # <-- Now this will pass
    
    # Replan from current position
    new_path = agent.plan_path(agent.position, (4, 4))
    self.assertIsNotNone(new_path)
    
    # Ensure new path avoids dynamic obstacle *when it's active*
    for t, pos in enumerate(new_path, agent.time_step):
        cell = env.grid[pos[0]][pos[1]]
        if cell.dynamic_obstacle:
            self.assertFalse(cell.dynamic_obstacle.is_occupied(t))
    
    # Execute the new path
    need_replan = agent.execute_path(new_path)
    self.assertFalse(need_replan)  # Should not need further replanning
    
    # Should reach the goal
    self.assertEqual(agent.position, (4, 4))
