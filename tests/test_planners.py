import unittest
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import directly from modules (remove src. prefix)
from environment import GridEnvironment
from planners.uninformed import BFSPlanner, UniformCostPlanner
from planners.informed import AStarPlanner
from planners.local_search import LocalSearchPlanner

class TestPlanners(unittest.TestCase):
    def setUp(self):
        # Create a simple 3x3 environment
        self.env = GridEnvironment(3, 3)
        
        # Add an obstacle in the middle
        self.env.grid[1][1].is_obstacle = True
        
        # Set start and goal
        self.start = (0, 0)
        self.goal = (2, 2)
    
    def test_bfs_planner(self):
        planner = BFSPlanner(self.env)
        path = planner.plan(self.start, self.goal)
        
        # Should find a path around the obstacle
        self.assertIsNotNone(path)
        self.assertEqual(path[0], self.start)
        self.assertEqual(path[-1], self.goal)
        
        # Path should not go through the obstacle
        self.assertNotIn((1, 1), path)
        
        # Should be one of the two possible shortest paths
        self.assertIn(len(path), [5, 5])  # (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) OR
                                        # (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2)
    
    def test_ucs_planner(self):
        planner = UniformCostPlanner(self.env)
        path = planner.plan(self.start, self.goal)
        
        # Should find a path around the obstacle
        self.assertIsNotNone(path)
        self.assertEqual(path[0], self.start)
        self.assertEqual(path[-1], self.goal)
        
        # Path should not go through the obstacle
        self.assertNotIn((1, 1), path)
        
        # Should be one of the two possible shortest paths
        self.assertIn(len(path), [5, 5])
    
    def test_astar_planner(self):
        planner = AStarPlanner(self.env)
        path = planner.plan(self.start, self.goal)
        
        # Should find a path around the obstacle
        self.assertIsNotNone(path)
        self.assertEqual(path[0], self.start)
        self.assertEqual(path[-1], self.goal)
        
        # Path should not go through the obstacle
        self.assertNotIn((1, 1), path)
        
        # Should be one of the two possible shortest paths
        self.assertIn(len(path), [5, 5])
    
    def test_local_search_planner(self):
        planner = LocalSearchPlanner(self.env, max_restarts=3, max_iterations=100)
        path = planner.plan(self.start, self.goal)
        
        # Should find a path around the obstacle
        self.assertIsNotNone(path)
        self.assertEqual(path[0], self.start)
        self.assertEqual(path[-1], self.goal)
        
        # Path should not go through the obstacle
        self.assertNotIn((1, 1), path)
    
    def test_no_path_scenario(self):
        # Create an environment with no possible path
        env = GridEnvironment(3, 3)
        
        # Add obstacles blocking all paths
        env.grid[0][1].is_obstacle = True
        env.grid[1][0].is_obstacle = True
        env.grid[1][1].is_obstacle = True
        env.grid[1][2].is_obstacle = True
        env.grid[2][1].is_obstacle = True
        
        start = (0, 0)
        goal = (2, 2)
        
        # Test all planners
        planners = [
            BFSPlanner(env),
            UniformCostPlanner(env),
            AStarPlanner(env),
            LocalSearchPlanner(env, max_restarts=3, max_iterations=100)
        ]
        
        for planner in planners:
            path = planner.plan(start, goal)
            self.assertIsNone(path)
    
    def test_varying_terrain_costs(self):
        # Create an environment with varying terrain costs
        env = GridEnvironment(3, 3)
        
        # Add high cost in the direct path
        env.grid[0][1].terrain_cost = 10
        env.grid[1][1].terrain_cost = 10
        env.grid[2][1].terrain_cost = 10
        
        # Add low cost in the alternative path
        env.grid[1][0].terrain_cost = 1
        env.grid[1][2].terrain_cost = 1
        
        start = (0, 0)
        goal = (2, 2)
        
        # UCS and A* should choose the longer but cheaper path
        ucs_planner = UniformCostPlanner(env)
        ucs_path = ucs_planner.plan(start, goal)
        
        astar_planner = AStarPlanner(env)
        astar_path = astar_planner.plan(start, goal)
        
        # Both should find the path around the high-cost cells
        self.assertIsNotNone(ucs_path)
        self.assertIsNotNone(astar_path)
        
        # The optimal path should go through (1,0) or (1,2)
        self.assertTrue((1, 0) in ucs_path or (1, 2) in ucs_path)
        self.assertTrue((1, 0) in astar_path or (1, 2) in astar_path)
        
        # BFS should find the shortest path regardless of cost
        bfs_planner = BFSPlanner(env)
        bfs_path = bfs_planner.plan(start, goal)
        
        self.assertIsNotNone(bfs_path)
        # BFS might choose the direct path through high-cost cells

if __name__ == '__main__':
    unittest.main()