import unittest
import sys
import os
import tempfile

# Get the current test file directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (src) to Python path
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)

# Now import directly from the modules (not from src package)
from environment import GridEnvironment, GridCell, DynamicObstacle

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = GridEnvironment(5, 5)
    
    def test_grid_initialization(self):
        self.assertEqual(self.env.width, 5)
        self.assertEqual(self.env.height, 5)
        # Test that all cells are initialized properly
        for y in range(5):
            for x in range(5):
                self.assertIsInstance(self.env.grid[y][x], GridCell)
                self.assertEqual(self.env.grid[y][x].terrain_cost, 1)
                self.assertFalse(self.env.grid[y][x].is_obstacle)
                self.assertIsNone(self.env.grid[y][x].dynamic_obstacle)

if __name__ == '__main__':
    unittest.main()
    
    def test_get_neighbors(self):
        # Test getting neighbors from the center
        neighbors = self.env.get_neighbors(2, 2, 0)
        self.assertEqual(len(neighbors), 4)  # 4-connected grid
        
        # Test getting neighbors from a corner
        neighbors = self.env.get_neighbors(0, 0, 0)
        self.assertEqual(len(neighbors), 2)  # Only right and down
        
        # Test with obstacle
        self.env.grid[1][0].is_obstacle = True
        neighbors = self.env.get_neighbors(0, 0, 0)
        self.assertEqual(len(neighbors), 1)  # Only right
        
    def test_dynamic_obstacle(self):
        # Add a dynamic obstacle
        dynamic_obs = DynamicObstacle([1, 3, 5])
        self.env.grid[2][2].dynamic_obstacle = dynamic_obs
        
        # Test at different time steps
        self.assertTrue(self.env.grid[2][2].is_traversable(0))
        self.assertFalse(self.env.grid[2][2].is_traversable(1))
        self.assertTrue(self.env.grid[2][2].is_traversable(2))
        self.assertFalse(self.env.grid[2][2].is_traversable(3))
        
    def test_load_from_file(self):
        # Create a test map file
        test_map_content = """3 3
S 0 0
P 1 1 2 2
1 1 1
1 X 1
1 1 1"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.map', delete=False) as f:
            f.write(test_map_content)
            temp_path = f.name
        
        try:
            # Load the map
            env = GridEnvironment(0, 0)
            env.load_from_file(temp_path)
            
            # Test properties
            self.assertEqual(env.width, 3)
            self.assertEqual(env.height, 3)
            self.assertEqual(env.agent_start, (0, 0))
            self.assertEqual(len(env.packages), 1)
            self.assertEqual(env.packages[0], ((1, 1), (2, 2)))
            
            # Test obstacles
            self.assertFalse(env.grid[0][0].is_obstacle)
            self.assertFalse(env.grid[0][1].is_obstacle)
            self.assertFalse(env.grid[0][2].is_obstacle)
            self.assertFalse(env.grid[1][0].is_obstacle)
            self.assertTrue(env.grid[1][1].is_obstacle)
            self.assertFalse(env.grid[1][2].is_obstacle)
            self.assertFalse(env.grid[2][0].is_obstacle)
            self.assertFalse(env.grid[2][1].is_obstacle)
            self.assertFalse(env.grid[2][2].is_obstacle)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_generate_random_obstacles(self):
        self.env.generate_random_obstacles(0.5)
        
        obstacle_count = 0
        for y in range(5):
            for x in range(5):
                if self.env.grid[y][x].is_obstacle:
                    obstacle_count += 1
        
        # With probability 0.5, we should have roughly 12-13 obstacles in a 5x5 grid
        # (excluding the 25 * 0.5 = 12.5 expected)
        self.assertGreater(obstacle_count, 8)
        self.assertLess(obstacle_count, 17)
    
    def test_generate_random_terrain(self):
        self.env.generate_random_terrain(5)
        
        for y in range(5):
            for x in range(5):
                if not self.env.grid[y][x].is_obstacle:
                    self.assertGreaterEqual(self.env.grid[y][x].terrain_cost, 1)
                    self.assertLessEqual(self.env.grid[y][x].terrain_cost, 5)
    
    def test_generate_random_packages(self):
        self.env.generate_random_packages(2)
        
        self.assertEqual(len(self.env.packages), 2)
        
        for start, destination in self.env.packages:
            # Check that start and destination are within bounds
            self.assertGreaterEqual(start[0], 0)
            self.assertLess(start[0], 5)
            self.assertGreaterEqual(start[1], 0)
            self.assertLess(start[1], 5)
            
            self.assertGreaterEqual(destination[0], 0)
            self.assertLess(destination[0], 5)
            self.assertGreaterEqual(destination[1], 0)
            self.assertLess(destination[1], 5)
            
            # Check that start and destination are not obstacles
            self.assertFalse(self.env.grid[start[1]][start[0]].is_obstacle)
            self.assertFalse(self.env.grid[destination[1]][destination[0]].is_obstacle)
            
            # Check that start and destination are different
            self.assertNotEqual(start, destination)

if __name__ == '__main__':
    unittest.main()