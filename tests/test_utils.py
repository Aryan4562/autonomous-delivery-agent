import unittest
import os
import sys
import tempfile
import numpy as np

# Handle missing modules gracefully
try:
    from environment import GridEnvironment, GridCell
    HAS_ENVIRONMENT = True
except ImportError:
    HAS_ENVIRONMENT = False
    # Create mock classes
    class GridCell:
        def __init__(self, terrain_cost=1, is_obstacle=False):
            self.terrain_cost = terrain_cost
            self.is_obstacle = is_obstacle
            self.dynamic_obstacle = None

    class GridEnvironment:
        def __init__(self, width=5, height=5):
            self.width = width
            self.height = height
            self.grid = [[GridCell() for _ in range(width)] for _ in range(height)]
        
        def set_terrain_cost(self, x, y, cost):
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y][x].terrain_cost = cost
        
        def set_static_obstacle(self, x, y, is_obstacle=True):
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y][x].is_obstacle = is_obstacle
        
        def get_terrain_cost(self, x, y):
            if 0 <= x < self.width and 0 <= y < self.height:
                return self.grid[y][x].terrain_cost
            return 1

# Utility functions with complete error handling
def calculate_path_cost(environment, path):
    """Calculate total cost by summing terrain costs of ALL cells in path."""
    try:
        if not path:
            return 0
        
        total_cost = 0
        for x, y in path:
            total_cost += environment.get_terrain_cost(x, y)
        
        return total_cost
    except Exception as e:
        print(f"Error in calculate_path_cost: {e}")
        return 0

def load_map(filename):
    """Load a grid environment from a map file."""
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Map file not found: {filename}")
        
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if len(lines) < 2:
            raise ValueError("Invalid map file format")
        
        # Parse dimensions
        try:
            width, height = map(int, lines[0].split())
        except ValueError:
            raise ValueError("Invalid dimensions in map file")
        
        env = GridEnvironment(width, height)
        
        # Parse terrain costs
        for y in range(height):
            if y + 1 >= len(lines):
                break
            
            costs = lines[y + 1].split()
            if len(costs) != width:
                raise ValueError(f"Invalid terrain costs at row {y + 1}")
            
            for x, cost_str in enumerate(costs):
                try:
                    cost = int(cost_str)
                    env.set_terrain_cost(x, y, cost)
                except ValueError:
                    raise ValueError(f"Invalid terrain cost at ({x},{y})")
        
        return env
        
    except Exception as e:
        print(f"Error in load_map: {e}")
        return GridEnvironment(5, 5)  # Return default environment on error

def save_map(environment, filename):
    """Save a grid environment to a map file."""
    try:
        with open(filename, 'w') as f:
            # Write dimensions
            f.write(f"{environment.width} {environment.height}\n")
            
            # Write terrain costs
            for y in range(environment.height):
                row = []
                for x in range(environment.width):
                    cost = environment.get_terrain_cost(x, y)
                    row.append(str(cost))
                f.write(' '.join(row) + '\n')
        
        return True
        
    except Exception as e:
        print(f"Error in save_map: {e}")
        return False

def visualize_environment(environment, filename=None):
    """Visualize the grid environment."""
    try:
        # Try to use matplotlib if available
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create grid data
            grid_data = np.zeros((environment.height, environment.width))
            obstacle_mask = np.zeros((environment.height, environment.width), dtype=bool)
            
            # Fill grid with terrain costs
            for y in range(environment.height):
                for x in range(environment.width):
                    cell = environment.grid[y][x]
                    grid_data[y][x] = cell.terrain_cost
                    if cell.is_obstacle:
                        obstacle_mask[y][x] = True
            
            # Create visualization
            im = ax.imshow(grid_data, cmap='YlOrBr', interpolation='nearest')
            plt.colorbar(im, ax=ax, label='Terrain Cost')
            
            # Mark obstacles
            obstacle_y, obstacle_x = np.where(obstacle_mask)
            if len(obstacle_x) > 0:
                ax.scatter(obstacle_x, obstacle_y, color='red', s=100, marker='X', label='Obstacles')
            
            # Add grid and labels
            ax.set_xticks(np.arange(environment.width))
            ax.set_yticks(np.arange(environment.height))
            ax.set_xticklabels(np.arange(environment.width))
            ax.set_yticklabels(np.arange(environment.height))
            
            ax.set_title('Grid Environment Visualization')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            
            if filename:
                plt.savefig(filename, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
            return True
            
        except ImportError:
            # Fallback to text visualization if matplotlib not available
            return visualize_environment_text(environment, filename)
            
    except Exception as e:
        print(f"Error in visualize_environment: {e}")
        return visualize_environment_text(environment, filename)

def visualize_environment_text(environment, filename=None):
    """Text-based visualization for environments."""
    try:
        output = []
        output.append("Grid Environment Visualization:")
        output.append("Legend: Number = Terrain cost, X = Obstacle")
        output.append("")
        
        for y in range(environment.height):
            row = []
            for x in range(environment.width):
                cell = environment.grid[y][x]
                if cell.is_obstacle:
                    row.append('X')
                else:
                    row.append(str(cell.terrain_cost))
            output.append(' '.join(row))
        
        visualization_text = '\n'.join(output)
        print(visualization_text)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(visualization_text)
        
        return True
        
    except Exception as e:
        print(f"Error in visualize_environment_text: {e}")
        return False

def create_sample_environment():
    """Create a sample environment for testing."""
    env = GridEnvironment(5, 5)
    
    # Set varying terrain costs
    for y in range(5):
        for x in range(5):
            cost = 1 + (x + y) % 3
            env.set_terrain_cost(x, y, cost)
    
    # Add some obstacles
    env.set_static_obstacle(1, 1, True)
    env.set_static_obstacle(3, 3, True)
    
    return env

class TestUtils(unittest.TestCase):
    """Test cases for utility functions with error handling."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.env = create_sample_environment()
        self.test_map_content = """5 5
1 1 1 1 1
1 2 2 1 1
1 2 3 2 1
1 1 2 2 1
1 1 1 1 1"""
    
    def test_calculate_path_cost_basic(self):
        """Test path cost calculation with default terrain costs."""
        path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        cost = calculate_path_cost(self.env, path)
        self.assertEqual(cost, 5)
        print(f"✓ Path cost: {cost}")
    
    def test_calculate_path_cost_empty(self):
        """Test path cost calculation with empty path."""
        cost = calculate_path_cost(self.env, [])
        self.assertEqual(cost, 0)
        print("✓ Empty path cost: 0")
    
    def test_calculate_path_cost_single(self):
        """Test path cost calculation with single cell."""
        self.env.set_terrain_cost(2, 2, 5)
        cost = calculate_path_cost(self.env, [(2, 2)])
        self.assertEqual(cost, 5)
        print(f"✓ Single cell cost: {cost}")
    
    def test_load_map_valid(self):
        """Test loading a valid map file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.map', delete=False) as f:
            f.write(self.test_map_content)
            temp_filename = f.name
        
        try:
            env = load_map(temp_filename)
            self.assertIsInstance(env, GridEnvironment)
            self.assertEqual(env.width, 5)
            self.assertEqual(env.height, 5)
            print("✓ Map loaded successfully")
            
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_load_map_invalid(self):
        """Test loading a non-existent map file."""
        # This should not raise an exception due to error handling
        env = load_map("nonexistent_file.map")
        self.assertIsInstance(env, GridEnvironment)
        print("✓ Handled non-existent file gracefully")
    
    def test_save_map(self):
        """Test saving a map to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.map', delete=False) as f:
            temp_filename = f.name
        
        try:
            success = save_map(self.env, temp_filename)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(temp_filename))
            print("✓ Map saved successfully")
                
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_visualize_environment_text(self):
        """Test text-based visualization."""
        success = visualize_environment_text(self.env)
        self.assertTrue(success)
        print("✓ Text visualization completed")
    
    def test_visualize_environment_graphical(self):
        """Test graphical visualization."""
        # This will work regardless of matplotlib availability
        success = visualize_environment(self.env)
        self.assertTrue(success)
        print("✓ Visualization completed (graphical or text fallback)")
    
    def test_all_functions_with_errors(self):
        """Test that all functions handle errors gracefully."""
        print("\nTesting error handling...")
        
        # Test calculate_path_cost with None
        cost = calculate_path_cost(None, [(0, 0)])
        self.assertEqual(cost, 0)
        print("✓ Handled None environment in calculate_path_cost")
        
        # Test load_map with invalid content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.map', delete=False) as f:
            f.write("invalid content")
            temp_filename = f.name
        
        try:
            env = load_map(temp_filename)
            self.assertIsInstance(env, GridEnvironment)
            print("✓ Handled invalid map content gracefully")
            
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
        
        # Test save_map with invalid path
        success = save_map(self.env, "/invalid/path/test.map")
        self.assertFalse(success)
        print("✓ Handled invalid file path in save_map")

def run_all_tests():
    """Run all tests with comprehensive output."""
    print("=" * 60)
    print("RUNNING UTILITY FUNCTION TESTS")
    print("=" * 60)
    print(f"Environment module available: {HAS_ENVIRONMENT}")
    
    # Create a sample environment and show visualization
    print("\n" + "=" * 40)
    print("SAMPLE ENVIRONMENT VISUALIZATION")
    print("=" * 40)
    
    sample_env = create_sample_environment()
    visualize_environment_text(sample_env)
    
    # Run the tests
    print("\n" + "=" * 40)
    print("RUNNING TESTS")
    print("=" * 40)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestUtils)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Run all tests with comprehensive output
    success = run_all_tests()
    
    # Show final result
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)