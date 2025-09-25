import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment import GridEnvironment, DynamicObstacle
from agent import DeliveryAgent

class TestAgent(unittest.TestCase):  # Make sure the class name is TestAgent
    def setUp(self):
        # Create a simple 3x3 environment
        self.env = GridEnvironment(3, 3)
        
        # Add a package
        self.env.packages = [((0, 0), (2, 2))]
        self.env.agent_start = (0, 0)
    
    def test_agent_initialization(self):
        agent = DeliveryAgent(self.env, 'astar')
        
        self.assertEqual(agent.position, (0, 0))
        self.assertEqual(agent.time_step, 0)
        self.assertEqual(agent.fuel, 1000)
        self.assertEqual(agent.packages_delivered, 0)
        self.assertEqual(agent.total_cost, 0)
        self.assertIsNotNone(agent.planner)
    
    # ... other test methods ...

if __name__ == '__main__':
    unittest.main()
    