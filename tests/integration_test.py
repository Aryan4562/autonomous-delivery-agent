import sys
import os

from planners.informed import AStarPlanner
from planners.local_search import LocalSearchPlanner
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment import GridEnvironment
from src.agent import DeliveryAgent
from src.utils import benchmark_planners, plot_benchmark_results
from src.planners import UniformCostPlanner, BFSPlanner

def integration_test():
    print("Running integration test...")
    
    # Load the small map
    env = GridEnvironment(0, 0)
    env.load_from_file("maps/small.map")
    
    print(f"Map loaded: {env.width}x{env.height}")
    print(f"Packages: {len(env.packages)}")
    print(f"Start position: {env.agent_start}")
    
    # Test each planner
    planners = {
        'BFS': BFSPlanner(env),
        'UCS': UniformCostPlanner(env),
        'A*': AStarPlanner(env),
        'Local Search': LocalSearchPlanner(env, max_restarts=3, max_iterations=100)
    }
    
    # Benchmark the planners
    results = benchmark_planners(env, env.agent_start, env.packages[0][1], planners)
    
    print("\nBenchmark Results:")
    for planner, data in results.items():
        print(f"{planner}:")
        print(f"  Success: {data['success']}")
        print(f"  Path length: {data['path_length']}")
        print(f"  Path cost: {data['path_cost']}")
        print(f"  Computation time: {data['computation_time']:.6f}s")
        print(f"  Nodes expanded: {data['nodes_expanded']}")
    
    # Test the agent
    print("\nTesting agent delivery...")
    agent = DeliveryAgent(env, 'astar')
    delivery_results = agent.deliver_packages()
    
    print(f"Packages delivered: {agent.packages_delivered}/{len(env.packages)}")
    print(f"Total time steps: {agent.time_step}")
    print(f"Total cost: {agent.total_cost}")
    print(f"Fuel remaining: {agent.fuel}")
    
    # Create a visualization
    try:
        from src.utils import visualize_environment
        visualize_environment(env, save_path="integration_test.png")
        print("Visualization saved to integration_test.png")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print("Integration test completed successfully!")

if __name__ == "__main__":
    integration_test()