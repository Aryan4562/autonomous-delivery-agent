import argparse
from environment import GridEnvironment
from agent import DeliveryAgent

def main():
    parser = argparse.ArgumentParser(description="Autonomous Delivery Agent")
    parser.add_argument("map_file", help="Path to the map file")
    parser.add_argument("--planner", choices=["bfs", "ucs", "astar", "local"], 
                       default="astar", help="Path planning algorithm")
    parser.add_argument("--packages", type=int, default=1, 
                       help="Number of packages to deliver")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Load environment
    env = GridEnvironment(0, 0)  # Will be resized when loading from file
    env.load_from_file(args.map_file)
    
    # Generate random packages
    env.generate_random_packages(args.packages)
    
    # Create and run agent
    agent = DeliveryAgent(env, args.planner)
    results = agent.deliver_packages()
    
    # Print results
    print(f"Packages delivered: {agent.packages_delivered}/{args.packages}")
    print(f"Total time steps: {agent.time_step}")
    print(f"Total cost: {agent.total_cost}")
    print(f"Fuel remaining: {agent.fuel}")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            for result in results:
                f.write(f"Package {result['package_id']}: "
                       f"Time={result['delivery_time']}, "
                       f"Fuel={result['fuel_used']}\n")
            f.write(f"Total: Time={agent.time_step}, Cost={agent.total_cost}\n")

if __name__ == "__main__":
    main()