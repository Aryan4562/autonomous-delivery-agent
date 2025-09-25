from planners.informed import AStarPlanner
from planners.local_search import LocalSearchPlanner
from planners.uninformed import BFSPlanner, UniformCostPlanner


class DeliveryAgent:
    def __init__(self, environment, planner_type='astar'):
        self.env = environment
        self.position = environment.agent_start
        self.time_step = 0
        self.fuel = 1000  # Initial fuel
        self.packages_delivered = 0
        self.total_cost = 0
        
        if planner_type == 'bfs':
            self.planner = BFSPlanner(environment)
        elif planner_type == 'ucs':
            self.planner = UniformCostPlanner(environment)
        elif planner_type == 'astar':
            self.planner = AStarPlanner(environment)
        elif planner_type == 'local':
            self.planner = LocalSearchPlanner(environment)
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
    
    def deliver_packages(self):
        results = []
        for package_id, (start, destination) in enumerate(self.env.packages):
            path = self.plan_path(self.position, destination)
            if not path:
                print(f"Failed to find path to package {package_id}")
                continue
            
            # Execute path to package
            self.execute_path(path)
            
            # Pick up package
            print(f"Picked up package {package_id}")
            
            # Plan path to destination
            path = self.plan_path(self.position, destination)
            if not path:
                print(f"Failed to find path to deliver package {package_id}")
                continue
            
            # Execute path to destination
            self.execute_path(path)
            
            # Deliver package
            print(f"Delivered package {package_id}")
            self.packages_delivered += 1
            results.append({
                'package_id': package_id,
                'delivery_time': self.time_step,
                'fuel_used': self.total_cost
            })
        
        return results
    
    def plan_path(self, start, goal):
        return self.planner.plan(start, goal, self.time_step)
    
    def execute_path(self, path):
        for step in path[1:]:  # Skip the first position (current position)
            self.position = step
            self.time_step += 1
            self.fuel -= self.env.grid[step[1]][step[0]].terrain_cost
            self.total_cost += self.env.grid[step[1]][step[0]].terrain_cost
            
            # Check for dynamic obstacles that might require replanning
            if self.env.grid[step[1]][step[0]].dynamic_obstacle:
                if self.env.grid[step[1]][step[0]].dynamic_obstacle.is_occupied(self.time_step):
                    print("Encountered dynamic obstacle! Replanning...")
                    return True  # Signal that replanning is needed
        
        return False