import random
import math
from .informed import AStarPlanner

class LocalSearchPlanner:
    def __init__(self, environment, max_restarts=10, max_iterations=1000):
        self.env = environment
        self.max_restarts = max_restarts
        self.max_iterations = max_iterations
        self.nodes_expanded = 0
    
    def plan(self, start, goal, start_time=0):
        # First try A* to get a good initial solution
        astar_planner = AStarPlanner(self.env)
        initial_path = astar_planner.plan(start, goal, start_time)
        self.nodes_expanded += astar_planner.nodes_expanded
        
        if not initial_path:
            # If A* fails, use a random path as initial solution
            initial_path = self.generate_random_path(start, goal, start_time)
            if not initial_path:
                return None
        
        best_path = initial_path
        best_cost = self.calculate_path_cost(best_path, start_time)
        
        # Hill climbing with random restarts
        for restart in range(self.max_restarts):
            current_path = best_path
            current_cost = best_cost
            
            for iteration in range(self.max_iterations):
                # Generate neighbor by modifying the path
                neighbor_path = self.generate_neighbor(current_path, goal, start_time)
                if not neighbor_path:
                    continue
                
                neighbor_cost = self.calculate_path_cost(neighbor_path, start_time)
                
                # Simulated annealing acceptance criterion
                if neighbor_cost < current_cost or random.random() < math.exp((current_cost - neighbor_cost) / self.temperature(iteration)):
                    current_path = neighbor_path
                    current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_path = current_path
                    best_cost = current_cost
            
            # Random restart
            if restart < self.max_restarts - 1:
                random_path = self.generate_random_path(start, goal, start_time)
                if random_path:
                    random_cost = self.calculate_path_cost(random_path, start_time)
                    if random_cost < best_cost:
                        best_path = random_path
                        best_cost = random_cost
        
        return best_path
    
    def temperature(self, iteration):
        # Cooling schedule for simulated annealing
        return 1.0 / (iteration + 1)
    
    def calculate_path_cost(self, path, start_time):
        cost = 0
        time = start_time
        for i in range(1, len(path)):
            x, y = path[i]
            cost += self.env.grid[y][x].terrain_cost
            time += 1
        return cost
    
    def generate_neighbor(self, path, goal, start_time):
        # Modify the path by changing a segment
        if len(path) <= 2:
            return path
        
        # Select a random segment to modify
        i = random.randint(1, len(path) - 2)
        j = random.randint(i + 1, min(i + 5, len(path) - 1))
        
        # Try to find a better path between these points
        segment_start = path[i-1]
        segment_end = path[j]
        
        # Use A* to find a better segment
        astar_planner = AStarPlanner(self.env)
        new_segment = astar_planner.plan(segment_start, segment_end, start_time + i - 1)
        self.nodes_expanded += astar_planner.nodes_expanded
        
        if new_segment:
            return path[:i] + new_segment[1:] + path[j+1:]
        
        return path
    
    def generate_random_path(self, start, goal, start_time):
        # Generate a random valid path from start to goal
        path = [start]
        current = start
        time = start_time
        visited = set([start])
        
        for _ in range(self.env.width * self.env.height * 2):  # Limit iterations
            if current == goal:
                return path
            
            neighbors = self.env.get_neighbors(current[0], current[1], time)
            if not neighbors:
                break
            
            # Filter out already visited neighbors
            valid_neighbors = [(n, cost) for n, cost in neighbors if n not in visited]
            if not valid_neighbors:
                # If all neighbors are visited, allow revisiting
                valid_neighbors = neighbors
            
            next_node, cost = random.choice(valid_neighbors)
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            time += 1
        
        return None if current != goal else path