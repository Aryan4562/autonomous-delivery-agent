from collections import deque
import heapq

class BFSPlanner:
    def __init__(self, environment):
        self.env = environment
    
    def plan(self, start, goal, start_time=0):
        queue = deque([(start, [start], start_time)])
        visited = set([start])
        
        while queue:
            (x, y), path, time = queue.popleft()
            
            if (x, y) == goal:
                return path
            
            for (nx, ny), cost in self.env.get_neighbors(x, y, time):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)], time + 1))
        
        return None

class UniformCostPlanner:
    def __init__(self, environment):
        self.env = environment
    
    def plan(self, start, goal, start_time=0):
        priority_queue = [(0, start, [start], start_time)]
        visited = set()
        
        while priority_queue:
            cost_so_far, (x, y), path, time = heapq.heappop(priority_queue)
            
            if (x, y) == goal:
                return path
            
            if (x, y) in visited:
                continue
            visited.add((x, y))
            
            for (nx, ny), move_cost in self.env.get_neighbors(x, y, time):
                if (nx, ny) not in visited:
                    new_cost = cost_so_far + move_cost
                    heapq.heappush(priority_queue, 
                                  (new_cost, (nx, ny), path + [(nx, ny)], time + 1))
        
        return None