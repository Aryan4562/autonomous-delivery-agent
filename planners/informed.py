import heapq
import math

class AStarPlanner:
    def __init__(self, environment):
        self.env = environment
        self.nodes_expanded = 0  # Add this line
    
    def heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def plan(self, start, goal, start_time=0):
        open_set = [(0, start, [start], 0, start_time)]  # (f, position, path, g, time)
        closed_set = set()
        g_score = {start: 0}
        self.nodes_expanded = 0  # Reset counter
        
        while open_set:
            f, (x, y), path, g, time = heapq.heappop(open_set)
            self.nodes_expanded += 1
            
            if (x, y) == goal:
                return path
            
            if (x, y) in closed_set:
                continue
            closed_set.add((x, y))
            
            for (nx, ny), move_cost in self.env.get_neighbors(x, y, time):
                if (nx, ny) in closed_set:
                    continue
                
                tentative_g = g + move_cost
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    f_score = tentative_g + self.heuristic((nx, ny), goal)
                    heapq.heappush(open_set, 
                                  (f_score, (nx, ny), path + [(nx, ny)], tentative_g, time + 1))
        
        return None