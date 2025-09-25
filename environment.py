import pygame
import sys
import numpy as np
from typing import List, Tuple, Dict, Optional, Set, Deque
from enum import Enum
import time
import heapq
import math
import random
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1400, 900
GRID_SIZE = 50
GRID_OFFSET_X = 150
GRID_OFFSET_Y = 100
SIDEBAR_WIDTH = 400
METRICS_HEIGHT = 200

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
YELLOW = (255, 255, 0)
PURPLE = (200, 0, 255)
GRAY = (150, 150, 150)
LIGHT_BLUE = (173, 216, 230)
BROWN = (165, 42, 42)
DARK_GREEN = (0, 100, 0)
SAND = (237, 201, 175)
ORANGE = (255, 165, 0)
DARK_RED = (139, 0, 0)
LIGHT_GRAY = (220, 220, 220)
DARK_BLUE = (0, 0, 139)
PINK = (255, 105, 180)
LIGHT_GREEN = (144, 238, 144)

# Terrain colors
TERRAIN_COLORS = {
    1: (200, 200, 200),  # ROAD - Light gray
    2: (34, 139, 34),    # GRASS - Forest green
    3: (139, 69, 19),    # MUD - Brown
    4: (65, 105, 225),   # WATER - Royal blue
    5: (139, 137, 137)   # MOUNTAIN - Dark gray
}

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Delivery Agent Simulation - Multiple Algorithms")

# Fonts
font = pygame.font.SysFont(None, 24)
small_font = pygame.font.SysFont(None, 18)
title_font = pygame.font.SysFont(None, 32)
large_font = pygame.font.SysFont(None, 40)

class AlgorithmType(Enum):
    BFS = "Breadth-First Search"
    UCS = "Uniform-Cost Search"
    ASTAR = "A* Search"
    GREEDY = "Greedy Best-First"
    HILL_CLIMBING = "Hill Climbing"

class TerrainType(Enum):
    ROAD = 1
    GRASS = 2
    MUD = 3
    WATER = 4
    MOUNTAIN = 5

class MovementType(Enum):
    DETERMINISTIC = "deterministic"
    UNPREDICTABLE = "unpredictable"

class GridCell:
    def __init__(self, terrain_type: TerrainType = TerrainType.ROAD, is_obstacle: bool = False):
        self.terrain_type = terrain_type
        self.terrain_cost = terrain_type.value
        self.is_obstacle = is_obstacle
        self.dynamic_obstacles: Set[str] = set()
        self.visited = False
        self.visit_time = None
        self.visited_by = None
    
    def add_dynamic_obstacle(self, obstacle_id: str):
        self.dynamic_obstacles.add(obstacle_id)
    
    def remove_dynamic_obstacle(self, obstacle_id: str):
        self.dynamic_obstacles.discard(obstacle_id)
    
    def has_dynamic_obstacle(self) -> bool:
        return len(self.dynamic_obstacles) > 0
    
    def mark_visited(self, agent_id: str, time: int):
        self.visited = True
        self.visited_by = agent_id
        self.visit_time = time
    
    def __str__(self):
        status = []
        if self.is_obstacle:
            status.append("Static Obstacle")
        if self.has_dynamic_obstacle():
            status.append(f"Dynamic Obstacles: {list(self.dynamic_obstacles)}")
        status_str = " | ".join(status) if status else "Clear"
        return f"Cell(Terrain={self.terrain_type.name}, Cost={self.terrain_cost}, Status={status_str})"

class DynamicObstacle:
    def __init__(self, obstacle_id: str, start_pos: Tuple[int, int], 
                 movement_type: MovementType = MovementType.DETERMINISTIC,
                 schedule: List[Tuple[int, int, int]] = None,
                 movement_pattern: List[Tuple[int, int]] = None):
        self.obstacle_id = obstacle_id
        self.current_pos = start_pos
        self.movement_type = movement_type
        self.schedule = schedule or []  # For deterministic: [(x, y, time)]
        self.movement_pattern = movement_pattern or []  # For unpredictable: sequence of moves
        self.current_pattern_index = 0
        self.last_move_time = 0
    
    def get_position_at_time(self, time_step: int) -> Tuple[int, int]:
        if self.movement_type == MovementType.DETERMINISTIC:
            return self._get_deterministic_position(time_step)
        else:
            return self._get_unpredictable_position(time_step)
    
    def _get_deterministic_position(self, time_step: int) -> Tuple[int, int]:
        for x, y, t in self.schedule:
            if t == time_step:
                return (x, y)
        return self.current_pos  # Stay in place if no schedule entry
    
    def _get_unpredictable_position(self, time_step: int) -> Tuple[int, int]:
        if time_step > self.last_move_time:
            if self.movement_pattern:
                # Follow pattern
                dx, dy = self.movement_pattern[self.current_pattern_index]
                self.current_pattern_index = (self.current_pattern_index + 1) % len(self.movement_pattern)
                new_x, new_y = self.current_pos[0] + dx, self.current_pos[1] + dy
                self.current_pos = (new_x, new_y)
            else:
                # Random movement (4-connected)
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                dx, dy = random.choice(directions)
                new_x, new_y = self.current_pos[0] + dx, self.current_pos[1] + dy
                self.current_pos = (new_x, new_y)
            
            self.last_move_time = time_step
        
        return self.current_pos

class GridEnvironment:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[GridCell() for _ in range(width)] for _ in range(height)]
        self.dynamic_obstacles: Dict[str, DynamicObstacle] = {}
        self.time_step = 0
        self.obstacle_history: Dict[int, Set[Tuple[int, int]]] = {}  # time -> obstacle positions
        self.algorithm_stats: Dict[AlgorithmType, Dict] = {}
    
    def set_terrain(self, x: int, y: int, terrain_type: TerrainType):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x].terrain_type = terrain_type
            self.grid[y][x].terrain_cost = terrain_type.value
    
    def set_static_obstacle(self, x: int, y: int, is_obstacle: bool = True):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x].is_obstacle = is_obstacle
    
    def add_dynamic_obstacle(self, obstacle_id: str, start_pos: Tuple[int, int], 
                           movement_type: MovementType = MovementType.DETERMINISTIC,
                           schedule: List[Tuple[int, int, int]] = None,
                           movement_pattern: List[Tuple[int, int]] = None) -> DynamicObstacle:
        obstacle = DynamicObstacle(obstacle_id, start_pos, movement_type, schedule, movement_pattern)
        self.dynamic_obstacles[obstacle_id] = obstacle
        
        # Add to initial position
        x, y = start_pos
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x].add_dynamic_obstacle(obstacle_id)
        
        return obstacle
    
    def update_time_step(self, new_time: int):
        """Update all dynamic obstacles to their positions at the given time"""
        self.time_step = new_time
        obstacle_positions = set()
        
        # Clear previous dynamic obstacles
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x].dynamic_obstacles.clear()
        
        # Update obstacle positions
        for obstacle_id, obstacle in self.dynamic_obstacles.items():
            x, y = obstacle.get_position_at_time(new_time)
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y][x].add_dynamic_obstacle(obstacle_id)
                obstacle_positions.add((x, y))
        
        # Store history
        self.obstacle_history[new_time] = obstacle_positions
    
    def get_neighbors(self, x: int, y: int, time: Optional[int] = None) -> List[Tuple[Tuple[int, int], int]]:
        """Get valid neighboring cells with movement costs"""
        if time is not None:
            self.update_time_step(time)
        
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-connected
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < self.width and 0 <= ny < self.height:
                cell = self.grid[ny][nx]
                
                # Skip if blocked
                if cell.is_obstacle or cell.has_dynamic_obstacle():
                    continue
                
                neighbors.append(((nx, ny), cell.terrain_cost))
        
        return neighbors
    
    def is_valid_position(self, x: int, y: int, time: Optional[int] = None) -> bool:
        """Check if a position is valid at the given time"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        
        if time is not None:
            self.update_time_step(time)
        
        cell = self.grid[y][x]
        return not (cell.is_obstacle or cell.has_dynamic_obstacle())
    
    def get_terrain_cost(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x].terrain_cost
        return float('inf')
    
    def mark_visited(self, x: int, y: int, agent_id: str, visit_time: int = None):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x].mark_visited(agent_id, visit_time or self.time_step)
    
    def get_obstacle_positions_at_time(self, time: int) -> Set[Tuple[int, int]]:
        """Get all obstacle positions (static + dynamic) at a specific time"""
        static_obstacles = set()
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x].is_obstacle:
                    static_obstacles.add((x, y))
        
        dynamic_obstacles = self.obstacle_history.get(time, set())
        return static_obstacles.union(dynamic_obstacles)
    
    def predict_obstacle_positions(self, start_time: int, horizon: int) -> Dict[int, Set[Tuple[int, int]]]:
        """Predict obstacle positions for future time steps"""
        predictions = {}
        
        for t in range(start_time, start_time + horizon + 1):
            obstacle_positions = set()
            
            # Static obstacles
            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y][x].is_obstacle:
                        obstacle_positions.add((x, y))
            
            # Dynamic obstacles (predict their positions)
            for obstacle_id, obstacle in self.dynamic_obstacles.items():
                if obstacle.movement_type == MovementType.DETERMINISTIC:
                    x, y = obstacle.get_position_at_time(t)
                    obstacle_positions.add((x, y))
                # For unpredictable obstacles, we can't predict future positions
            
            predictions[t] = obstacle_positions
        
        return predictions

class DeliveryAgent:
    def __init__(self, agent_id: str, environment: GridEnvironment, start_pos: Tuple[int, int], 
                 start_fuel: int = 100, max_time: int = 1000):
        self.agent_id = agent_id
        self.env = environment
        self.position = start_pos
        self.start_pos = start_pos
        self.fuel = start_fuel
        self.max_fuel = start_fuel
        self.max_time = max_time
        self.current_time = 0
        self.packages_delivered = 0
        self.total_distance = 0
        self.total_cost = 0
        self.collisions_avoided = 0
        self.actual_collisions = 0
        self.replans = 0
        self.path_history: List[Tuple[int, int, int]] = []  # (x, y, time)
        self.current_path = []
        self.current_goal = None
        self.assigned_packages = []
        self.completed_deliveries = []
        self.status = "IDLE"
        self.delivery_times = []  # Track time taken for each delivery
        self.algorithm = AlgorithmType.ASTAR
        self.nodes_expanded = 0
        self.search_time = 0
        self.replan_log = []
    
    def assign_package(self, package_loc: Tuple[int, int], delivery_loc: Tuple[int, int]):
        self.assigned_packages.append((package_loc, delivery_loc, self.current_time))  # Store assignment time
        self.status = "ASSIGNED"
    
    def set_algorithm(self, algorithm: AlgorithmType):
        self.algorithm = algorithm
    
    def manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def euclidean_distance(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def plan_path_bfs(self, start, goal):
        """Breadth-First Search path planning"""
        self.nodes_expanded = 0
        start_time = time.time()
        
        queue = deque([(start, [])])
        visited = set()
        
        while queue:
            (x, y), path = queue.popleft()
            self.nodes_expanded += 1
            
            if (x, y) == goal:
                self.search_time = time.time() - start_time
                return path + [(x, y)]
                
            if (x, y) in visited:
                continue
                
            visited.add((x, y))
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                    cell = self.env.grid[ny][nx]
                    if cell.is_obstacle or cell.has_dynamic_obstacle():
                        continue
                    
                    queue.append(((nx, ny), path + [(x, y)]))
        
        self.search_time = time.time() - start_time
        return None
    
    def plan_path_ucs(self, start, goal):
        """Uniform-Cost Search path planning"""
        self.nodes_expanded = 0
        start_time = time.time()
        
        heap = [(0, start, [])]
        cost_so_far = {start: 0}
        
        while heap:
            cost, (x, y), path = heapq.heappop(heap)
            self.nodes_expanded += 1
            
            if (x, y) == goal:
                self.search_time = time.time() - start_time
                return path + [(x, y)]
                
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                    cell = self.env.grid[ny][nx]
                    if cell.is_obstacle or cell.has_dynamic_obstacle():
                        continue
                    
                    new_cost = cost + cell.terrain_cost
                    if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                        cost_so_far[(nx, ny)] = new_cost
                        heapq.heappush(heap, (new_cost, (nx, ny), path + [(x, y)]))
        
        self.search_time = time.time() - start_time
        return None
    
    def plan_path_astar(self, start, goal):
        """A* Search path planning with Manhattan heuristic"""
        self.nodes_expanded = 0
        start_time = time.time()
        
        def heuristic(a, b):
            return self.manhattan_distance(a, b)
            
        heap = [(0, start, [])]
        cost_so_far = {start: 0}
        
        while heap:
            priority, (x, y), path = heapq.heappop(heap)
            self.nodes_expanded += 1
            
            if (x, y) == goal:
                self.search_time = time.time() - start_time
                return path + [(x, y)]
                
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                    cell = self.env.grid[ny][nx]
                    if cell.is_obstacle or cell.has_dynamic_obstacle():
                        continue
                    
                    new_cost = cost_so_far.get((x, y), 0) + cell.terrain_cost
                    if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                        cost_so_far[(nx, ny)] = new_cost
                        priority = new_cost + heuristic(goal, (nx, ny))
                        heapq.heappush(heap, (priority, (nx, ny), path + [(x, y)]))
        
        self.search_time = time.time() - start_time
        return None
    
    def plan_path_greedy(self, start, goal):
        """Greedy Best-First Search path planning"""
        self.nodes_expanded = 0
        start_time = time.time()
        
        def heuristic(a, b):
            return self.manhattan_distance(a, b)
            
        heap = [(heuristic(start, goal), start, [])]
        visited = set()
        
        while heap:
            priority, (x, y), path = heapq.heappop(heap)
            self.nodes_expanded += 1
            
            if (x, y) == goal:
                self.search_time = time.time() - start_time
                return path + [(x, y)]
                
            if (x, y) in visited:
                continue
                
            visited.add((x, y))
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                    cell = self.env.grid[ny][nx]
                    if cell.is_obstacle or cell.has_dynamic_obstacle():
                        continue
                    
                    heapq.heappush(heap, (heuristic((nx, ny), goal), (nx, ny), path + [(x, y)]))
        
        self.search_time = time.time() - start_time
        return None
    
    def plan_path_hill_climbing(self, start, goal):
        """Hill Climbing path planning with backtracking"""
        self.nodes_expanded = 0
        start_time = time.time()
        
        def heuristic(a, b):
            return self.manhattan_distance(a, b)
            
        current = start
        path = [start]
        visited = set([start])
        
        while current != goal:
            self.nodes_expanded += 1
            
            # Get all neighbors
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                    cell = self.env.grid[ny][nx]
                    if cell.is_obstacle or cell.has_dynamic_obstacle() or (nx, ny) in visited:
                        continue
                    
                    neighbors.append((nx, ny))
            
            if not neighbors:
                # Backtrack if no good moves
                if len(path) <= 1:
                    self.search_time = time.time() - start_time
                    return None  # No path found
                
                path.pop()
                current = path[-1]
                continue
            
            # Choose the neighbor with best heuristic
            best_neighbor = min(neighbors, key=lambda n: heuristic(n, goal))
            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor
        
        self.search_time = time.time() - start_time
        return path
    
    def plan_path(self, start, goal):
        """Route to the appropriate path planning algorithm based on selection"""
        if self.algorithm == AlgorithmType.BFS:
            return self.plan_path_bfs(start, goal)
        elif self.algorithm == AlgorithmType.UCS:
            return self.plan_path_ucs(start, goal)
        elif self.algorithm == AlgorithmType.ASTAR:
            return self.plan_path_astar(start, goal)
        elif self.algorithm == AlgorithmType.GREEDY:
            return self.plan_path_greedy(start, goal)
        elif self.algorithm == AlgorithmType.HILL_CLIMBING:
            return self.plan_path_hill_climbing(start, goal)
        else:
            return self.plan_path_astar(start, goal)  # Default to A*
    
    def execute_step(self):
        """Execute one step of movement"""
        if not self.assigned_packages:
            self.status = "IDLE"
            return False
        
        # Sync agent time with environment
        self.current_time = self.env.time_step
        
        if not self.current_path:
            # Get next package to deliver
            package_loc, delivery_loc, assignment_time = self.assigned_packages[0]
            
            if self.position == package_loc:
                # Already at package, plan to delivery location
                self.current_goal = delivery_loc
                self.status = "DELIVERING"
            else:
                # Plan to package location
                self.current_goal = package_loc
                self.status = "PICKING_UP"
            
            # Plan path to goal
            self.current_path = self.plan_path(self.position, self.current_goal)
            
            if not self.current_path:
                self.status = "STUCK"
                return False
        
        # Check if path is still valid (dynamic obstacles might have moved)
        path_still_valid = True
        obstacle_positions = set()
        
        for i, (x, y) in enumerate(self.current_path):
            if i > 0:  # Don't check current position
                if not self.env.is_valid_position(x, y, self.current_time + i):
                    path_still_valid = False
                    obstacle_positions.add((x, y))
                    break
        
        if not path_still_valid:
            self.replans += 1
            log_msg = f"Step {self.current_time}: Obstacle detected at {obstacle_positions}, replanning with {self.algorithm.value}"
            self.replan_log.append(log_msg)
            print(log_msg)
            
            self.current_path = self.plan_path(self.position, self.current_goal)
            if not self.current_path:
                self.status = "STUCK"
                return False
        
        # Move to next position in path
        if len(self.current_path) > 1:
            next_pos = self.current_path[1]  # Skip current position
            
            # Check for collision at the next time step
            if not self.env.is_valid_position(*next_pos, self.current_time + 1):
                self.collisions_avoided += 1
                self.replans += 1
                log_msg = f"Step {self.current_time}: Collision avoided at {next_pos}, replanning with {self.algorithm.value}"
                self.replan_log.append(log_msg)
                print(log_msg)
                
                self.current_path = self.plan_path(self.position, self.current_goal)
                if not self.current_path:
                    self.status = "STUCK"
                return True
            
            move_cost = self.env.get_terrain_cost(*next_pos)
            
            # Execute move
            self.position = next_pos
            self.fuel -= move_cost
            self.total_cost += move_cost
            self.current_time += 1
            self.total_distance += 1
            self.current_path.pop(0)  # Remove the position we just moved from
            
            # Record path
            self.path_history.append((*next_pos, self.current_time))
            
            # Mark cell as visited
            self.env.mark_visited(*next_pos, self.agent_id, self.current_time)
            
            # Check if goal reached
            if self.position == self.current_goal:
                if self.status == "PICKING_UP":
                    self.status = "DELIVERING"
                    self.current_goal = self.assigned_packages[0][1]  # Delivery location
                    self.current_path = self.plan_path(self.position, self.current_goal)
                elif self.status == "DELIVERING":
                    # Delivery complete
                    package_loc, delivery_loc, assignment_time = self.assigned_packages.pop(0)
                    delivery_time = self.current_time - assignment_time
                    self.completed_deliveries.append((package_loc, delivery_loc, delivery_time))
                    self.delivery_times.append(delivery_time)
                    self.packages_delivered += 1
                    self.current_path = []
                    self.current_goal = None
                    self.status = "COMPLETED" if not self.assigned_packages else "ASSIGNED"
        
        return True
    
    def get_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate delivery efficiency metrics"""
        total_assigned = len(self.completed_deliveries) + len(self.assigned_packages)
        avg_delivery_time = sum(self.delivery_times) / len(self.delivery_times) if self.delivery_times else 0
        
        return {
            'agent_id': self.agent_id,
            'algorithm': self.algorithm.value,
            'packages_delivered': self.packages_delivered,
            'total_time': self.current_time,
            'total_cost': self.total_cost,
            'fuel_used': self.max_fuel - self.fuel,
            'total_distance': self.total_distance,
            'collisions_avoided': self.collisions_avoided,
            'actual_collisions': self.actual_collisions,
            'replans': self.replans,
            'nodes_expanded': self.nodes_expanded,
            'search_time': self.search_time,
            'success_rate': self.packages_delivered / total_assigned if total_assigned > 0 else 1.0,
            'avg_delivery_time': avg_delivery_time,
            'efficiency': self.packages_delivered / self.current_time if self.current_time > 0 else 0
        }

def create_small_environment():
    """Create a small environment for testing"""
    env = GridEnvironment(8, 6)
    
    # Set terrain
    for y in range(env.height):
        for x in range(env.width):
            env.set_terrain(x, y, TerrainType.ROAD)
    
    # Add some obstacles
    obstacle_positions = [(2, 2), (3, 2), (4, 3), (5, 4)]
    for x, y in obstacle_positions:
        env.set_static_obstacle(x, y, True)
    
    return env

def create_medium_environment():
    """Create a medium environment for testing"""
    env = GridEnvironment(12, 10)
    
    # Set varying terrain
    for y in range(env.height):
        for x in range(env.width):
            # Create terrain patterns based on position
            terrain_val = (x + y) % 5
            if terrain_val == 0:
                env.set_terrain(x, y, TerrainType.ROAD)
            elif terrain_val == 1:
                env.set_terrain(x, y, TerrainType.GRASS)
            elif terrain_val == 2:
                env.set_terrain(x, y, TerrainType.MUD)
            elif terrain_val == 3:
                env.set_terrain(x, y, TerrainType.WATER)
            else:
                env.set_terrain(x, y, TerrainType.MOUNTAIN)
    
    # Add static obstacles
    obstacle_positions = [(2, 4), (5, 4), (8, 4)]
    for x, y in obstacle_positions:
        env.set_static_obstacle(x, y, True)
    
    return env

def create_large_environment():
    """Create a large environment for testing"""
    env = GridEnvironment(20, 15)
    
    # Set varying terrain
    for y in range(env.height):
        for x in range(env.width):
            # Create terrain patterns
            terrain_val = (x * y) % 7
            if terrain_val == 0:
                env.set_terrain(x, y, TerrainType.ROAD)
            elif terrain_val in [1, 2]:
                env.set_terrain(x, y, TerrainType.GRASS)
            elif terrain_val == 3:
                env.set_terrain(x, y, TerrainType.MUD)
            elif terrain_val == 4:
                env.set_terrain(x, y, TerrainType.WATER)
            else:
                env.set_terrain(x, y, TerrainType.MOUNTAIN)
    
    # Add more obstacles
    for i in range(10):
        x = random.randint(2, env.width-3)
        y = random.randint(2, env.height-3)
        env.set_static_obstacle(x, y, True)
    
    return env

def create_dynamic_environment():
    """Create an environment with dynamic obstacles"""
    env = create_medium_environment()
    
    # Add deterministic dynamic obstacle
    env.add_dynamic_obstacle(
        "det_obs", (2, 2), MovementType.DETERMINISTIC,
        schedule=[(2,2,0), (3,2,5), (4,2,10), (4,3,15), (4,4,20), (4,5,25)]
    )
    
    # Add unpredictable dynamic obstacle
    env.add_dynamic_obstacle(
        "rand_obs", (8, 7), MovementType.UNPREDICTABLE,
        movement_pattern=[(0, 1), (1, 0), (0, -1), (-1, 0)]
    )
    
    return env

def run_experiment(env_func, packages, algorithm, runs=3):
    """Run an experiment with a specific algorithm and environment"""
    results = []
    
    for run in range(runs):
        env = env_func()
        agent = DeliveryAgent(f"Agent-{algorithm.value}-{run}", env, (0, 0), 500, 1000)
        agent.set_algorithm(algorithm)
        
        # Assign packages
        for package in packages:
            agent.assign_package(*package)
        
        # Run simulation
        max_steps = 200
        for step in range(max_steps):
            env.update_time_step(step)
            if not agent.execute_step():
                break
        
        # Collect results
        metrics = agent.get_efficiency_metrics()
        results.append(metrics)
    
    return results

def run_comparative_experiments():
    """Run comparative experiments across all algorithms and environments"""
    all_results = []
    
    # Define test environments and packages
    test_cases = [
        ("Small", create_small_environment, [((1, 1), (6, 4))]),
        ("Medium", create_medium_environment, [((2, 3), (8, 7)), ((5, 5), (10, 2))]),
        ("Large", create_large_environment, [((3, 2), (16, 12)), ((8, 5), (12, 10)), ((15, 3), (5, 12))]),
        ("Dynamic", create_dynamic_environment, [((2, 3), (8, 7)), ((5, 5), (10, 2))])
    ]
    
    algorithms = [AlgorithmType.BFS, AlgorithmType.UCS, AlgorithmType.ASTAR, 
                 AlgorithmType.GREEDY, AlgorithmType.HILL_CLIMBING]
    
    # Run experiments
    for env_name, env_func, packages in test_cases:
        print(f"Running experiments for {env_name} environment...")
        
        for algorithm in algorithms:
            results = run_experiment(env_func, packages, algorithm)
            
            for result in results:
                result['environment'] = env_name
                all_results.append(result)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    # Calculate averages for each algorithm in each environment
    summary = df.groupby(['environment', 'algorithm']).agg({
        'total_cost': 'mean',
        'total_time': 'mean',
        'nodes_expanded': 'mean',
        'search_time': 'mean',
        'success_rate': 'mean'
    }).round(2)
    
    print("\nExperimental Results Summary:")
    print(summary)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total Cost comparison
    cost_pivot = df.pivot_table(index='environment', columns='algorithm', values='total_cost', aggfunc='mean')
    cost_pivot.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Total Cost by Algorithm')
    axes[0, 0].set_ylabel('Cost')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Total Time comparison
    time_pivot = df.pivot_table(index='environment', columns='algorithm', values='total_time', aggfunc='mean')
    time_pivot.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Total Time by Algorithm')
    axes[0, 1].set_ylabel('Time Steps')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Nodes Expanded comparison
    nodes_pivot = df.pivot_table(index='environment', columns='algorithm', values='nodes_expanded', aggfunc='mean')
    nodes_pivot.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Nodes Expanded by Algorithm')
    axes[1, 0].set_ylabel('Nodes')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Success Rate comparison
    success_pivot = df.pivot_table(index='environment', columns='algorithm', values='success_rate', aggfunc='mean')
    success_pivot.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Success Rate by Algorithm')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    plt.show()
    
    return df, summary

# Create environment and single agent
env = create_medium_environment()
agent = DeliveryAgent("Delivery-Bot", env, (0, 0), 200, 1000)

# Define packages and delivery locations
packages = [
    ((2, 3), (8, 7)),  # Package 1
    ((5, 5), (10, 2)), # Package 2
]

# Assign all packages to the single agent
for package in packages:
    agent.assign_package(*package)

# Track delivered packages for visualization
delivered_packages = set()
completed_deliveries = []

# Main visualization loop
running = True
current_time = 0
animation_speed = 0.5  # Slower speed for better observation
last_update = time.time()
paused = True  # Start paused
show_terrain_costs = True
show_visited_cells = True
show_obstacle_prediction = False
show_paths = True
show_agent_ids = True
simulation_complete = False
current_algorithm = AlgorithmType.ASTAR
show_replan_log = False
experiment_results = None

clock = pygame.time.Clock()

while running:
    current_time_real = time.time()
    delta_time = current_time_real - last_update
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_RIGHT and paused:
                current_time = min(current_time + 1, 100)
                env.update_time_step(int(current_time))
                if not simulation_complete:
                    agent.execute_step()
                    # Check for completed deliveries
                    if len(agent.completed_deliveries) > len(completed_deliveries):
                        completed_deliveries = agent.completed_deliveries.copy()
                        # Update delivered packages set for visualization
                        for pkg_loc, deliv_loc, delivery_time in completed_deliveries:
                            delivered_packages.add(deliv_loc)
            elif event.key == pygame.K_LEFT and paused:
                current_time = max(current_time - 1, 0)
                env.update_time_step(int(current_time))
            elif event.key == pygame.K_t:
                show_terrain_costs = not show_terrain_costs
            elif event.key == pygame.K_v:
                show_visited_cells = not show_visited_cells
            elif event.key == pygame.K_o:
                show_obstacle_prediction = not show_obstacle_prediction
            elif event.key == pygame.K_p:
                show_paths = not show_paths
            elif event.key == pygame.K_i:
                show_agent_ids = not show_agent_ids
            elif event.key == pygame.K_r:
                # Reset
                env = create_medium_environment()
                agent = DeliveryAgent("Delivery-Bot", env, (0, 0), 200, 1000)
                agent.set_algorithm(current_algorithm)
                for package in packages:
                    agent.assign_package(*package)
                current_time = 0
                env.update_time_step(int(current_time))
                delivered_packages = set()
                completed_deliveries = []
                simulation_complete = False
            elif event.key == pygame.K_l:
                show_replan_log = not show_replan_log
            elif event.key == pygame.K_1:
                current_algorithm = AlgorithmType.BFS
                agent.set_algorithm(current_algorithm)
            elif event.key == pygame.K_2:
                current_algorithm = AlgorithmType.UCS
                agent.set_algorithm(current_algorithm)
            elif event.key == pygame.K_3:
                current_algorithm = AlgorithmType.ASTAR
                agent.set_algorithm(current_algorithm)
            elif event.key == pygame.K_4:
                current_algorithm = AlgorithmType.GREEDY
                agent.set_algorithm(current_algorithm)
            elif event.key == pygame.K_5:
                current_algorithm = AlgorithmType.HILL_CLIMBING
                agent.set_algorithm(current_algorithm)
            elif event.key == pygame.K_e:
                # Run experiments
                experiment_results, summary = run_comparative_experiments()
                print("Experiments completed. Results saved to algorithm_comparison.png")
    
    # Update time if not paused and simulation not complete
    if not paused and not simulation_complete:
        current_time += animation_speed * delta_time
        current_time = min(current_time, 100)  # Cap at 100 time steps
        env.update_time_step(int(current_time))
        
        # Execute agent actions
        steps_to_execute = int(animation_speed)
        for _ in range(steps_to_execute):
            if not simulation_complete:
                agent.execute_step()
        
        # Check for completed deliveries
        if len(agent.completed_deliveries) > len(completed_deliveries):
            completed_deliveries = agent.completed_deliveries.copy()
            # Update delivered packages set for visualization
            for pkg_loc, deliv_loc, delivery_time in completed_deliveries:
                delivered_packages.add(deliv_loc)
        
        # Check if all packages are delivered
        if agent.packages_delivered >= len(packages):
            simulation_complete = True
            agent.status = "MISSION_COMPLETE"
    
    last_update = current_time_real
    
    # Draw background
    screen.fill(WHITE)
    
    # Draw title
    title_text = large_font.render("Delivery Agent Simulation - Multiple Algorithms", True, DARK_BLUE)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 20))
    
    # Draw algorithm info
    algo_text = font.render(f"Algorithm: {current_algorithm.value}", True, BLUE)
    screen.blit(algo_text, (WIDTH // 2 - algo_text.get_width() // 2, 60))
    
    # Draw mission status
    if simulation_complete:
        status_text = large_font.render("MISSION COMPLETE! ALL PACKAGES DELIVERED", True, GREEN)
        screen.blit(status_text, (WIDTH // 2 - status_text.get_width() // 2, 90))
    
    # Draw grid
    for y in range(env.height):
        for x in range(env.width):
            cell_x = GRID_OFFSET_X + x * GRID_SIZE
            cell_y = GRID_OFFSET_Y + y * GRID_SIZE
            
            # Draw terrain
            terrain_type = env.grid[y][x].terrain_type
            terrain_color = TERRAIN_COLORS.get(terrain_type.value, GRAY)
            
            # Darken if visited
            if show_visited_cells and env.grid[y][x].visited:
                terrain_color = tuple(max(0, c - 40) for c in terrain_color)
            
            pygame.draw.rect(screen, terrain_color, (cell_x, cell_y, GRID_SIZE, GRID_SIZE))
            
            # Draw grid lines
            pygame.draw.rect(screen, GRAY, (cell_x, cell_y, GRID_SIZE, GRID_SIZE), 1)
            
            # Draw terrain cost
            if show_terrain_costs and not env.grid[y][x].is_obstacle:
                cost_text = small_font.render(str(env.grid[y][x].terrain_cost), True, BLACK)
                screen.blit(cost_text, (cell_x + 5, cell_y + 5))
            
            # Draw static obstacles
            if env.grid[y][x].is_obstacle:
                pygame.draw.rect(screen, BLACK, (cell_x, cell_y, GRID_SIZE, GRID_SIZE))
                pygame.draw.line(screen, RED, (cell_x, cell_y), (cell_x + GRID_SIZE, cell_y + GRID_SIZE), 2)
                pygame.draw.line(screen, RED, (cell_x + GRID_SIZE, cell_y), (cell_x, cell_y + GRID_SIZE), 2)
            
            # Draw dynamic obstacles
            if env.grid[y][x].has_dynamic_obstacle():
                pygame.draw.circle(screen, RED, (cell_x + GRID_SIZE//2, cell_y + GRID_SIZE//2), GRID_SIZE//3)
    
    # Draw package and delivery locations
    for i, (pkg_loc, deliv_loc) in enumerate(packages):
        # Check if this package has been delivered
        is_delivered = any(deliv_loc == comp[1] for comp in completed_deliveries)
        is_picked_up = any(pkg_loc == comp[0] for comp in completed_deliveries)
        
        # Package location (only show if not yet picked up)
        pkg_x, pkg_y = pkg_loc
        if not is_picked_up and not is_delivered:
            pygame.draw.rect(screen, ORANGE, 
                            (GRID_OFFSET_X + pkg_x * GRID_SIZE, 
                             GRID_OFFSET_Y + pkg_y * GRID_SIZE, 
                             GRID_SIZE, GRID_SIZE))
            pkg_text = font.render(f"P{i+1}", True, BLACK)
            screen.blit(pkg_text, (GRID_OFFSET_X + pkg_x * GRID_SIZE + 15, 
                                  GRID_OFFSET_Y + pkg_y * GRID_SIZE + 15))
        
        # Delivery location - show different color if delivered
        del_x, del_y = deliv_loc
        if is_delivered:
            # Delivered - show with green checkmark
            pygame.draw.rect(screen, LIGHT_GREEN, 
                            (GRID_OFFSET_X + del_x * GRID_SIZE, 
                             GRID_OFFSET_Y + del_y * GRID_SIZE, 
                             GRID_SIZE, GRID_SIZE))
            # Draw checkmark
            pygame.draw.line(screen, GREEN, 
                            (GRID_OFFSET_X + del_x * GRID_SIZE + 10, GRID_OFFSET_Y + del_y * GRID_SIZE + 20),
                            (GRID_OFFSET_X + del_x * GRID_SIZE + 20, GRID_OFFSET_Y + del_y * GRID_SIZE + 30), 3)
            pygame.draw.line(screen, GREEN, 
                            (GRID_OFFSET_X + del_x * GRID_SIZE + 20, GRID_OFFSET_Y + del_y * GRID_SIZE + 30),
                            (GRID_OFFSET_X + del_x * GRID_SIZE + 30, GRID_OFFSET_Y + del_y * GRID_SIZE + 10), 3)
        else:
            # Not yet delivered
            pygame.draw.rect(screen, PURPLE, 
                            (GRID_OFFSET_X + del_x * GRID_SIZE, 
                             GRID_OFFSET_Y + del_y * GRID_SIZE, 
                             GRID_SIZE, GRID_SIZE))
        
        del_text = font.render(f"D{i+1}", True, WHITE)
        screen.blit(del_text, (GRID_OFFSET_X + del_x * GRID_SIZE + 15, 
                              GRID_OFFSET_Y + del_y * GRID_SIZE + 15))
    
    # Draw agent path
    if show_paths and agent.current_path:
        for j in range(len(agent.current_path) - 1):
            x1, y1 = agent.current_path[j]
            x2, y2 = agent.current_path[j+1]
            start_x = GRID_OFFSET_X + x1 * GRID_SIZE + GRID_SIZE // 2
            start_y = GRID_OFFSET_Y + y1 * GRID_SIZE + GRID_SIZE // 2
            end_x = GRID_OFFSET_X + x2 * GRID_SIZE + GRID_SIZE // 2
            end_y = GRID_OFFSET_Y + y2 * GRID_SIZE + GRID_SIZE // 2
            
            pygame.draw.line(screen, BLUE, (start_x, start_y), (end_x, end_y), 3)
    
    # Draw agent
    agent_x, agent_y = agent.position
    screen_x = GRID_OFFSET_X + agent_x * GRID_SIZE + GRID_SIZE // 2
    screen_y = GRID_OFFSET_Y + agent_y * GRID_SIZE + GRID_SIZE // 2
    
    # Draw agent
    pygame.draw.circle(screen, RED, (screen_x, screen_y), GRID_SIZE // 2 - 2)
    
    # Draw agent ID
    if show_agent_ids:
        agent_id_text = font.render(agent.agent_id, True, WHITE)
        screen.blit(agent_id_text, (screen_x - agent_id_text.get_width() // 2, 
                                   screen_y - agent_id_text.get_height() // 2))
    
    # Draw sidebar
    pygame.draw.rect(screen, LIGHT_GRAY, (WIDTH - SIDEBAR_WIDTH, 0, SIDEBAR_WIDTH, HEIGHT - METRICS_HEIGHT))
    
    # Draw time info
    time_text = font.render(f"Time Step: {int(current_time)}", True, BLACK)
    screen.blit(time_text, (WIDTH - SIDEBAR_WIDTH + 20, 100))
    
    # Draw agent status
    status_y = 150
    status_text = font.render("Agent Status:", True, BLACK)
    screen.blit(status_text, (WIDTH - SIDEBAR_WIDTH + 20, status_y))
    
    status_color = GREEN if agent.status == "COMPLETED" or agent.status == "MISSION_COMPLETE" else \
                  BLUE if agent.status == "DELIVERING" else \
                  ORANGE if agent.status == "PICKING_UP" else \
                  YELLOW if agent.status == "ASSIGNED" else \
                  RED if agent.status == "STUCK" else GRAY
    
    agent_status = font.render(f"{agent.agent_id}: {agent.status}", True, status_color)
    screen.blit(agent_status, (WIDTH - SIDEBAR_WIDTH + 20, status_y + 30))
    
    # Show assigned packages
    pkg_text = font.render(f"Packages: {agent.packages_delivered}/{len(packages)} delivered", True, BLACK)
    screen.blit(pkg_text, (WIDTH - SIDEBAR_WIDTH + 20, status_y + 60))
    
    # Draw terrain legend
    legend_y = status_y + 100
    legend_text = font.render("Terrain Types:", True, BLACK)
    screen.blit(legend_text, (WIDTH - SIDEBAR_WIDTH + 20, legend_y))
    
    for i, terrain in enumerate(TerrainType):
        pygame.draw.rect(screen, TERRAIN_COLORS[terrain.value], 
                        (WIDTH - SIDEBAR_WIDTH + 20, legend_y + 30 + i*30, 20, 20))
        terrain_name = font.render(terrain.name, True, BLACK)
        screen.blit(terrain_name, (WIDTH - SIDEBAR_WIDTH + 50, legend_y + 30 + i*30))
    
    # Draw algorithm controls
    algo_controls_y = legend_y + 30 + len(TerrainType)*30 + 20
    algo_text = font.render("Algorithm Controls:", True, BLACK)
    screen.blit(algo_text, (WIDTH - SIDEBAR_WIDTH + 20, algo_controls_y))
    
    algo_controls = [
        "1: BFS",
        "2: UCS",
        "3: A*",
        "4: Greedy",
        "5: Hill Climbing"
    ]
    
    for i, control in enumerate(algo_controls):
        control_text = small_font.render(control, True, BLACK)
        screen.blit(control_text, (WIDTH - SIDEBAR_WIDTH + 20, algo_controls_y + 30 + i*20))
    
    # Draw controls
    controls_y = algo_controls_y + 30 + len(algo_controls)*20 + 20
    controls_text = font.render("Simulation Controls:", True, BLACK)
    screen.blit(controls_text, (WIDTH - SIDEBAR_WIDTH + 20, controls_y))
    
    controls = [
        "SPACE: Pause/Resume",
        "RIGHT: Step forward",
        "LEFT: Step backward",
        "T: Toggle terrain costs",
        "V: Toggle visited cells",
        "P: Toggle paths",
        "I: Toggle agent IDs",
        "L: Toggle replan log",
        "E: Run experiments",
        "R: Reset simulation"
    ]
    
    for i, control in enumerate(controls):
        control_text = small_font.render(control, True, BLACK)
        screen.blit(control_text, (WIDTH - SIDEBAR_WIDTH + 20, controls_y + 30 + i*20))
    
    # Draw replan log if enabled
    if show_replan_log and agent.replan_log:
        log_y = controls_y + 30 + len(controls)*20 + 20
        log_text = font.render("Replanning Log:", True, BLACK)
        screen.blit(log_text, (WIDTH - SIDEBAR_WIDTH + 20, log_y))
        
        # Show last 3 log entries
        for i, log_entry in enumerate(agent.replan_log[-3:]):
            log_entry_text = small_font.render(log_entry, True, BLACK)
            screen.blit(log_entry_text, (WIDTH - SIDEBAR_WIDTH + 20, log_y + 30 + i*20))
    
    # Draw metrics panel
    pygame.draw.rect(screen, LIGHT_GRAY, (0, HEIGHT - METRICS_HEIGHT, WIDTH, METRICS_HEIGHT))
    metrics_title = font.render("Performance Metrics:", True, BLACK)
    screen.blit(metrics_title, (20, HEIGHT - METRICS_HEIGHT + 20))
    
    # Calculate and display metrics
    metrics = agent.get_efficiency_metrics()
    
    main_metrics = [
        f"Packages: {metrics['packages_delivered']}/{len(packages)} ({metrics['success_rate']*100:.1f}%)",
        f"Total Time: {metrics['total_time']}",
        f"Total Cost: {metrics['total_cost']}",
        f"Total Distance: {metrics['total_distance']}",
        f"Nodes Expanded: {metrics['nodes_expanded']}",
        f"Search Time: {metrics['search_time']:.4f}s",
    ]
    
    for i, metric in enumerate(main_metrics):
        metric_text = font.render(metric, True, BLACK)
        screen.blit(metric_text, (20, HEIGHT - METRICS_HEIGHT + 50 + i*25))
    
    # Draw collision metrics
    collision_text = font.render(f"Collisions Avoided: {metrics['collisions_avoided']}", True, BLACK)
    screen.blit(collision_text, (400, HEIGHT - METRICS_HEIGHT + 50))
    
    replan_text = font.render(f"Replans: {metrics['replans']}", True, BLACK)
    screen.blit(replan_text, (400, HEIGHT - METRICS_HEIGHT + 75))
    
    # Draw progress bar
    progress_width = 600
    pygame.draw.rect(screen, BLACK, (20, HEIGHT - 40, progress_width, 20), 1)
    if len(packages) > 0:
        fill_width = int((metrics['packages_delivered'] / len(packages)) * progress_width)
        pygame.draw.rect(screen, GREEN, (20, HEIGHT - 40, fill_width, 20))
    
    progress_text = font.render(f"Progress: {metrics['success_rate']*100:.1f}%", True, BLACK)
    screen.blit(progress_text, (progress_width + 30, HEIGHT - 40))
    
    pygame.display.flip()
    clock.tick(30)  # Cap at 30 FPS

pygame.quit()
sys.exit()