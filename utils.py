# src/utils.py
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import heapq
import random
import math

def visualize_environment(env, agent_path=None, current_step=0, save_path=None):
    """
    Visualize the grid environment with obstacles, terrain costs, and agent path.
    
    Args:
        env: GridEnvironment instance
        agent_path: List of (x, y) positions representing the agent's path
        current_step: Current time step for dynamic obstacles
        save_path: File path to save the visualization (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a grid representation
    grid = np.zeros((env.height, env.width))
    obstacle_mask = np.zeros((env.height, env.width), dtype=bool)
    dynamic_obstacle_mask = np.zeros((env.height, env.width), dtype=bool)
    
    for y in range(env.height):
        for x in range(env.width):
            cell = env.grid[y][x]
            grid[y][x] = cell.terrain_cost
            
            if cell.is_obstacle:
                obstacle_mask[y][x] = True
            elif cell.dynamic_obstacle and cell.dynamic_obstacle.is_occupied(current_step):
                dynamic_obstacle_mask[y][x] = True
    
    # Display the base grid
    im = ax.imshow(grid, cmap='YlOrBr', alpha=0.7)
    
    # Mark obstacles
    for y in range(env.height):
        for x in range(env.width):
            if obstacle_mask[y][x]:
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, fill=True, color='black', alpha=0.7))
            elif dynamic_obstacle_mask[y][x]:
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, fill=True, color='red', alpha=0.7))
    
    # Mark start position
    start_x, start_y = env.agent_start
    ax.plot(start_x, start_y, 'gs', markersize=15, label='Start')
    
    # Mark package positions
    for i, (start, destination) in enumerate(env.packages):
        ax.plot(start[0], start[1], 'bo', markersize=12, label='Package Start' if i == 0 else "")
        ax.plot(destination[0], destination[1], 'bd', markersize=12, label='Package Destination' if i == 0 else "")
        ax.annotate(f'P{i}', (start[0], start[1]), color='white', weight='bold', 
                   fontsize=10, ha='center', va='center')
    
    # Draw agent path if provided
    if agent_path:
        path_x = [pos[0] for pos in agent_path]
        path_y = [pos[1] for pos in agent_path]
        ax.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.7)
        ax.plot(path_x, path_y, 'ro', markersize=4, alpha=0.7)
        
        # Mark current position
        if current_step < len(agent_path):
            current_x, current_y = agent_path[current_step]
            ax.plot(current_x, current_y, 'ro', markersize=15, label='Current Position')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add labels and title
    ax.set_xticks(np.arange(0, env.width, 1))
    ax.set_yticks(np.arange(0, env.height, 1))
    ax.set_xticklabels(np.arange(0, env.width, 1))
    ax.set_yticklabels(np.arange(0, env.height, 1))
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Grid Environment Visualization')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add colorbar for terrain costs
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Terrain Cost')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_random_map(width, height, obstacle_prob=0.2, max_terrain_cost=5, num_packages=1, save_path=None):
    """
    Create a random map file.
    
    Args:
        width: Map width
        height: Map height
        obstacle_prob: Probability of a cell being an obstacle
        max_terrain_cost: Maximum terrain cost (minimum is always 1)
        num_packages: Number of packages to include
        save_path: Path to save the map file
    """
    lines = []
    
    # Header
    lines.append(f"{width} {height}")
    
    # Start position (always top-left)
    lines.append(f"S 0 0")
    
    # Generate packages
    for i in range(num_packages):
        # Find valid start and destination positions
        while True:
            start = (random.randint(0, width-1), random.randint(0, height-1))
            if start != (0, 0):  # Don't put package at start
                break
        
        while True:
            destination = (random.randint(0, width-1), random.randint(0, height-1))
            if destination != start and destination != (0, 0):
                break
        
        lines.append(f"P {start[0]} {start[1]} {destination[0]} {destination[1]}")
    
    # Generate grid
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            if (x, y) == (0, 0):
                # Start position always has cost 1
                row.append("1")
            elif random.random() < obstacle_prob:
                row.append("X")
            else:
                row.append(str(random.randint(1, max_terrain_cost)))
        grid.append(" ".join(row))
    
    lines.extend(grid)
    
    # Save to file if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write("\n".join(lines))
    
    return "\n".join(lines)

def benchmark_planners(env, start, goal, planners, num_trials=1):
    """
    Benchmark different planners on the same problem.
    
    Args:
        env: GridEnvironment instance
        start: Start position (x, y)
        goal: Goal position (x, y)
        planners: Dictionary of planner names to planner instances
        num_trials: Number of trials to run for each planner
    
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    for name, planner in planners.items():
        print(f"Benchmarking {name}...")
        trial_results = []
        
        for trial in range(num_trials):
            # Reset nodes expanded counter if it exists
            if hasattr(planner, 'nodes_expanded'):
                planner.nodes_expanded = 0
            
            start_time = time.time()
            path = planner.plan(start, goal)
            end_time = time.time()
            
            if path:
                path_cost = calculate_path_cost(env, path)
                trial_results.append({
                    'success': True,
                    'path_length': len(path),
                    'path_cost': path_cost,
                    'computation_time': end_time - start_time,
                    'nodes_expanded': getattr(planner, 'nodes_expanded', 0)
                })
            else:
                trial_results.append({
                    'success': False,
                    'path_length': 0,
                    'path_cost': float('inf'),
                    'computation_time': end_time - start_time,
                    'nodes_expanded': getattr(planner, 'nodes_expanded', 0)
                })
        
        # Calculate averages
        if num_trials > 1:
            success_rate = sum(1 for r in trial_results if r['success']) / num_trials
            avg_path_length = np.mean([r['path_length'] for r in trial_results if r['success']]) if success_rate > 0 else 0
            avg_path_cost = np.mean([r['path_cost'] for r in trial_results if r['success']]) if success_rate > 0 else float('inf')
            avg_computation_time = np.mean([r['computation_time'] for r in trial_results])
            avg_nodes_expanded = np.mean([r['nodes_expanded'] for r in trial_results])
            
            results[name] = {
                'success_rate': success_rate,
                'avg_path_length': avg_path_length,
                'avg_path_cost': avg_path_cost,
                'avg_computation_time': avg_computation_time,
                'avg_nodes_expanded': avg_nodes_expanded,
                'trials': trial_results
            }
        else:
            results[name] = trial_results[0]
    
    return results

def calculate_path_cost(env, path):
    """
    Calculate the total cost of a path.
    
    Args:
        env: GridEnvironment instance
        path: List of (x, y) positions
    
    Returns:
        Total cost of the path
    """
    cost = 0
    for i in range(1, len(path)):
        x, y = path[i]
        cost += env.grid[y][x].terrain_cost
    return cost

def export_results(results, filename):
    """
    Export benchmark results to a CSV file.
    
    Args:
        results: Dictionary of results from benchmark_planners
        filename: Output filename
    """
    import csv
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['planner', 'success_rate', 'avg_path_length', 'avg_path_cost', 
                     'avg_computation_time', 'avg_nodes_expanded']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for planner, data in results.items():
            if 'success_rate' in data:  # Multiple trials
                writer.writerow({
                    'planner': planner,
                    'success_rate': data['success_rate'],
                    'avg_path_length': data['avg_path_length'],
                    'avg_path_cost': data['avg_path_cost'],
                    'avg_computation_time': data['avg_computation_time'],
                    'avg_nodes_expanded': data['avg_nodes_expanded']
                })
            else:  # Single trial
                writer.writerow({
                    'planner': planner,
                    'success_rate': 1.0 if data['success'] else 0.0,
                    'avg_path_length': data['path_length'],
                    'avg_path_cost': data['path_cost'],
                    'avg_computation_time': data['computation_time'],
                    'avg_nodes_expanded': data['nodes_expanded']
                })

def plot_benchmark_results(results, save_path=None):
    """
    Create visualizations of benchmark results.
    
    Args:
        results: Dictionary of results from benchmark_planners
        save_path: Path to save the plot (optional)
    """
    planners = list(results.keys())
    
    # Extract data for plotting
    success_rates = [results[p].get('success_rate', 1.0 if results[p].get('success', False) else 0.0) for p in planners]
    path_costs = [results[p].get('avg_path_cost', results[p].get('path_cost', 0)) for p in planners]
    computation_times = [results[p].get('avg_computation_time', results[p].get('computation_time', 0)) for p in planners]
    nodes_expanded = [results[p].get('avg_nodes_expanded', results[p].get('nodes_expanded', 0)) for p in planners]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Success rate plot
    axes[0, 0].bar(planners, success_rates, color='skyblue')
    axes[0, 0].set_title('Success Rate')
    axes[0, 0].set_ylim(0, 1.1)
    for i, v in enumerate(success_rates):
        axes[0, 0].text(i, v + 0.05, f'{v:.2f}', ha='center')
    
    # Path cost plot
    axes[0, 1].bar(planners, path_costs, color='lightgreen')
    axes[0, 1].set_title('Path Cost')
    for i, v in enumerate(path_costs):
        axes[0, 1].text(i, v + max(path_costs)*0.05, f'{v:.2f}', ha='center')
    
    # Computation time plot
    axes[1, 0].bar(planners, computation_times, color='lightcoral')
    axes[1, 0].set_title('Computation Time (seconds)')
    for i, v in enumerate(computation_times):
        axes[1, 0].text(i, v + max(computation_times)*0.05, f'{v:.4f}', ha='center')
    
    # Nodes expanded plot
    axes[1, 1].bar(planners, nodes_expanded, color='gold')
    axes[1, 1].set_title('Nodes Expanded')
    for i, v in enumerate(nodes_expanded):
        axes[1, 1].text(i, v + max(nodes_expanded)*0.05, f'{v:.0f}', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_gif_of_path(env, path, output_path, duration=1000):
    """
    Create a GIF visualization of the agent following a path.
    
    Args:
        env: GridEnvironment instance
        path: List of (x, y) positions
        output_path: Path to save the GIF
        duration: Duration between frames in milliseconds
    """
    try:
        from PIL import Image
        import os
        import glob
        
        # Create temporary directory for frames
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create each frame
        frame_paths = []
        for i in range(len(path)):
            frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
            visualize_environment(env, path, i, frame_path)
            frame_paths.append(frame_path)
            plt.close()  # Close the figure to free memory
        
        # Create GIF from frames
        frames = [Image.open(frame) for frame in frame_paths]
        frames[0].save(
            output_path,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=duration,
            loop=0
        )
        
        # Clean up temporary frames
        for frame_path in frame_paths:
            os.remove(frame_path)
        os.rmdir(temp_dir)
        
        print(f"GIF saved to {output_path}")
        
    except ImportError:
        print("PIL library required for GIF creation. Install with: pip install Pillow")

def manhattan_distance(a, b):
    """Calculate Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_distance(a, b):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def chebyshev_distance(a, b):
    """Calculate Chebyshev distance between two points."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))