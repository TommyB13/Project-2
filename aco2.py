import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy
import math
import time

def run_aco_simulation(G, pheromone_levels, num_ants, iterations, source, sink, adaptive_evaporation=False, apply_local_search=False, multi_pheromone=False, secondary_pheromone_levels=None):
    # Parameters for ACO
    # Initial values for alpha and beta
    alpha = 1.0  # Pheromone influence
    beta = 2.0   # Distance influence
    evaporation_rate = 0.5  # Initial evaporation rate

    def choose_next_node(current_node, visited):
        neighbors = list(G.neighbors(current_node))
        probabilities = []
        for neighbor in neighbors:
            if neighbor not in visited:
                edge = (min(current_node, neighbor), max(current_node, neighbor))
                pheromone = pheromone_levels[edge]
                distance = np.linalg.norm(np.array(G.nodes[current_node]['pos']) - np.array(G.nodes[neighbor]['pos']))
                prob = (pheromone ** alpha) * ((1.0 / distance) ** beta)
                if multi_pheromone and secondary_pheromone_levels:
                    secondary_pheromone = secondary_pheromone_levels[edge]
                    prob *= (secondary_pheromone ** 0.5)  # Weight secondary pheromone
                probabilities.append(prob)
            else:
                probabilities.append(0.0)

        total_prob = sum(probabilities)
        if total_prob == 0:
            return random.choice(neighbors)  # Random fallback
        probabilities = [p / total_prob for p in probabilities]
        return random.choices(neighbors, probabilities)[0]

    def local_search(path):
        # Local search strategy: reverse a segment of the path to see if it improves quality
        if len(path) > 3:
            new_path = deepcopy(path)
            i, j = sorted(random.sample(range(1, len(path) - 1), 2))
            new_path[i:j] = reversed(new_path[i:j])
            return new_path
        return path

    best_path = None
    best_path_length = float('inf')
    convergence_iteration = None
    all_path_lengths = []

    start_time = time.time()

    for i in range(iterations):
        # Calculate pheromone entropy before adjusting parameters
        pheromone_values = np.array(list(pheromone_levels.values()))
        pheromone_probabilities = pheromone_values / pheromone_values.sum()
        pheromone_entropy = -np.sum(pheromone_probabilities * np.log(pheromone_probabilities + 1e-9))  # Small epsilon to avoid log(0)
        # Adjust alpha and beta dynamically based on previous success and pheromone entropy
        if i > 0 and len(all_path_lengths) > 1:
            if all_path_lengths[-1] < all_path_lengths[-2]:
                # If the average path length improved, increase exploitation
                alpha = min(3.0, alpha + 0.05)  # Cap alpha to avoid over-exploitation
                beta = max(1.0, beta - 0.05)    # Decrease beta to reduce exploration
            else:
                # If the average path length did not improve, increase exploration
                alpha = max(1.0, alpha - 0.05)
                beta = min(3.0, beta + 0.05)  # Cap beta to avoid over-exploration
            # Adjust parameters based on pheromone entropy
            if pheromone_entropy < 1.5:
                # Low entropy: increase exploration
                alpha = max(1.0, alpha - 0.05)
                beta = min(3.0, beta + 0.05)
            elif pheromone_entropy > 2.5:
                # High entropy: increase exploitation
                alpha = min(3.0, alpha + 0.05)
                beta = max(1.0, beta - 0.05)
        
        all_paths = []
        total_path_length = 0
        for ant in range(num_ants):
            current_node = source
            visited = [current_node]
            while current_node != sink:
                next_node = choose_next_node(current_node, visited)
                visited.append(next_node)
                current_node = next_node
            if apply_local_search:
                visited = local_search(visited)  # Apply local search if enabled
            all_paths.append(visited)
            total_path_length += len(visited)

            # Update best path
            if len(visited) < best_path_length:
                best_path = visited
                best_path_length = len(visited)
                convergence_iteration = i

        all_path_lengths.append(total_path_length / num_ants)  # Average path length for this iteration

        # Adjust evaporation rate dynamically
        if adaptive_evaporation:
            if pheromone_entropy > 2.5:
                evaporation_rate = max(0.3, evaporation_rate - 0.01)  # Decrease evaporation to retain pheromone longer
            elif pheromone_entropy < 1.5:
                evaporation_rate = min(0.7, evaporation_rate + 0.01)  # Increase evaporation to promote exploration  # Gradually increase evaporation

        # Evaporate pheromone
        for edge in pheromone_levels:
            pheromone_levels[edge] *= (1 - evaporation_rate)
            if multi_pheromone and secondary_pheromone_levels:
                secondary_pheromone_levels[edge] *= (1 - evaporation_rate)

        # Update pheromones based on path quality
        for path in all_paths:
            for j in range(len(path) - 1):
                edge = (min(path[j], path[j + 1]), max(path[j + 1], path[j]))
                pheromone_levels[edge] += 1.0 / len(path)  # Shorter paths get more pheromone
                if multi_pheromone and secondary_pheromone_levels:
                    secondary_pheromone_levels[edge] += 0.5 / len(path)  # Update secondary pheromone

    end_time = time.time()
    runtime = end_time - start_time

      # Small epsilon to avoid log(0)

    # Output metrics
    return {
        "best_path_length": best_path_length,
        "convergence_iteration": convergence_iteration if convergence_iteration is not None else "Did not converge",
        "average_path_length": np.mean(all_path_lengths),
        "pheromone_entropy": pheromone_entropy,
        "runtime": runtime
    }

# Step 1: Run the simulations on multiple graphs
num_graphs = 10
num_nodes = 50
radius = 15
results_without_optimizations = []
results_with_optimizations = []
results_multi_pheromone = []

for i in range(num_graphs):
    print(f"\nRunning simulations on graph {i + 1}/{num_graphs}...")
    G = nx.random_geometric_graph(num_nodes, radius, dim=2)
    pos = nx.spring_layout(G, seed=42)

    # Run ACO without optimizations
    print("Running ACO without optimizations...")
    pheromone_levels_initial = {edge: 1 for edge in G.edges}  # Initialize pheromone levels
    result = run_aco_simulation(G, pheromone_levels_initial, num_ants=10, iterations=1000, source=0, sink=num_nodes - 1)
    results_without_optimizations.append(result)

    # Run ACO with optimizations
    print("Running ACO with optimizations...")
    pheromone_levels_optimized = {edge: 1 for edge in G.edges}  # Reinitialize pheromone levels
    result = run_aco_simulation(G, pheromone_levels_optimized, num_ants=10, iterations=1000, source=0, sink=num_nodes - 1, adaptive_evaporation=True, apply_local_search=True)
    results_with_optimizations.append(result)

    # Run ACO with adaptive multi-pheromone system
    print("Running ACO with adaptive multi-pheromone system...")
    primary_pheromone_levels = {edge: 1 for edge in G.edges}  # Primary pheromone levels
    secondary_pheromone_levels = {edge: 1 for edge in G.edges}  # Secondary pheromone levels
    result = run_aco_simulation(G, primary_pheromone_levels, num_ants=10, iterations=1000, source=0, sink=num_nodes - 1, multi_pheromone=True, secondary_pheromone_levels=secondary_pheromone_levels)
    results_multi_pheromone.append(result)

    # Run ACO with adaptive multi-pheromone system and dynamic alpha-beta
    print("Running ACO with adaptive multi-pheromone system and dynamic alpha-beta...")
    primary_pheromone_levels_dynamic = {edge: 1 for edge in G.edges}  # Primary pheromone levels
    secondary_pheromone_levels_dynamic = {edge: 1 for edge in G.edges}  # Secondary pheromone levels
    result = run_aco_simulation(G, primary_pheromone_levels_dynamic, num_ants=10, iterations=1000, source=0, sink=num_nodes - 1, multi_pheromone=True, secondary_pheromone_levels=secondary_pheromone_levels_dynamic)
    results_multi_pheromone.append(result)

    # Run ACO with hybrid implementation
    print("Running ACO with hybrid implementation...")
    hybrid_pheromone_levels = {edge: 1 for edge in G.edges}  # Hybrid pheromone levels
    hybrid_secondary_pheromone_levels = {edge: 1 for edge in G.edges}  # Secondary pheromone levels
    result = run_aco_simulation(G, hybrid_pheromone_levels, num_ants=10, iterations=1000, source=0, sink=num_nodes - 1, adaptive_evaporation=True, apply_local_search=True, multi_pheromone=True, secondary_pheromone_levels=hybrid_secondary_pheromone_levels)
    results_multi_pheromone.append(result)

# Step 2: Calculate average metrics for each implementation
def calculate_average_metrics(results):
    avg_best_path_length = np.mean([r["best_path_length"] for r in results])
    avg_convergence_iteration = np.mean([r["convergence_iteration"] if isinstance(r["convergence_iteration"], int) else iterations for r in results])
    avg_average_path_length = np.mean([r["average_path_length"] for r in results])
    avg_pheromone_entropy = np.mean([r["pheromone_entropy"] for r in results])
    avg_runtime = np.mean([r["runtime"] for r in results])
    return {
        "avg_best_path_length": avg_best_path_length,
        "avg_convergence_iteration": avg_convergence_iteration,
        "avg_average_path_length": avg_average_path_length,
        "avg_pheromone_entropy": avg_pheromone_entropy,
        "avg_runtime": avg_runtime
    }

average_metrics_without_optimizations = calculate_average_metrics(results_without_optimizations)
average_metrics_with_optimizations = calculate_average_metrics(results_with_optimizations)
average_metrics_multi_pheromone = calculate_average_metrics(results_multi_pheromone[:num_graphs])
average_metrics_multi_pheromone_dynamic = calculate_average_metrics(results_multi_pheromone[num_graphs:num_graphs*2])
average_metrics_hybrid = calculate_average_metrics(results_multi_pheromone[num_graphs*2:])

# Step 3: Print average metrics
print("\nAverage Metrics for ACO without Optimizations:")
print(f"  - Average Best Path Length: {average_metrics_without_optimizations['avg_best_path_length']:.2f}")
print(f"  - Average Convergence Iteration: {average_metrics_without_optimizations['avg_convergence_iteration']:.2f}")
print(f"  - Average Path Length Over Iterations: {average_metrics_without_optimizations['avg_average_path_length']:.2f}")
print(f"  - Average Pheromone Entropy: {average_metrics_without_optimizations['avg_pheromone_entropy']:.2f}")
print(f"  - Average Runtime (seconds): {average_metrics_without_optimizations['avg_runtime']:.2f}")

print("\nAverage Metrics for ACO with Optimizations:")
print(f"  - Average Best Path Length: {average_metrics_with_optimizations['avg_best_path_length']:.2f}")
print(f"  - Average Convergence Iteration: {average_metrics_with_optimizations['avg_convergence_iteration']:.2f}")
print(f"  - Average Path Length Over Iterations: {average_metrics_with_optimizations['avg_average_path_length']:.2f}")
print(f"  - Average Pheromone Entropy: {average_metrics_with_optimizations['avg_pheromone_entropy']:.2f}")
print(f"  - Average Runtime (seconds): {average_metrics_with_optimizations['avg_runtime']:.2f}")

print("\nAverage Metrics for ACO with Adaptive Multi-Pheromone System:")
print(f"  - Average Best Path Length: {average_metrics_multi_pheromone['avg_best_path_length']:.2f}")
print(f"  - Average Convergence Iteration: {average_metrics_multi_pheromone['avg_convergence_iteration']:.2f}")
print(f"  - Average Path Length Over Iterations: {average_metrics_multi_pheromone['avg_average_path_length']:.2f}")
print(f"  - Average Pheromone Entropy: {average_metrics_multi_pheromone['avg_pheromone_entropy']:.2f}")
print(f"  - Average Runtime (seconds): {average_metrics_multi_pheromone['avg_runtime']:.2f}")

print("\nAverage Metrics for ACO with Adaptive Multi-Pheromone System and Dynamic Alpha-Beta:")
print(f"  - Average Best Path Length: {average_metrics_multi_pheromone_dynamic['avg_best_path_length']:.2f}")
print(f"  - Average Convergence Iteration: {average_metrics_multi_pheromone_dynamic['avg_convergence_iteration']:.2f}")
print(f"  - Average Path Length Over Iterations: {average_metrics_multi_pheromone_dynamic['avg_average_path_length']:.2f}")
print(f"  - Average Pheromone Entropy: {average_metrics_multi_pheromone_dynamic['avg_pheromone_entropy']:.2f}")
print(f"  - Average Runtime (seconds): {average_metrics_multi_pheromone_dynamic['avg_runtime']:.2f}")

print("\nAverage Metrics for ACO with Hybrid Implementation:")
print(f"  - Average Best Path Length: {average_metrics_hybrid['avg_best_path_length']:.2f}")
print(f"  - Average Convergence Iteration: {average_metrics_hybrid['avg_convergence_iteration']:.2f}")
print(f"  - Average Path Length Over Iterations: {average_metrics_hybrid['avg_average_path_length']:.2f}")
print(f"  - Average Pheromone Entropy: {average_metrics_hybrid['avg_pheromone_entropy']:.2f}")
print(f"  - Average Runtime (seconds): {average_metrics_hybrid['avg_runtime']:.2f}")
