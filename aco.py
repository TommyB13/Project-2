import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy

# Step 1: Create the graph
num_nodes = 50
radius = 15
G = nx.random_geometric_graph(num_nodes, radius, dim=2)
pheromone_levels = {edge: 1 for edge in G.edges}  # Initialize pheromone levels

# Parameters for ACO
alpha = 1.0  # Pheromone influence
beta = 2.0   # Distance influence
evaporation_rate = 0.5
num_ants = 10
iterations = 100
source = 0  # Source node
sink = num_nodes - 1  # Sink node
adaptive_evaporation = True  # Adaptive evaporation rate

# Step 2: Define ACO logic
def choose_next_node(current_node, visited):
    neighbors = list(G.neighbors(current_node))
    probabilities = []
    for neighbor in neighbors:
        if neighbor not in visited:
            edge = (min(current_node, neighbor), max(current_node, neighbor))
            pheromone = pheromone_levels[edge]
            distance = np.linalg.norm(np.array(G.nodes[current_node]['pos']) - np.array(G.nodes[neighbor]['pos']))
            prob = (pheromone ** alpha) * ((1.0 / distance) ** beta)
            probabilities.append(prob)
        else:
            probabilities.append(0.0)

    total_prob = sum(probabilities)
    if total_prob == 0:
        return random.choice(neighbors)  # Random fallback
    probabilities = [p / total_prob for p in probabilities]
    return random.choices(neighbors, probabilities)[0]

# Step 3: ACO simulation
def local_search(path):
    # Local search strategy: reverse a segment of the path to see if it improves quality
    if len(path) > 3:
        new_path = deepcopy(path)
        i, j = sorted(random.sample(range(1, len(path) - 1), 2))
        new_path[i:j] = reversed(new_path[i:j])
        return new_path
    return path

for i in range(iterations):
    all_paths = []
    for ant in range(num_ants):
        current_node = source
        visited = [current_node]
        while current_node != sink:
            next_node = choose_next_node(current_node, visited)
            visited.append(next_node)
            current_node = next_node
        optimized_path = local_search(visited)  # Apply local search
        all_paths.append(optimized_path)

    # Adjust evaporation rate dynamically
    if adaptive_evaporation:
        evaporation_rate = 0.3 + 0.2 * (i / iterations)  # Gradually increase evaporation

    # Evaporate pheromone
    for edge in pheromone_levels:
        pheromone_levels[edge] *= (1 - evaporation_rate)

    # Update pheromones based on path quality
    for path in all_paths:
        for j in range(len(path) - 1):
            edge = (min(path[j], path[j + 1]), max(path[j + 1], path[j]))
            pheromone_levels[edge] += 1.0 / len(path)  # Shorter paths get more pheromone

# Step 4: Visualization (optional)
# Get node positions
pos = nx.get_node_attributes(G, 'pos')

# Normalize pheromone levels for visualization
max_pheromone = max(pheromone_levels.values())
weights = [pheromone_levels[edge] / max_pheromone for edge in G.edges]

# Draw the graph
plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue')
nx.draw_networkx_labels(G, pos)

# Draw edges with pheromone levels
nx.draw_networkx_edges(G, pos, width=[w * 5 for w in weights], edge_color=weights, edge_cmap=plt.cm.Blues)

plt.title("ACO Routing with Pheromone Levels")
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), ax=plt.gca(), label="Pheromone Intensity")

# Save the graph
plt.savefig("aco_routing_pheromone_levels.png")
plt.show()
