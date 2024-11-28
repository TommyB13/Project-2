import random

# Parameters for ACO
pheromone_levels = {edge: 1 for edge in G.edges}  # Initialize pheromone levels
alpha = 1.0  # Pheromone influence
beta = 2.0   # Distance influence
evaporation_rate = 0.5
num_ants = 10
iterations = 100
source = 0  # Source node
sink = num_nodes - 1  # Sink node

# Function to choose next node based on pheromone and distance
def choose_next_node(current_node, visited):
    neighbors = list(G.neighbors(current_node))
    probabilities = []
    
    for neighbor in neighbors:
        if neighbor not in visited:
            pheromone = pheromone_levels[(min(current_node, neighbor), max(current_node, neighbor))]
            distance = np.linalg.norm(np.array(pos[current_node]) - np.array(pos[neighbor]))
            prob = (pheromone ** alpha) * ((1.0 / distance) ** beta)
            probabilities.append(prob)
        else:
            probabilities.append(0.0)
    
    total_prob = sum(probabilities)
    if total_prob == 0:
        return random.choice(neighbors)  # Random fallback
    probabilities = [p / total_prob for p in probabilities]
    return random.choices(neighbors, probabilities)[0]

# ACO main loop
for i in range(iterations):
    all_paths = []
    for ant in range(num_ants):
        current_node = source
        visited = [current_node]
        while current_node != sink:
            next_node = choose_next_node(current_node, visited)
            visited.append(next_node)
            current_node = next_node
        all_paths.append(visited)
    
    # Evaporate pheromone
    for edge in pheromone_levels:
        pheromone_levels[edge] *= (1 - evaporation_rate)

    # Update pheromones based on path quality
    for path in all_paths:
        for j in range(len(path) - 1):
            edge = (min(path[j], path[j + 1]), max(path[j], path[j + 1]))
            pheromone_levels[edge] += 1.0 / len(path)  # Shorter paths get more pheromone

# Visualize the pheromone levels (darker edges mean higher pheromone)
edges, weights = zip(*nx.get_edge_attributes(G, 'pheromone').items())
nx.draw(G, pos, edges=edges, edge_color=weights, width=2.0, edge_cmap=plt.cm.Blues)
plt.show()
