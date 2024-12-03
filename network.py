import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_nodes = 50
network_area = (100, 100)  # Define the area as a tuple (width, height)
radius = 15

# Create a random geometric graph in 2D
G = nx.random_geometric_graph(num_nodes, radius, dim=2)
pos = nx.get_node_attributes(G, 'pos')  # Get node positions

# Plot the network
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue')
plt.title("Random Geometric Graph for WSN")
plt.show()
