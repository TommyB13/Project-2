import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Set up the network
num_nodes = 50
network_area = (100, 100)

# Create a random WSN environment
G = nx.random_geometric_graph(num_nodes, radius=15, dim=network_area)
pos = nx.get_node_attributes(G, 'pos')  # Get node positions

# Plot the WSN
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue')
plt.show()
