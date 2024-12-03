# Project-2

Bio-inspired algorithm on Wireless Ad-Hoc Networks (WANET)



### Results

Based on the graphs you've provided, here's what they indicate:

1. **Graph without Optimizations**:
   
   - The pheromone levels seem to be distributed across many different paths, resulting in a large number of thick edges throughout the graph.
   - This suggests that without optimizations, ants are not converging efficiently to an optimal path. They are exploring a broader set of routes, leading to increased pheromone distribution along multiple paths, which can slow down convergence and result in less efficient routing.

2. **Graph with Optimizations**:
   
   - With optimizations, the pheromone levels are concentrated along fewer paths, and we see more distinct thick edges, especially toward a specific route to the sink node.
   - This indicates that the optimization mechanisms (adaptive evaporation and local search) have helped the ants converge more effectively towards a specific path.
   - The thick edges and higher pheromone intensity on specific routes show that the algorithm is focusing more on a few efficient paths, which is what we expect from a successful optimization—better exploitation of good paths with less exploration of suboptimal routes.

##### Key Differences:

- **Without Optimizations**:
  
  - The graph has many thick lines throughout, indicating that the pheromones are distributed more evenly across the graph, leading to a slower and less directed convergence.
  - It highlights less efficient path finding, where ants deposit pheromones on a larger number of routes.

- **With Optimizations**:
  
  - The graph shows fewer thick lines concentrated on specific routes, indicating the ants are more consistently choosing certain paths.
  - This suggests a faster convergence and more optimal route selection, as the pheromone is concentrated where the paths are better in terms of efficiency.

##### What You Can Conclude:

- The optimizations applied to the ACO algorithm (adaptive pheromone evaporation and local search) have led to better pathfinding performance.
- **Adaptive Evaporation**: By dynamically adjusting the pheromone evaporation rate, the algorithm balances exploration and exploitation more effectively, allowing it to prevent premature convergence while still focusing on optimal paths as iterations proceed.
- **Local Search**: The use of local search helps improve the quality of each path found, making the ants more likely to reinforce better routes.

##### Improvements Shown by Graphs:

- The optimization enhancements help in reducing the number of active paths, which leads to a more efficient and optimal route discovery process.
- The optimized graph is closer to what we want from ACO—finding the shortest or most efficient path consistently while avoiding suboptimal routes.
