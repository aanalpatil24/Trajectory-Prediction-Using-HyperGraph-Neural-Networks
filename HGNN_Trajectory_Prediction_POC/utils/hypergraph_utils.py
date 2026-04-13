import torch

class HypergraphConstructor:
    def __init__(self, radius=10.0, min_size=2, max_size=5):
        # Settings for grouping people
        self.radius = radius
        self.min_size = min_size
        self.max_size = max_size
    
    def construct_from_positions(self, positions):
        N = positions.size(0)
        
        # Fast GPU-native distance calculation (replaces Scikit-Learn DBSCAN)
        dist_matrix = torch.cdist(positions, positions)
        
        visited = torch.zeros(N, dtype=torch.bool, device=positions.device)
        hyperedges = []
        
        # Group people who have the most neighbors first
        degrees = (dist_matrix < self.radius).sum(dim=1)
        node_order = torch.argsort(degrees, descending=True)
        
        for i in node_order:
            if visited[i]: continue
            
            # Find everyone within radius
            neighbors = (dist_matrix[i] < self.radius).nonzero(as_tuple=True)[0]
            
            if len(neighbors) >= self.min_size:
                # Cap the group size to prevent massive edges
                if len(neighbors) > self.max_size:
                    dists = dist_matrix[i, neighbors]
                    neighbors = neighbors[torch.argsort(dists)[:self.max_size]]
                
                hyperedges.append(neighbors)
                visited[neighbors] = True
                
        # Give loners their own group so the network doesn't crash
        for i in range(N):
            if not visited[i]:
                hyperedges.append(torch.tensor([i], device=positions.device))
                
        # Format for PyTorch: Row 0 is Person ID, Row 1 is Group ID
        node_indices, edge_indices = [], []
        for he_idx, nodes in enumerate(hyperedges):
            node_indices.extend(nodes.tolist())
            edge_indices.extend([he_idx] * len(nodes))
            
        hyperedge_index = torch.tensor([node_indices, edge_indices], device=positions.device)
        return positions, hyperedge_index
