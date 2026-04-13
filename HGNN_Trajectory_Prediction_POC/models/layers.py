import torch
import torch.nn as nn

class HypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Weights for updating people (nodes) and groups (edges)
        self.lin_node = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_edge = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.zeros_(self.bias)
    
    def forward(self, x, hyperedge_index):
        N = x.size(0)
        num_edges = hyperedge_index[1].max().item() + 1
        
        # Step 1: Project node features
        x_node = self.lin_node(x)
        
        # Step 2: Pool people into groups (Nodes -> Edges)
        hyperedge_attr = torch.zeros(num_edges, x_node.size(1), device=x.device)
        hyperedge_attr.index_add_(0, hyperedge_index[1], x_node[hyperedge_index[0]])
        
        # Average the group features
        node_counts = torch.zeros(num_edges, device=x.device)
        node_counts.index_add_(0, hyperedge_index[1], torch.ones_like(hyperedge_index[1], dtype=torch.float))
        hyperedge_attr = hyperedge_attr / (node_counts.unsqueeze(1) + 1e-8)
        
        # Step 3: Broadcast group info back to individuals (Edges -> Nodes)
        out = torch.zeros_like(x_node)
        out.index_add_(0, hyperedge_index[0], hyperedge_attr[hyperedge_index[1]])
        
        # Average by how many groups a person is in
        degrees = torch.zeros(N, device=x.device)
        degrees.index_add_(0, hyperedge_index[0], torch.ones_like(hyperedge_index[0], dtype=torch.float))
        out = out / (degrees.unsqueeze(1) + 1e-8)
        
        # Add original features (Skip connection)
        return out + x_node + self.bias

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # LSTM reads the past trajectory steps
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        B, T, N, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        
        # Get the final memory state of the LSTM
        _, (h_n, _) = self.lstm(x)
        return h_n[-1].view(B, N, self.hidden_dim)

class TrajectoryDecoder(nn.Module):
    def __init__(self, hidden_dim, pred_len, output_dim=2):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        # One-shot MLP: Predicts all future steps instantly
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * output_dim)
        )
    
    def forward(self, h):
        B, N, _ = h.shape
        out = self.mlp(h)
        
        # Reshape flat output into (Batch, Time, Nodes, XY)
        out = out.view(B, N, self.pred_len, self.output_dim)
        return out.permute(0, 2, 1, 3)
