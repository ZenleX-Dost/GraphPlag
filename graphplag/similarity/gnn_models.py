"""
GNN Similarity Module

Implements Graph Neural Network-based similarity computation.
Uses Siamese GNN architecture for learning graph embeddings.
"""

from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

from graphplag.core.models import DocumentGraph, SimilarityScore


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for document graphs.
    
    Uses GCN or GAT layers to encode graphs into fixed-size embeddings.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3,
        gnn_type: str = "gcn",
        dropout: float = 0.2,
        pooling: str = "mean"
    ):
        """
        Initialize GNN encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ('gcn' or 'gat')
            dropout: Dropout rate
            pooling: Pooling method ('mean', 'max', or 'attention')
        """
        super(GNNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        self.pooling = pooling
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        if self.gnn_type == "gcn":
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif self.gnn_type == "gat":
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
            hidden_dim = hidden_dim * 4  # Adjust for concatenated heads
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if self.gnn_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == "gat":
                self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if self.gnn_type == "gcn":
            self.convs.append(GCNConv(hidden_dim, output_dim))
        elif self.gnn_type == "gat":
            self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))
        
        # Attention pooling (if used)
        if pooling == "attention":
            self.attention = nn.Linear(output_dim, 1)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            data: PyG Data object with graph structure
            
        Returns:
            Graph-level embedding
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)
        
        # Apply GNN layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (no activation)
        x = self.convs[-1](x, edge_index)
        
        # Graph-level pooling
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "attention":
            # Attention-based pooling
            attention_weights = F.softmax(self.attention(x), dim=0)
            x = global_mean_pool(x * attention_weights, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        return x


class SiameseGNN(nn.Module):
    """
    Siamese GNN for computing graph similarity.
    
    Uses shared GNN encoder for both graphs and computes similarity
    based on their embeddings.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3,
        gnn_type: str = "gcn",
        similarity_metric: str = "cosine"
    ):
        """
        Initialize Siamese GNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN
            similarity_metric: Similarity metric ('cosine', 'euclidean', or 'learned')
        """
        super(SiameseGNN, self).__init__()
        
        self.encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            gnn_type=gnn_type
        )
        
        self.similarity_metric = similarity_metric
        
        # Learned similarity (optional)
        if similarity_metric == "learned":
            self.similarity_mlp = nn.Sequential(
                nn.Linear(output_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, data1: Data, data2: Data) -> torch.Tensor:
        """
        Compute similarity between two graphs.
        
        Args:
            data1: First graph
            data2: Second graph
            
        Returns:
            Similarity score
        """
        # Encode both graphs
        emb1 = self.encoder(data1)
        emb2 = self.encoder(data2)
        
        # Compute similarity
        if self.similarity_metric == "cosine":
            similarity = F.cosine_similarity(emb1, emb2, dim=1)
            similarity = (similarity + 1) / 2  # Scale to [0, 1]
        elif self.similarity_metric == "euclidean":
            distance = torch.norm(emb1 - emb2, p=2, dim=1)
            similarity = 1 / (1 + distance)  # Convert distance to similarity
        elif self.similarity_metric == "learned":
            combined = torch.cat([emb1, emb2], dim=1)
            similarity = self.similarity_mlp(combined).squeeze()
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return similarity
    
    def encode(self, data: Data) -> torch.Tensor:
        """Encode a graph to embedding."""
        return self.encoder(data)


class GNNSimilarity:
    """
    Compute similarity using trained GNN model.
    """
    
    def __init__(
        self,
        model: Optional[SiameseGNN] = None,
        model_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize GNN similarity computer.
        
        Args:
            model: Pre-initialized Siamese GNN model
            model_path: Path to saved model weights
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        
        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            # Default model (will need training)
            self.model = SiameseGNN(
                input_dim=768,  # Default embedding size
                hidden_dim=128,
                output_dim=64
            ).to(self.device)
        
        self.model.eval()
    
    def compute_similarity(
        self,
        graph1: DocumentGraph,
        graph2: DocumentGraph
    ) -> SimilarityScore:
        """
        Compute similarity between two document graphs using GNN.
        
        Args:
            graph1: First document graph
            graph2: Second document graph
            
        Returns:
            SimilarityScore object
        """
        # Convert to PyG format if needed
        data1 = self._prepare_graph(graph1)
        data2 = self._prepare_graph(graph2)
        
        # Move to device
        data1 = data1.to(self.device)
        data2 = data2.to(self.device)
        
        # Compute similarity
        with torch.no_grad():
            similarity = self.model(data1, data2)
        
        score = float(similarity.item())
        
        return SimilarityScore(
            score=score,
            method="gnn",
            confidence=score,
            details={
                "model_type": type(self.model).__name__,
                "device": str(self.device)
            }
        )
    
    def encode_graph(self, graph: DocumentGraph) -> np.ndarray:
        """
        Encode a graph to an embedding vector.
        
        Args:
            graph: Document graph
            
        Returns:
            Embedding as numpy array
        """
        data = self._prepare_graph(graph).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode(data)
        
        return embedding.cpu().numpy()
    
    def _prepare_graph(self, graph: DocumentGraph) -> Data:
        """Prepare graph for GNN processing."""
        import torch
        import networkx as nx
        
        # Check if already PyG Data object
        if isinstance(graph.graph_data, Data):
            return graph.graph_data
        
        # Convert from NetworkX or GraKeL to PyG format
        try:
            # Try to get the NetworkX graph
            if hasattr(graph.graph_data, 'graph_'):
                # It's a GraKeL Graph object - get the underlying NetworkX
                nx_graph = graph.graph_data.graph_
            elif isinstance(graph.graph_data, nx.Graph):
                nx_graph = graph.graph_data
            else:
                # Fallback: reconstruct from nodes and edges
                nx_graph = nx.Graph()
                for node in graph.nodes:
                    nx_graph.add_node(node.node_id, features=node.features)
                for edge in graph.edges:
                    nx_graph.add_edge(edge.source, edge.target, weight=edge.weight)
            
            # Convert NetworkX to PyG Data
            nodes = sorted(nx_graph.nodes())
            node_mapping = {node: idx for idx, node in enumerate(nodes)}
            
            # Extract node features
            node_features = []
            for node in nodes:
                if 'features' in nx_graph.nodes[node]:
                    features = nx_graph.nodes[node]['features']
                else:
                    # Use node features from DocumentGraph
                    node_obj = next((n for n in graph.nodes if n.node_id == node), None)
                    features = node_obj.features if node_obj else np.zeros(768)
                node_features.append(features)
            
            x = torch.tensor(np.array(node_features), dtype=torch.float)
            
            # Extract edges
            edge_list = []
            for edge in nx_graph.edges():
                src_idx = node_mapping[edge[0]]
                tgt_idx = node_mapping[edge[1]]
                edge_list.append([src_idx, tgt_idx])
                edge_list.append([tgt_idx, src_idx])  # Add reverse edge for undirected
            
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            else:
                # No edges - create empty edge index
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            return Data(x=x, edge_index=edge_index)
            
        except Exception as e:
            # If all else fails, create a simple graph from the document nodes
            node_features = np.array([node.features for node in graph.nodes])
            x = torch.tensor(node_features, dtype=torch.float)
            
            # Create sequential edges
            edge_list = []
            for i in range(len(graph.nodes) - 1):
                edge_list.append([i, i + 1])
                edge_list.append([i + 1, i])
            
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            return Data(x=x, edge_index=edge_index)
    
    def _load_model(self, model_path: str) -> SiameseGNN:
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Reconstruct model
        model = SiameseGNN(
            input_dim=checkpoint.get('input_dim', 768),
            hidden_dim=checkpoint.get('hidden_dim', 128),
            output_dim=checkpoint.get('output_dim', 64),
            num_layers=checkpoint.get('num_layers', 3),
            gnn_type=checkpoint.get('gnn_type', 'gcn')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def train_model(
        self,
        train_data: List[Tuple[DocumentGraph, DocumentGraph, float]],
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001
    ):
        """
        Train the GNN model on labeled pairs.
        
        Args:
            train_data: List of (graph1, graph2, similarity_label) tuples
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Simple batch processing (can be improved with DataLoader)
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                optimizer.zero_grad()
                
                batch_loss = 0.0
                for graph1, graph2, label in batch:
                    data1 = self._prepare_graph(graph1).to(self.device)
                    data2 = self._prepare_graph(graph2).to(self.device)
                    
                    pred = self.model(data1, data2)
                    target = torch.tensor([label], dtype=torch.float, device=self.device)
                    
                    loss = criterion(pred, target)
                    batch_loss += loss
                
                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
            
            avg_loss = total_loss / (len(train_data) // batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def __repr__(self) -> str:
        return f"GNNSimilarity(model={type(self.model).__name__}, device={self.device})"
