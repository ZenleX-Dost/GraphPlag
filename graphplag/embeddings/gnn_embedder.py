"""
GNN Embedder Module

Generates vector embeddings from graphs or text using pre-trained models.
"""

import numpy as np
from typing import Union
from sentence_transformers import SentenceTransformer


class GNNEmbedder:
    """
    Generate embeddings from graphs or text.
    Uses sentence transformers for semantic embeddings.
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        """
        Initialize GNN embedder.
        
        Args:
            model_name: Sentence transformer model to use
        """
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Failed to load {model_name}, using fallback model")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model_name = 'all-MiniLM-L6-v2'
    
    def embed(self, graph_or_text: Union[object, str]) -> np.ndarray:
        """
        Generate embedding from graph or text.
        
        Args:
            graph_or_text: NetworkX graph or text string
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Check if input is a graph (has nodes attribute)
            if hasattr(graph_or_text, 'nodes'):
                # Extract text from graph nodes
                text_parts = []
                for _, node_data in graph_or_text.nodes(data=True):
                    # Try different possible text attributes
                    text = (
                        node_data.get('text') or 
                        node_data.get('content') or 
                        node_data.get('label') or 
                        str(node_data)
                    )
                    if text:
                        text_parts.append(text)
                
                # Combine all text
                combined_text = ' '.join(text_parts)[:10000]  # Limit to 10k chars
                
                if not combined_text.strip():
                    # Empty graph, return zero vector
                    return np.zeros(self.model.get_sentence_embedding_dimension())
            else:
                # Treat as text
                combined_text = str(graph_or_text)[:10000]
            
            # Generate embedding
            embedding = self.model.encode(combined_text)
            return embedding
            
        except Exception as e:
            print(f"Embedding generation error: {e}")
            # Return zero vector as fallback
            try:
                dim = self.model.get_sentence_embedding_dimension()
            except:
                dim = 768  # Default dimension
            return np.zeros(dim)
    
    def embed_batch(self, graphs_or_texts: list) -> np.ndarray:
        """
        Generate embeddings for multiple inputs.
        
        Args:
            graphs_or_texts: List of graphs or texts
            
        Returns:
            Array of embeddings
        """
        embeddings = [self.embed(item) for item in graphs_or_texts]
        return np.array(embeddings)
    
    def __repr__(self) -> str:
        return f"GNNEmbedder(model='{self.model_name}')"
