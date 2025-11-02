"""
Graph Visualization Module

Visualize document graphs and detection results.
"""

from typing import Optional, List
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import numpy as np

from graphplag.core.models import DocumentGraph, PlagiarismMatch


class GraphVisualizer:
    """
    Visualize document graphs and plagiarism detection results.
    """
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize graph visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
    
    def visualize_graph(
        self,
        graph: DocumentGraph,
        highlight_nodes: Optional[List[int]] = None,
        output_file: Optional[str] = None,
        interactive: bool = False
    ):
        """
        Visualize a document graph.
        
        Args:
            graph: DocumentGraph to visualize
            highlight_nodes: Optional list of node IDs to highlight
            output_file: Optional output file path
            interactive: Whether to create interactive visualization
        """
        if interactive:
            self._visualize_interactive(graph, highlight_nodes, output_file)
        else:
            self._visualize_static(graph, highlight_nodes, output_file)
    
    def _visualize_static(
        self,
        graph: DocumentGraph,
        highlight_nodes: Optional[List[int]] = None,
        output_file: Optional[str] = None
    ):
        """Create static matplotlib visualization."""
        G = graph.graph_data
        
        if not isinstance(G, nx.Graph):
            print("Warning: Graph is not in NetworkX format, skipping visualization")
            return
        
        plt.figure(figsize=self.figsize)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Node colors
        node_colors = []
        for node in G.nodes():
            if highlight_nodes and node in highlight_nodes:
                node_colors.append('#ff6b6b')
            else:
                node_colors.append('#4ecdc4')
        
        # Draw
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=500,
            alpha=0.9
        )
        
        nx.draw_networkx_edges(
            G, pos,
            alpha=0.3,
            edge_color='gray',
            width=1
        )
        
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold'
        )
        
        plt.title(f"Document Graph: {graph.document.doc_id or 'Untitled'}", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def _visualize_interactive(
        self,
        graph: DocumentGraph,
        highlight_nodes: Optional[List[int]] = None,
        output_file: Optional[str] = None
    ):
        """Create interactive PyVis visualization."""
        G = graph.graph_data
        
        if not isinstance(G, nx.Graph):
            print("Warning: Graph is not in NetworkX format, skipping visualization")
            return
        
        net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='black')
        
        # Add nodes
        for node in G.nodes():
            sentence_text = G.nodes[node].get('sentence_text', f'Node {node}')
            title = f"<b>Node {node}</b><br>{sentence_text[:100]}..."
            
            color = '#ff6b6b' if highlight_nodes and node in highlight_nodes else '#4ecdc4'
            
            net.add_node(
                node,
                label=str(node),
                title=title,
                color=color,
                size=25
            )
        
        # Add edges
        for edge in G.edges():
            edge_data = G.get_edge_data(*edge)
            weight = edge_data.get('weight', 1.0) if edge_data else 1.0
            
            net.add_edge(
                edge[0],
                edge[1],
                value=weight,
                title=f"Weight: {weight:.3f}"
            )
        
        # Configure physics
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 100
            }
          }
        }
        """)
        
        # Save or show
        output_file = output_file or 'graph_visualization.html'
        net.show(output_file)
        print(f"Interactive visualization saved to: {output_file}")
    
    def visualize_plagiarism_alignment(
        self,
        matches: List[PlagiarismMatch],
        doc1_sentences: List[str],
        doc2_sentences: List[str],
        output_file: Optional[str] = None
    ):
        """
        Visualize plagiarism matches between documents.
        
        Args:
            matches: List of plagiarism matches
            doc1_sentences: Sentences from document 1
            doc2_sentences: Sentences from document 2
            output_file: Optional output file path
        """
        if not matches:
            print("No matches to visualize")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot matches as connections between two columns
        y_doc1 = np.arange(len(doc1_sentences))
        y_doc2 = np.arange(len(doc2_sentences))
        
        # Draw document columns
        ax.scatter([0] * len(doc1_sentences), y_doc1, c='blue', s=100, alpha=0.6, label='Document 1')
        ax.scatter([1] * len(doc2_sentences), y_doc2, c='green', s=100, alpha=0.6, label='Document 2')
        
        # Draw matches
        for match in matches:
            idx1 = match.doc1_segment[0]
            idx2 = match.doc2_segment[0]
            
            if idx1 < len(doc1_sentences) and idx2 < len(doc2_sentences):
                # Color based on similarity
                color_intensity = match.similarity
                color = plt.cm.Reds(color_intensity)
                
                ax.plot([0, 1], [idx1, idx2], 
                       color=color, alpha=0.6, linewidth=2)
        
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-1, max(len(doc1_sentences), len(doc2_sentences)) + 1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Document 1', 'Document 2'])
        ax.set_ylabel('Sentence Index')
        ax.set_title('Plagiarism Alignment Visualization', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_similarity_distribution(
        self,
        scores: List[float],
        threshold: Optional[float] = None,
        output_file: Optional[str] = None
    ):
        """
        Plot distribution of similarity scores.
        
        Args:
            scores: List of similarity scores
            threshold: Optional threshold line to draw
            output_file: Optional output file path
        """
        plt.figure(figsize=self.figsize)
        
        plt.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        if threshold is not None:
            plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold: {threshold:.2f}')
            plt.legend()
        
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Similarity Score Distribution', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def __repr__(self) -> str:
        return f"GraphVisualizer(figsize={self.figsize})"
