"""
Example usage of GraphPlag system.

Demonstrates basic plagiarism detection workflow.
"""

from graphplag import PlagiarismDetector
from graphplag.detection.report_generator import ReportGenerator
from graphplag.utils.visualization import GraphVisualizer


def basic_example():
    """Basic plagiarism detection between two documents."""
    
    print("Basic Plagiarism Detection Example")
    print("=" * 60)
    
    # Sample documents
    doc1 = """
    Machine learning is a subset of artificial intelligence that focuses on 
    the development of algorithms and statistical models. These models enable 
    computer systems to improve their performance on a specific task through 
    experience. Machine learning algorithms build a mathematical model based on 
    sample data, known as training data, in order to make predictions or decisions 
    without being explicitly programmed to do so.
    """
    
    doc2 = """
    Machine learning represents a branch of artificial intelligence concerned with 
    creating algorithms and statistical models. Such models allow computer systems 
    to enhance their performance on particular tasks through experience. These 
    algorithms construct mathematical models using sample data called training data, 
    enabling them to make predictions or decisions without explicit programming.
    """
    
    # Initialize detector
    detector = PlagiarismDetector(
        method='kernel',  # Use graph kernels only for this example
        threshold=0.7,
        language='en'
    )
    
    # Detect plagiarism
    print("\nAnalyzing documents...")
    report = detector.detect_plagiarism(doc1, doc2, doc1_id="doc1", doc2_id="doc2")
    
    # Print results
    print("\n" + report.summary())
    
    # Generate HTML report
    report_gen = ReportGenerator(output_dir="./examples/reports")
    report_gen.save_report(report, filename="basic_example.html")
    
    return report


def batch_comparison_example():
    """Example of comparing multiple documents."""
    
    print("\nBatch Comparison Example")
    print("=" * 60)
    
    documents = [
        "Python is a high-level programming language.",
        "Python is a high-level programming language widely used in data science.",
        "Java is an object-oriented programming language.",
        "Machine learning is a field of artificial intelligence.",
        "Python is commonly used for machine learning applications."
    ]
    
    doc_ids = [f"doc_{i+1}" for i in range(len(documents))]
    
    # Initialize detector
    detector = PlagiarismDetector(method='kernel', threshold=0.6)
    
    # Compute similarity matrix
    print("\nComputing pairwise similarities...")
    similarity_matrix = detector.batch_compare(documents, doc_ids)
    
    print("\nSimilarity Matrix:")
    print(similarity_matrix)
    
    # Find suspicious pairs
    suspicious_pairs = detector.identify_suspicious_pairs(documents, doc_ids)
    
    print(f"\nFound {len(suspicious_pairs)} suspicious pairs:")
    for idx1, idx2, score in suspicious_pairs:
        print(f"  {doc_ids[idx1]} <-> {doc_ids[idx2]}: {score:.3f}")
    
    # Visualize
    report_gen = ReportGenerator(output_dir="./examples/reports")
    report_gen.plot_similarity_heatmap(
        similarity_matrix,
        labels=doc_ids,
        output_file="./examples/reports/similarity_heatmap.png"
    )
    
    return similarity_matrix


def graph_visualization_example():
    """Example of visualizing document graphs."""
    
    print("\nGraph Visualization Example")
    print("=" * 60)
    
    from graphplag.core.document_parser import DocumentParser
    from graphplag.core.graph_builder import GraphBuilder
    
    text = """
    Artificial intelligence is revolutionizing many industries. Machine learning,
    a subset of AI, enables systems to learn from data. Deep learning uses neural
    networks with multiple layers. These technologies are transforming healthcare,
    finance, and transportation.
    """
    
    # Parse and build graph
    parser = DocumentParser(language='en')
    builder = GraphBuilder()
    
    print("\nParsing document...")
    document = parser.parse_document(text, doc_id="example_doc")
    
    print(f"Parsed {len(document.sentences)} sentences")
    
    print("\nBuilding graph...")
    graph = builder.build_graph(document)
    
    print(f"Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Visualize
    visualizer = GraphVisualizer()
    visualizer.visualize_graph(
        graph,
        output_file="./examples/visualizations/document_graph.png"
    )
    
    print("\nGraph visualization saved to: ./examples/visualizations/document_graph.png")
    
    return graph


def ensemble_detection_example():
    """Example using ensemble of kernel and GNN methods."""
    
    print("\nEnsemble Detection Example")
    print("=" * 60)
    
    doc1 = """
    Climate change is one of the most pressing issues facing humanity today.
    Rising global temperatures are causing sea levels to rise, extreme weather
    events to become more frequent, and ecosystems to be disrupted. Scientists
    agree that human activities, particularly the burning of fossil fuels, are
    the primary driver of these changes.
    """
    
    doc2 = """
    The climate crisis represents humanity's greatest challenge in the modern era.
    Increasing planetary temperatures lead to rising ocean levels, more frequent
    severe weather patterns, and ecological system disruptions. The scientific
    consensus confirms that human actions, especially fossil fuel combustion,
    are the main cause of these environmental shifts.
    """
    
    # Note: For full ensemble, you would need a trained GNN model
    # For this example, we'll use kernel-only
    detector = PlagiarismDetector(
        method='kernel',
        threshold=0.65
    )
    
    print("\nAnalyzing documents with ensemble method...")
    report = detector.detect_plagiarism(doc1, doc2)
    
    print("\n" + report.summary())
    
    if report.matches:
        print(f"\nTop 5 matches:")
        for i, match in enumerate(report.matches[:5], 1):
            print(f"  {i}. Sentences {match.doc1_segment} <-> {match.doc2_segment}: {match.similarity:.3f}")
    
    return report


def main():
    """Run all examples."""
    
    print("\n" + "=" * 60)
    print("GraphPlag - Example Usage")
    print("=" * 60 + "\n")
    
    # Create output directories
    import os
    os.makedirs("./examples/reports", exist_ok=True)
    os.makedirs("./examples/visualizations", exist_ok=True)
    
    # Run examples
    try:
        # Example 1: Basic detection
        report1 = basic_example()
        
        # Example 2: Batch comparison
        similarity_matrix = batch_comparison_example()
        
        # Example 3: Graph visualization
        graph = graph_visualization_example()
        
        # Example 4: Ensemble detection
        report2 = ensemble_detection_example()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
