#!/usr/bin/env python
"""
Example: Using AI Detection and Integrated Detector

This script demonstrates how to use GraphPlag's AI detection capabilities
to detect both plagiarism and AI-generated content.
"""

from graphplag.detection.ai_detector import AIDetector
from graphplag.detection.integrated_detector import IntegratedDetector


def example_ai_detection_only():
    """Example 1: Detect if text was written by AI"""
    print("=" * 70)
    print("EXAMPLE 1: AI Content Detection Only")
    print("=" * 70)
    
    # Sample texts
    human_text = """
    The evolution of artificial intelligence has been remarkable over the past decade.
    From early rule-based systems to deep learning neural networks, AI has transformed
    how we approach problem-solving. However, with great power comes great responsibility.
    We must consider ethical implications and ensure AI systems are fair and transparent.
    """
    
    ai_text = """
    Artificial intelligence has undergone significant evolution throughout the past decade.
    The progression from foundational rule-based systems to contemporary deep learning
    neural networks has fundamentally transformed methodologies for problem-solving approaches.
    Nevertheless, this technological advancement necessitates careful consideration of ethical
    implications, ensuring that AI systems maintain fairness and operate with transparency.
    """
    
    # Initialize detector
    detector = AIDetector()
    
    # Detect human text
    print("\nAnalyzing human-written text...")
    result_human = detector.detect_ai_content(human_text)
    print(f"Is AI-generated: {result_human['is_ai']}")
    print(f"Confidence: {result_human['confidence']:.2%}")
    print(f"Method scores: {result_human['scores']}")
    
    # Detect AI text
    print("\nAnalyzing AI-written text...")
    result_ai = detector.detect_ai_content(ai_text)
    print(f"Is AI-generated: {result_ai['is_ai']}")
    print(f"Confidence: {result_ai['confidence']:.2%}")
    print(f"Method scores: {result_ai['scores']}")


def example_plagiarism_detection():
    """Example 2: Detect plagiarism between documents"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Plagiarism Detection Only")
    print("=" * 70)
    
    from graphplag import PlagiarismDetector
    
    doc1 = """
    Machine learning is a subset of artificial intelligence that focuses on enabling
    computers to learn from data without being explicitly programmed. The core idea is
    to feed data into algorithms that can identify patterns and make predictions.
    """
    
    doc2 = """
    Machine learning represents a subset of AI that emphasizes teaching computers to
    learn from data without explicit programming. The fundamental concept involves
    inputting data into algorithms capable of recognizing patterns and generating predictions.
    """
    
    detector = PlagiarismDetector(method="ensemble")
    report = detector.detect_plagiarism(doc1, doc2)
    
    print(f"Similarity Score: {report.similarity_score:.2%}")
    print(f"Is Plagiarized: {report.is_plagiarism}")
    print(f"Number of Matches: {len(report.matches) if report.matches else 0}")


def example_integrated_detection():
    """Example 3: Combined plagiarism + AI detection"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Integrated Plagiarism + AI Detection")
    print("=" * 70)
    
    # Sample documents
    doc1 = """
    The Internet of Things (IoT) refers to the vast network of physical devices,
    vehicles, and appliances embedded with sensors and connectivity. These devices
    constantly collect and share data, enabling unprecedented insights into how
    we live and work.
    """
    
    doc2 = """
    The Internet of Things (IoT) represents an extensive network comprising physical
    devices, vehicles, and appliances integrated with sensors and communication capabilities.
    These interconnected devices perpetually gather and transmit data, facilitating
    comprehensive insights regarding modern living and operational methodologies.
    """
    
    # Initialize integrated detector
    detector = IntegratedDetector(
        ai_detection_enabled=True,
        plagiarism_threshold=0.5
    )
    
    # Run comprehensive analysis
    results = detector.analyze(doc1, doc2)
    
    # Display results
    print("\n--- PLAGIARISM DETECTION ---")
    plag = results["plagiarism_results"]
    print(f"Similarity Score: {plag['similarity_score']:.2%}")
    print(f"Is Plagiarized: {plag['is_plagiarized']}")
    
    print("\n--- AI CONTENT DETECTION ---")
    ai = results["ai_results"]
    print(f"Document 1 - AI Score: {ai['document_1']['confidence']:.2%}")
    print(f"Document 1 - Is AI: {ai['document_1']['is_ai']}")
    print(f"Document 2 - AI Score: {ai['document_2']['confidence']:.2%}")
    print(f"Document 2 - Is AI: {ai['document_2']['is_ai']}")
    
    print("\n--- RISK ASSESSMENT ---")
    risk = results["risk_assessment"]
    print(f"Overall Risk Level: {risk['overall_risk_level']}")
    print(f"Risk Score: {risk['risk_score']:.2%}")
    print(f"Risk Factors: {', '.join(risk['risk_factors'])}")
    
    print("\n--- RECOMMENDATIONS ---")
    for i, rec in enumerate(results["recommendations"], 1):
        print(f"{i}. {rec}")


def example_generate_report():
    """Example 4: Generate formatted reports"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Generating Reports in Different Formats")
    print("=" * 70)
    
    doc1 = """
    Blockchain technology has revolutionized how we think about data security and
    distributed systems. The immutable nature of blockchain makes it ideal for
    applications requiring transparency and tamper-proof records.
    """
    
    doc2 = """
    Blockchain technology has fundamentally transformed perspectives on data security
    and distributed systems architecture. The immutable characteristics of blockchain
    technology render it exceptionally suitable for applications demanding transparency
    and protection against unauthorized modifications.
    """
    
    detector = IntegratedDetector()
    
    # Generate text report
    print("\n--- TEXT REPORT ---")
    text_report = detector.generate_report(doc1, doc2, output_format="text")
    print(text_report)
    
    # Generate JSON report
    print("\n--- JSON REPORT (first 500 chars) ---")
    json_report = detector.generate_report(doc1, doc2, output_format="json")
    print(json_report[:500] + "...")


def example_comparison():
    """Example 5: Compare AI likelihood between two texts"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Comparing AI Likelihood Between Two Texts")
    print("=" * 70)
    
    text1 = "This essay discusses the importance of renewable energy sources."
    text2 = """
    This comprehensive analysis elucidates the multifaceted significance of renewable
    energy modalities in contemporary environmental sustainability discourse.
    """
    
    detector = AIDetector()
    
    comparison = detector.compare_ai_content(text1, text2)
    
    print(f"Text 1 AI Score: {comparison['text1_ai_score']:.2%}")
    print(f"Text 2 AI Score: {comparison['text2_ai_score']:.2%}")
    print(f"Likely Both AI: {comparison['likely_both_ai']}")
    print(f"Likely Both Human: {comparison['likely_both_human']}")
    print(f"Mixed (One AI, One Human): {comparison['mixed']}")


def example_batch_analysis():
    """Example 6: Batch analysis of multiple documents"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Batch Analysis of Multiple Documents")
    print("=" * 70)
    
    documents = [
        {
            "id": "doc1",
            "text": "Climate change poses unprecedented challenges to global ecosystems."
        },
        {
            "id": "doc2",
            "text": "The phenomenon of climate modification represents an extraordinary challenge."
        },
        {
            "id": "doc3",
            "text": "Global warming is a serious issue that needs immediate attention."
        }
    ]
    
    detector = IntegratedDetector()
    ai_detector = AIDetector()
    
    print("\nAI Detection Results:")
    print("-" * 70)
    
    results = []
    for doc in documents:
        ai_result = ai_detector.detect_ai_content(doc["text"])
        results.append({
            "id": doc["id"],
            "ai_confidence": ai_result["confidence"],
            "is_ai": ai_result["is_ai"]
        })
        
        status = "AI-Generated" if ai_result["is_ai"] else "Human-Written"
        print(f"{doc['id']}: {status} (confidence: {ai_result['confidence']:.2%})")
    
    print("\nPlagiarism Cross-Check (sample pairs):")
    print("-" * 70)
    
    for i in range(len(documents) - 1):
        analysis = detector.analyze(
            documents[i]["text"],
            documents[i + 1]["text"],
            doc1_id=documents[i]["id"],
            doc2_id=documents[i + 1]["id"]
        )
        
        sim = analysis["plagiarism_results"]["similarity_score"]
        print(f"{documents[i]['id']} vs {documents[i + 1]['id']}: {sim:.2%} similarity")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRAPHPLAG: AI DETECTION AND INTEGRATED ANALYSIS EXAMPLES")
    print("=" * 70)
    
    try:
        # Run all examples
        example_ai_detection_only()
        example_plagiarism_detection()
        example_integrated_detection()
        example_generate_report()
        example_comparison()
        example_batch_analysis()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
