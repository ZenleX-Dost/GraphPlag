"""
Unit tests for PlagiarismDetector module.
"""

import pytest
import numpy as np

from graphplag.detection.detector import PlagiarismDetector
from graphplag.core.models import PlagiarismReport


@pytest.fixture
def detector():
    """Create a PlagiarismDetector instance."""
    return PlagiarismDetector(method='kernel', threshold=0.7, language='en')


def test_detector_initialization(detector):
    """Test detector initializes correctly."""
    assert detector.method == 'kernel'
    assert detector.threshold == 0.7
    assert detector.parser is not None
    assert detector.graph_builder is not None


def test_detect_plagiarism_similar(detector):
    """Test plagiarism detection for similar documents."""
    doc1 = "Machine learning is a field of artificial intelligence."
    doc2 = "Machine learning is an artificial intelligence field."
    
    report = detector.detect_plagiarism(doc1, doc2)
    
    assert isinstance(report, PlagiarismReport)
    assert 0.0 <= report.similarity_score <= 1.0
    assert isinstance(report.is_plagiarism, bool)
    assert report.processing_time > 0


def test_detect_plagiarism_dissimilar(detector):
    """Test plagiarism detection for dissimilar documents."""
    doc1 = "The weather is sunny today."
    doc2 = "Quantum mechanics is fascinating."
    
    report = detector.detect_plagiarism(doc1, doc2)
    
    assert isinstance(report, PlagiarismReport)
    assert report.similarity_score < 0.9  # Should not be very similar


def test_detect_plagiarism_identical(detector):
    """Test plagiarism detection for identical documents."""
    text = "This is a test document for plagiarism detection."
    
    report = detector.detect_plagiarism(text, text)
    
    assert report.similarity_score > 0.9  # Should be very high
    assert report.is_plagiarism == True


def test_batch_compare(detector):
    """Test batch document comparison."""
    documents = [
        "First document about AI.",
        "Second document about AI.",
        "Third document about cooking."
    ]
    
    similarity_matrix = detector.batch_compare(documents)
    
    assert isinstance(similarity_matrix, np.ndarray)
    assert similarity_matrix.shape == (3, 3)
    
    # Diagonal should be 1.0
    assert np.allclose(np.diag(similarity_matrix), 1.0, atol=0.1)
    
    # Matrix should be symmetric
    assert np.allclose(similarity_matrix, similarity_matrix.T, atol=0.01)


def test_identify_suspicious_pairs(detector):
    """Test identifying suspicious document pairs."""
    documents = [
        "Machine learning is a field of AI.",
        "Machine learning is an AI field.",
        "The weather is sunny today.",
        "Quantum computing uses qubits."
    ]
    
    suspicious_pairs = detector.identify_suspicious_pairs(documents, threshold=0.5)
    
    assert isinstance(suspicious_pairs, list)
    
    # Each pair should be a tuple of (idx1, idx2, score)
    for pair in suspicious_pairs:
        assert len(pair) == 3
        assert isinstance(pair[0], int)
        assert isinstance(pair[1], int)
        assert isinstance(pair[2], float)
        assert 0.0 <= pair[2] <= 1.0


def test_report_structure(detector):
    """Test plagiarism report structure."""
    doc1 = "This is the first document."
    doc2 = "This is the second document."
    
    report = detector.detect_plagiarism(doc1, doc2, doc1_id="d1", doc2_id="d2")
    
    assert report.document1.doc_id == "d1"
    assert report.document2.doc_id == "d2"
    assert hasattr(report, 'similarity_score')
    assert hasattr(report, 'is_plagiarism')
    assert hasattr(report, 'threshold')
    assert hasattr(report, 'method')
    assert hasattr(report, 'processing_time')


def test_report_summary(detector):
    """Test report summary generation."""
    doc1 = "Test document one."
    doc2 = "Test document two."
    
    report = detector.detect_plagiarism(doc1, doc2)
    summary = report.summary()
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "Similarity" in summary or "similarity" in summary.lower()


def test_threshold_boundary(detector):
    """Test threshold boundary behavior."""
    # Create detector with specific threshold
    detector_low = PlagiarismDetector(method='kernel', threshold=0.3)
    detector_high = PlagiarismDetector(method='kernel', threshold=0.9)
    
    doc1 = "This is a test."
    doc2 = "This is another test."
    
    report_low = detector_low.detect_plagiarism(doc1, doc2)
    report_high = detector_high.detect_plagiarism(doc1, doc2)
    
    # Low threshold is more likely to detect plagiarism
    # High threshold is less likely to detect plagiarism
    assert isinstance(report_low.is_plagiarism, bool)
    assert isinstance(report_high.is_plagiarism, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
