"""
Similarity Detector Adapter

Wrapper around PlagiarismDetector to match expected import path.
"""

from graphplag.detection.detector import PlagiarismDetector


class SimilarityDetector:
    """
    Similarity detector adapter that wraps PlagiarismDetector.
    Provides similarity computation between documents.
    """
    
    def __init__(
        self,
        method: str = "kernel",
        threshold: float = 0.7,
        language: str = "en"
    ):
        """
        Initialize similarity detector.
        
        Args:
            method: Detection method ('kernel', 'gnn', or 'ensemble')
            threshold: Similarity threshold
            language: Language for parsing
        """
        self.method = method
        self.threshold = threshold
        self.detector = PlagiarismDetector(
            method=method,
            threshold=threshold,
            language=language
        )
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity score between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            report = self.detector.detect_plagiarism(text1, text2)
            return report.similarity_score
        except Exception as e:
            # Return 0 on error to prevent crashes
            print(f"Similarity computation error: {e}")
            return 0.0
    
    def detect(self, text1: str, text2: str) -> dict:
        """
        Detect plagiarism between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Detection results as dict
        """
        try:
            report = self.detector.detect_plagiarism(text1, text2)
            return {
                'similarity_score': report.similarity_score,
                'is_plagiarism': report.is_plagiarism,
                'threshold': report.threshold,
                'method': report.method,
                'num_matches': len(report.matches),
                'processing_time': report.processing_time
            }
        except Exception as e:
            return {
                'similarity_score': 0.0,
                'is_plagiarism': False,
                'error': str(e)
            }
    
    def __repr__(self) -> str:
        return f"SimilarityDetector(method='{self.method}', threshold={self.threshold})"
