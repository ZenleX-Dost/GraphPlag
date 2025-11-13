"""
AI-Generated Text Detection Module

Detects whether text was written by AI (ChatGPT, Claude, etc.) or humans.
Uses multiple detection methods:
1. Perplexity-based detection (statistical patterns)
2. Entropy analysis (repetition and diversity)
3. Neural-based detection (fine-tuned classifiers)
4. Linguistic markers (vocabulary, sentence structure)
"""

from typing import Optional, Dict, List, Tuple
import numpy as np
from collections import Counter
import math
import re

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from graphplag.core.models import Document, SimilarityScore


class AIDetector:
    """
    Detect AI-generated text using multiple methods.
    """
    
    def __init__(self, model_name: str = "openai-community/roberta-base-openai-detector"):
        """
        Initialize AI detector.
        
        Args:
            model_name: HuggingFace model for AI detection
        """
        self.model_name = model_name
        self.classifier = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                print(f"Warning: Could not load AI detection model: {e}")
                print("Falling back to statistical methods only")
    
    def detect_ai_content(
        self,
        text: str,
        method: str = "ensemble"
    ) -> Dict:
        """
        Detect if text was written by AI.
        
        Args:
            text: Input text to analyze
            method: Detection method ('ensemble', 'neural', 'statistical', 'linguistic')
            
        Returns:
            Dictionary with:
                - is_ai: Boolean (True if likely AI-generated)
                - confidence: Float 0-1 (confidence score)
                - scores: Dict with individual method scores
                - details: Dict with analysis details
        """
        if method == "ensemble":
            return self._ensemble_detect(text)
        elif method == "neural":
            return self._neural_detect(text)
        elif method == "statistical":
            return self._statistical_detect(text)
        elif method == "linguistic":
            return self._linguistic_detect(text)
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def _ensemble_detect(self, text: str) -> Dict:
        """
        Use multiple methods for robust AI detection.
        """
        scores = {}
        
        # Statistical analysis
        if len(text) > 50:  # Need sufficient text
            stat_result = self._statistical_detect(text)
            scores['statistical'] = stat_result['confidence']
        
        # Linguistic analysis
        ling_result = self._linguistic_detect(text)
        scores['linguistic'] = ling_result['confidence']
        
        # Neural detection if available
        if self.classifier:
            neural_result = self._neural_detect(text)
            scores['neural'] = neural_result['confidence']
        
        # Ensemble voting
        if scores:
            ensemble_score = np.mean(list(scores.values()))
        else:
            ensemble_score = 0.5
        
        # Threshold: >0.6 is likely AI
        is_ai = ensemble_score > 0.6
        
        return {
            "is_ai": is_ai,
            "confidence": float(ensemble_score),
            "scores": scores,
            "details": {
                "text_length": len(text),
                "method": "ensemble"
            }
        }
    
    def _neural_detect(self, text: str) -> Dict:
        """
        Use fine-tuned neural model for AI detection.
        """
        if not self.classifier:
            return {
                "is_ai": None,
                "confidence": 0.5,
                "scores": {},
                "details": {"error": "Neural model not available"}
            }
        
        try:
            # Truncate if too long (models have token limits)
            if len(text) > 512:
                text = text[:512]
            
            result = self.classifier(text)
            
            # Results: [{'label': 'human'/'fake', 'score': float}]
            label = result[0]['label']
            score = result[0]['score']
            
            # Score is confidence in the predicted label
            # If label is 'fake' (AI), return score as-is
            # If label is 'human', return 1-score
            if label == 'fake' or label == '1' or label == 'AI':
                ai_confidence = score
            else:
                ai_confidence = 1 - score
            
            return {
                "is_ai": ai_confidence > 0.5,
                "confidence": float(ai_confidence),
                "scores": {label: score},
                "details": {
                    "model": self.model_name,
                    "predicted_label": label,
                    "method": "neural"
                }
            }
        except Exception as e:
            return {
                "is_ai": None,
                "confidence": 0.5,
                "scores": {},
                "details": {"error": str(e), "method": "neural"}
            }
    
    def _statistical_detect(self, text: str) -> Dict:
        """
        Detect AI patterns using statistical analysis:
        - Word frequency patterns
        - Sentence length variance
        - Repetition rate
        """
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not words or not sentences:
            return {
                "is_ai": None,
                "confidence": 0.5,
                "scores": {},
                "details": {"error": "Insufficient text"}
            }
        
        # 1. Word frequency analysis (AI tends to have more uniform distribution)
        word_freq = Counter(words)
        freq_values = list(word_freq.values())
        avg_freq = np.mean(freq_values)
        std_freq = np.std(freq_values)
        
        # AI text has lower variance in word frequency
        # Normalize to 0-1 scale
        freq_score = min(std_freq / 5.0, 1.0)  # Empirical threshold
        
        # 2. Sentence length variance (AI tends to have more uniform sentence lengths)
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = np.mean(sent_lengths)
        std_sent_len = np.std(sent_lengths)
        
        # AI has lower variance
        length_score = min(std_sent_len / 8.0, 1.0)  # Empirical threshold
        
        # 3. Repetition rate (AI tends to repeat phrases)
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        bigram_freq = Counter(bigrams)
        repetition_score = len([f for f in bigram_freq.values() if f > 2]) / max(len(bigrams), 1)
        repetition_score = min(repetition_score, 1.0)
        
        # 4. Vocabulary diversity (AI has lower diversity)
        vocab_diversity = len(word_freq) / len(words)
        diversity_score = max(1 - vocab_diversity, 0)  # AI has lower diversity
        
        # Combine scores
        ai_score = (freq_score + repetition_score + diversity_score) / 3.0
        
        return {
            "is_ai": ai_score > 0.6,
            "confidence": float(ai_score),
            "scores": {
                "word_frequency": float(freq_score),
                "repetition": float(repetition_score),
                "vocabulary_diversity": float(diversity_score)
            },
            "details": {
                "vocab_size": len(word_freq),
                "avg_sentence_length": float(avg_sent_len),
                "sentence_length_variance": float(std_sent_len),
                "method": "statistical"
            }
        }
    
    def _linguistic_detect(self, text: str) -> Dict:
        """
        Detect AI patterns using linguistic markers:
        - Phrase patterns
        - Punctuation usage
        - Transition words
        - Active vs passive voice
        """
        words = text.lower().split()
        
        if not words:
            return {
                "is_ai": None,
                "confidence": 0.5,
                "scores": {},
                "details": {"error": "Insufficient text"}
            }
        
        # 1. Check for common AI phrases
        ai_phrases = [
            "as an ai", "as a language model", "i appreciate", "i understand",
            "furthermore", "in conclusion", "it is important to note",
            "let me explain", "in summary", "to put it simply"
        ]
        
        text_lower = text.lower()
        ai_phrase_count = sum(text_lower.count(phrase) for phrase in ai_phrases)
        phrase_score = min(ai_phrase_count / 3.0, 1.0)  # Normalize
        
        # 2. Punctuation patterns (AI uses more varied punctuation)
        punctuation_count = len(re.findall(r'[,;:\-()]', text))
        punctuation_score = min(punctuation_count / (len(words) / 2), 1.0)
        
        # 3. Transition words (AI overuses them)
        transitions = [
            "however", "therefore", "moreover", "furthermore", "additionally",
            "consequently", "ultimately", "notably", "specifically"
        ]
        transition_count = sum(text_lower.count(t) for t in transitions)
        transition_score = min(transition_count / 2.0, 1.0)
        
        # 4. Passive voice (AI tends to use more passive voice)
        passive_patterns = [
            r'\bwas\s+\w+ed\b', r'\bbeen\s+\w+ed\b', r'\bis\s+\w+ed\b'
        ]
        passive_count = sum(
            len(re.findall(pattern, text)) for pattern in passive_patterns
        )
        passive_score = min(passive_count / max(len(words) / 10, 1), 1.0)
        
        # Combine scores
        ai_score = (phrase_score + transition_score + passive_score) / 3.0
        
        return {
            "is_ai": ai_score > 0.5,
            "confidence": float(ai_score),
            "scores": {
                "ai_phrases": float(phrase_score),
                "transition_words": float(transition_score),
                "passive_voice": float(passive_score)
            },
            "details": {
                "ai_phrase_count": ai_phrase_count,
                "transition_count": transition_count,
                "passive_voice_count": passive_count,
                "method": "linguistic"
            }
        }
    
    def analyze_document(self, document: str) -> Dict:
        """
        Analyze document for both plagiarism indicators and AI-generated content.
        
        Args:
            document: Document text
            
        Returns:
            Comprehensive analysis report
        """
        ai_result = self.detect_ai_content(document, method="ensemble")
        
        return {
            "ai_detection": ai_result,
            "summary": {
                "is_ai_generated": ai_result["is_ai"],
                "ai_confidence": ai_result["confidence"],
                "text_length": len(document),
                "word_count": len(document.split())
            }
        }
    
    def compare_ai_content(
        self,
        text1: str,
        text2: str
    ) -> Dict:
        """
        Compare AI likelihood between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Comparison results
        """
        result1 = self.detect_ai_content(text1, method="ensemble")
        result2 = self.detect_ai_content(text2, method="ensemble")
        
        return {
            "text1_ai_score": result1["confidence"],
            "text2_ai_score": result2["confidence"],
            "likely_both_ai": result1["is_ai"] and result2["is_ai"],
            "likely_both_human": not result1["is_ai"] and not result2["is_ai"],
            "mixed": result1["is_ai"] != result2["is_ai"],
            "details": {
                "text1": result1,
                "text2": result2
            }
        }


class CombinedDetector:
    """
    Combined plagiarism + AI detection system.
    """
    
    def __init__(
        self,
        plagiarism_detector,
        ai_detector: Optional[AIDetector] = None
    ):
        """
        Initialize combined detector.
        
        Args:
            plagiarism_detector: PlagiarismDetector instance
            ai_detector: AIDetector instance (created if None)
        """
        self.plagiarism_detector = plagiarism_detector
        self.ai_detector = ai_detector or AIDetector()
    
    def comprehensive_analysis(
        self,
        doc1: str,
        doc2: str,
        doc1_id: Optional[str] = None,
        doc2_id: Optional[str] = None
    ) -> Dict:
        """
        Perform comprehensive analysis: plagiarism + AI detection.
        
        Args:
            doc1: First document
            doc2: Second document
            doc1_id: Optional ID for doc1
            doc2_id: Optional ID for doc2
            
        Returns:
            Combined analysis report with:
            - Plagiarism scores
            - AI detection results
            - Risk assessment
        """
        # Detect plagiarism
        plagiarism_report = self.plagiarism_detector.detect_plagiarism(
            doc1, doc2, doc1_id=doc1_id, doc2_id=doc2_id
        )
        
        # Detect AI content in both documents
        ai_analysis_doc1 = self.ai_detector.detect_ai_content(doc1, method="ensemble")
        ai_analysis_doc2 = self.ai_detector.detect_ai_content(doc2, method="ensemble")
        
        # Risk assessment
        plagiarism_score = plagiarism_report.similarity_score
        ai_risk_doc1 = ai_analysis_doc1["confidence"]
        ai_risk_doc2 = ai_analysis_doc2["confidence"]
        
        # Combined risk score
        combined_risk = {
            "plagiarism_risk": plagiarism_score,
            "ai_generation_risk_doc1": ai_risk_doc1,
            "ai_generation_risk_doc2": ai_risk_doc2,
            "overall_integrity_score": 1.0 - (
                (plagiarism_score + ai_risk_doc1 + ai_risk_doc2) / 3.0
            ),
            "flags": []
        }
        
        # Generate flags
        if plagiarism_score > 0.7:
            combined_risk["flags"].append("HIGH_PLAGIARISM_DETECTED")
        
        if ai_risk_doc1 > 0.7 and ai_risk_doc2 > 0.7:
            combined_risk["flags"].append("BOTH_DOCUMENTS_LIKELY_AI_GENERATED")
        elif ai_risk_doc1 > 0.7:
            combined_risk["flags"].append("DOCUMENT_1_LIKELY_AI_GENERATED")
        elif ai_risk_doc2 > 0.7:
            combined_risk["flags"].append("DOCUMENT_2_LIKELY_AI_GENERATED")
        
        if plagiarism_score > 0.5 and (ai_risk_doc1 > 0.6 or ai_risk_doc2 > 0.6):
            combined_risk["flags"].append("AI_ASSISTED_PLAGIARISM_POSSIBLE")
        
        return {
            "plagiarism_analysis": plagiarism_report,
            "ai_analysis": {
                "document_1": ai_analysis_doc1,
                "document_2": ai_analysis_doc2
            },
            "risk_assessment": combined_risk
        }
