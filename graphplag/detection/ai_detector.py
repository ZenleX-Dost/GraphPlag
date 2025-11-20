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
        
        # Weighted ensemble voting
        # Give higher weight to linguistic and statistical
        # Lower weight to neural (if it's not working well)
        if scores:
            weights = {
                'linguistic': 0.5,      # STRONGEST - explicit AI markers
                'statistical': 0.4,     # STRONG - patterns and structure
                'neural': 0.1          # WEAK - neural model performs poorly
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for method, score in scores.items():
                weight = weights.get(method, 0.33)
                weighted_sum += score * weight
                total_weight += weight
            
            ensemble_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        else:
            ensemble_score = 0.5
        
        # Lower threshold from 0.6 to 0.3 for better detection
        # Rationale: We want to catch real AI content, false positives are acceptable
        # Better to flag real AI content than to miss it
        is_ai = ensemble_score > 0.30
        
        return {
            "is_ai": is_ai,
            "confidence": float(ensemble_score),
            "scores": scores,
            "details": {
                "text_length": len(text),
                "method": "ensemble",
                "threshold": 0.45
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
        - Perplexity-like patterns
        - Token distribution
        - Sentence structure uniformity
        - Common n-grams
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
        
        ai_indicators = []
        
        # 1. Vocabulary richness (AI often shows lower diversity)
        word_freq = Counter(words)
        unique_words = len(word_freq)
        total_words = len(words)
        type_token_ratio = unique_words / max(total_words, 1)
        
        # AI tends toward 0.40-0.65 TTR, humans vary more
        # Penalize if in AI range
        if 0.40 <= type_token_ratio <= 0.65:
            ai_indicators.append(0.7)  # Likely AI range
        elif type_token_ratio > 0.70:
            ai_indicators.append(0.2)  # Too diverse for AI
        else:
            ai_indicators.append(0.5)  # Slightly repetitive
        
        # 2. Sentence length uniformity (AI is more uniform)
        sent_lengths = [len(s.split()) for s in sentences if s.split()]
        if len(sent_lengths) > 1:
            avg_len = np.mean(sent_lengths)
            std_len = np.std(sent_lengths)
            coeff_var = (std_len / avg_len) if avg_len > 0 else 0.5
            
            # AI has lower variation (0.25-0.45), humans higher (0.5-1.5)
            if coeff_var < 0.35:
                ai_indicators.append(0.75)  # Very uniform = likely AI
            elif coeff_var < 0.45:
                ai_indicators.append(0.6)   # Fairly uniform
            elif coeff_var > 0.9:
                ai_indicators.append(0.1)  # Varied = human
            else:
                ai_indicators.append(0.3)  # Neutral
        
        # 3. Common word patterns (AI uses more predictable patterns)
        most_common = word_freq.most_common(25)
        top_word_freq = sum(freq for _, freq in most_common) / total_words
        
        # If top 25 words make up >35% of text, might be AI
        if top_word_freq > 0.38:
            ai_indicators.append(0.7)
        elif top_word_freq < 0.20:
            ai_indicators.append(0.1)
        else:
            ai_indicators.append(0.4)
        
        # 4. Bigram and trigram repetition (AI repeats phrases more)
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-2)] if len(words) > 2 else []
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-3)] if len(words) > 3 else []
        
        if bigrams:
            bigram_freq = Counter(bigrams)
            # Count bigrams that appear 2+ times
            repeated_bigrams = sum(1 for f in bigram_freq.values() if f >= 2)
            bigram_score = min((repeated_bigrams * 1.5) / max(len(bigrams) / 5, 1), 1.0)
            ai_indicators.append(bigram_score * 0.7)
        
        # 5. Punctuation frequency (AI uses certain patterns more)
        punct_count = len(re.findall(r'[,;:\-—]', text))
        avg_punct_per_sent = punct_count / len(sentences) if sentences else 0
        
        # AI averages 1.2-2.8 per sentence, humans vary more widely
        if 1.0 <= avg_punct_per_sent <= 2.8:
            ai_indicators.append(0.6)
        elif avg_punct_per_sent > 3.0:
            ai_indicators.append(0.7)  # Heavy punctuation = AI
        else:
            ai_indicators.append(0.2)
        
        # 6. Parentheses and dash usage (AI uses more)
        paren_dash_count = len(re.findall(r'[()—\-]', text))
        if paren_dash_count >= len(sentences) * 0.8:
            ai_indicators.append(0.6)  # Heavy use suggests AI
        else:
            ai_indicators.append(0.2)
        
        # Calculate final score
        ai_score = np.mean(ai_indicators) if ai_indicators else 0.5
        
        return {
            "is_ai": ai_score > 0.5,
            "confidence": float(ai_score),
            "scores": {
                "type_token_ratio": float(type_token_ratio),
                "sentence_uniformity": float(1 - coeff_var) if len(sent_lengths) > 1 else 0.5,
                "phrase_repetition": float(ai_indicators[3]) if len(ai_indicators) > 3 else 0.0,
                "punctuation_patterns": float(ai_indicators[4]) if len(ai_indicators) > 4 else 0.0
            },
            "details": {
                "vocab_size": unique_words,
                "total_words": total_words,
                "unique_ratio": float(type_token_ratio),
                "avg_sentence_length": float(np.mean(sent_lengths)) if sent_lengths else 0,
                "method": "statistical"
            }
        }
    
    def _linguistic_detect(self, text: str) -> Dict:
        """
        Detect AI patterns using linguistic markers:
        - Formal vocabulary patterns
        - Phrase structures
        - Transition words
        - Academic language markers
        """
        words = text.lower().split()
        
        if not words:
            return {
                "is_ai": None,
                "confidence": 0.5,
                "scores": {},
                "details": {"error": "Insufficient text"}
            }
        
        text_lower = text.lower()
        ai_scores = []
        
        # 1. STRONG AI INDICATORS - Direct model phrases
        strong_ai_phrases = [
            "as an ai", "as a language model", "as an llm",
            "i'm unable", "i cannot", "i don't have", "i cannot access",
            "it's not possible for me", "i lack the ability",
            "my training data", "language model",
            "i appreciate your", "i understand you", "i can provide",
            "i can help", "i can assist", "as an artificial intelligence",
        ]
        
        strong_count = sum(text_lower.count(phrase) for phrase in strong_ai_phrases)
        if strong_count > 0:
            ai_scores.append(min(strong_count * 0.8, 1.0))  # Directly indicates AI
        else:
            ai_scores.append(0.0)
        
        # 2. FORMAL TRANSITION WORDS (AI overuses these significantly)
        formal_transitions = [
            "furthermore", "moreover", "additionally", "consequently",
            "ultimately", "notably", "specifically", "therefore",
            "hence", "thus", "accordingly", "as such",
            "in conclusion", "to summarize", "in summary",
            "it must be noted", "it is worth noting", "it is important",
            "the fact that", "the notion that", "the concept of",
            "in light of", "by contrast", "in contrast", "on the other hand",
            "it is clear", "it is evident", "it is important to note"
        ]
        
        transition_count = sum(text_lower.count(t) for t in formal_transitions)
        # Heavy use of transitions is strong AI indicator
        # Boost the sensitivity: lower denominator = higher score
        transition_score = min(transition_count / max(len(words) / 40, 1), 1.0)  # Was /50
        ai_scores.append(transition_score * 0.95)  # Boosted weight
        
        # 3. PASSIVE VOICE (AI uses significantly more)
        passive_patterns = [
            r'\bwas\s+\w{4,}ed\b',  # was + past participle (at least 4 letters)
            r'\bbeen\s+\w{4,}ed\b', 
            r'\bis\s+\w{4,}ed\b',
            r'\bbeing\s+\w{4,}ed\b',
        ]
        
        passive_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE)) 
            for pattern in passive_patterns
        )
        passive_score = min(passive_count / max(len(words) / 15, 1), 1.0)
        ai_scores.append(passive_score * 0.7)
        
        # 4. FORMAL VOCABULARY PATTERNS
        formal_words = [
            "utilize", "facilitate", "implement", "optimize", "paradigm",
            "framework", "methodology", "hypothesis", "phenomenon",
            "phenomenon", "implications", "significant", "substantial",
            "comprehensive", "systematic", "analytical", "intricate",
            "endeavor", "endeavors", "purported", "endeavor"
        ]
        
        formal_count = sum(text_lower.count(w) for w in formal_words)
        formal_score = min(formal_count / max(len(words) / 40, 1), 1.0)
        ai_scores.append(formal_score * 0.6)
        
        # 5. LACK OF CONTRACTIONS (AI avoids them)
        contractions = ["don't", "can't", "won't", "isn't", "wasn't", "weren't", 
                       "haven't", "hasn't", "couldn't", "shouldn't", "wouldn't",
                       "i'm", "it's", "that's", "there's", "they're", "we're",
                       "you're", "let's", "here's", "what's", "who's"]
        
        contraction_count = sum(text_lower.count(c) for c in contractions)
        # Few contractions = AI (humans use many)
        if len(words) > 20:
            contraction_ratio = contraction_count / (len(words) / 100)
            if contraction_ratio < 0.5:  # Less than 0.5 contractions per 100 words
                ai_scores.append(0.6)
            elif contraction_ratio > 2.0:
                ai_scores.append(0.1)  # Lots of contractions = human
            else:
                ai_scores.append(0.3)
        
        # 6. REPETITION OF KEY PHRASES (AI repeats patterns)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        phrase_patterns = [
            r"as (a|an) \w+",
            r"this \w+ (is|shows|indicates|demonstrates)",
            r"(in|during|throughout) \w+",
        ]
        
        repeated_phrases = 0
        for pattern in phrase_patterns:
            matches = re.findall(pattern, text_lower)
            if len(matches) > 2:  # Pattern repeats multiple times
                repeated_phrases += min(len(matches), 5)
        
        phrase_score = min(repeated_phrases / 8.0, 1.0)
        ai_scores.append(phrase_score * 0.5)
        
        # Calculate final score
        ai_score = np.mean(ai_scores) if ai_scores else 0.5
        
        # Boost confidence if strong AI markers found
        if strong_count > 0:
            ai_score = min(ai_score + 0.3, 1.0)
        
        return {
            "is_ai": ai_score > 0.45,  # Lower threshold since this is specific
            "confidence": float(ai_score),
            "scores": {
                "ai_phrases": float(ai_scores[0]) if len(ai_scores) > 0 else 0.0,
                "transition_words": float(ai_scores[1]) if len(ai_scores) > 1 else 0.0,
                "passive_voice": float(ai_scores[2]) if len(ai_scores) > 2 else 0.0,
                "formal_vocabulary": float(ai_scores[3]) if len(ai_scores) > 3 else 0.0,
            },
            "details": {
                "ai_phrase_count": strong_count,
                "transition_count": transition_count,
                "passive_voice_count": passive_count,
                "formal_word_count": formal_count,
                "contraction_count": contraction_count,
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
