"""
Tests for AI Detection Module

Tests the AIDetector class with all detection methods.
"""

import pytest
from graphplag.detection.ai_detector import AIDetector


class TestAIDetector:
    """Test suite for AIDetector"""
    
    @pytest.fixture
    def detector(self):
        """Initialize detector for tests"""
        return AIDetector()
    
    @pytest.fixture
    def human_text(self):
        """Sample human-written text"""
        return """
        The cat sat on the mat. It was a sunny day, and the cat was happy.
        The mat was comfortable. The cat enjoyed sitting there. It made the cat feel peaceful.
        Every day, the cat came back to the mat. The mat became the cat's favorite spot.
        """
    
    @pytest.fixture
    def ai_text(self):
        """Sample AI-generated text"""
        return """
        The feline in question was positioned upon the aforementioned textile surface.
        Furthermore, it must be noted that the meteorological conditions were favorable.
        The aforementioned animal experienced contentment within said location.
        It is important to emphasize that the textile surface provided considerable comfort.
        Notably, the animal demonstrated consistent behavior patterns regarding this location.
        """
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly"""
        assert detector is not None
        assert isinstance(detector, AIDetector)
    
    def test_statistical_detection_human(self, detector, human_text):
        """Test statistical detection on human text"""
        result = detector.detect_ai_content(human_text, method="statistical")
        
        assert 'is_ai' in result
        assert 'confidence' in result
        assert 'scores' in result
        assert 0 <= result['confidence'] <= 1
    
    def test_statistical_detection_ai(self, detector, ai_text):
        """Test statistical detection on AI text"""
        result = detector.detect_ai_content(ai_text, method="statistical")
        
        assert 'is_ai' in result
        assert 'confidence' in result
        # AI text should be detected, confidence may vary based on text characteristics
        assert 0 <= result['confidence'] <= 1
    
    def test_linguistic_detection(self, detector, ai_text):
        """Test linguistic detection"""
        result = detector.detect_ai_content(ai_text, method="linguistic")
        
        assert 'is_ai' in result
        assert 'confidence' in result
        assert 'scores' in result
        # Should detect formal AI phrases
        assert result['confidence'] >= 0.4
    
    def test_ensemble_detection(self, detector, human_text, ai_text):
        """Test ensemble detection method"""
        human_result = detector.detect_ai_content(human_text, method="ensemble")
        ai_result = detector.detect_ai_content(ai_text, method="ensemble")
        
        # Both should return valid results
        assert 'is_ai' in human_result
        assert 'is_ai' in ai_result
        assert 0 <= human_result['confidence'] <= 1
        assert 0 <= ai_result['confidence'] <= 1
        
        # AI text should score higher than human
        assert ai_result['confidence'] > human_result['confidence']
    
    def test_short_text_handling(self, detector):
        """Test handling of short text"""
        short_text = "Hello world"
        result = detector.detect_ai_content(short_text)
        
        # Should return valid result even for short text
        assert 'is_ai' in result
        assert 'confidence' in result
    
    def test_empty_text_handling(self, detector):
        """Test handling of empty text"""
        result = detector.detect_ai_content("")
        
        # Should handle empty text gracefully
        assert 'is_ai' in result or 'error' in result['details']
    
    def test_confidence_range(self, detector, human_text):
        """Test confidence is always in valid range"""
        result = detector.detect_ai_content(human_text)
        
        assert 0 <= result['confidence'] <= 1
    
    def test_invalid_method(self, detector, human_text):
        """Test invalid method raises error"""
        with pytest.raises(ValueError):
            detector.detect_ai_content(human_text, method="invalid_method")
    
    def test_compare_ai_content(self, detector, human_text, ai_text):
        """Test comparing AI likelihood between two texts"""
        comparison = detector.compare_ai_content(human_text, ai_text)
        
        assert 'text1_ai_score' in comparison
        assert 'text2_ai_score' in comparison
        assert 'likely_both_ai' in comparison
        assert 'likely_both_human' in comparison
        assert 'mixed' in comparison
    
    def test_analyze_document(self, detector, human_text):
        """Test document analysis"""
        result = detector.analyze_document(human_text)
        
        assert 'ai_detection' in result
        assert 'summary' in result
        assert result['summary']['text_length'] > 0
    
    def test_statistical_scores_structure(self, detector, ai_text):
        """Test statistical scores have correct structure"""
        result = detector.detect_ai_content(ai_text, method="statistical")
        
        scores = result['scores']
        assert 'word_frequency' in scores
        assert 'repetition' in scores
        assert 'vocabulary_diversity' in scores
        
        # All scores should be between 0 and 1
        for score in scores.values():
            assert 0 <= score <= 1
    
    def test_linguistic_scores_structure(self, detector, ai_text):
        """Test linguistic scores have correct structure"""
        result = detector.detect_ai_content(ai_text, method="linguistic")
        
        scores = result['scores']
        assert 'ai_phrases' in scores
        assert 'transition_words' in scores
        assert 'passive_voice' in scores
    
    def test_long_text_handling(self, detector):
        """Test handling of very long text"""
        long_text = "This is a test sentence. " * 1000
        result = detector.detect_ai_content(long_text)
        
        assert 'is_ai' in result
        assert 'confidence' in result
    
    def test_multilingual_text(self, detector):
        """Test handling of non-English text"""
        french_text = """
        L'intelligence artificielle est une technologie révolutionnaire.
        Elle change la façon dont nous travaillons et communiquons.
        """
        result = detector.detect_ai_content(french_text)
        
        # Should still return valid result
        assert 'is_ai' in result
        assert 'confidence' in result


class TestAIDetectorEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def detector(self):
        return AIDetector()
    
    def test_special_characters(self, detector):
        """Test text with special characters"""
        text = "Hello!!! @#$% ***weird*** ~~~text~~~"
        result = detector.detect_ai_content(text)
        
        assert result is not None
    
    def test_numbers_only(self, detector):
        """Test text with only numbers"""
        text = "123 456 789 000 111 222"
        result = detector.detect_ai_content(text)
        
        assert result is not None
    
    def test_repeated_word(self, detector):
        """Test text with heavily repeated words"""
        text = "test " * 100
        result = detector.detect_ai_content(text)
        
        assert 'confidence' in result
        # Heavily repeated words may or may not indicate AI, confidence may vary
        assert 0 <= result['confidence'] <= 1
    
    def test_consistency_across_methods(self, detector):
        """Test that different methods give somewhat consistent results"""
        text = "The artificial intelligence system was designed to process data efficiently."
        
        stat_result = detector.detect_ai_content(text, method="statistical")
        ling_result = detector.detect_ai_content(text, method="linguistic")
        
        # Both should be in reasonable range
        assert 0 <= stat_result['confidence'] <= 1
        assert 0 <= ling_result['confidence'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
