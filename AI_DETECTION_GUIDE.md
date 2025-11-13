# GraphPlag AI Detection - Quick Start Guide

## What's New? 🤖

GraphPlag now includes **AI-Generated Content Detection** alongside plagiarism detection. You can now check for both:
- ✅ **Plagiarism** - Copied content from other sources
- ✅ **AI-Generated Text** - Content written by ChatGPT, Claude, Gemini, etc.

## Quick Start

### 1. Detect If Text Is AI-Generated

```python
from graphplag.detection.ai_detector import AIDetector

detector = AIDetector()
result = detector.detect_ai_content("Your text here")

print(f"Is AI-generated: {result['is_ai']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 2. Check Both Plagiarism AND AI in One Go

```python
from graphplag.detection.integrated_detector import IntegratedDetector

detector = IntegratedDetector()
results = detector.analyze(document1, document2)

# Plagiarism score
print(f"Similarity: {results['plagiarism_results']['similarity_score']:.2%}")

# AI scores
print(f"Doc1 is AI: {results['ai_results']['document_1']['is_ai']}")
print(f"Doc2 is AI: {results['ai_results']['document_2']['is_ai']}")

# Risk assessment
print(f"Risk Level: {results['risk_assessment']['overall_risk_level']}")
print(f"Recommendations: {results['recommendations']}")
```

### 3. Using the Web Interface

```powershell
.\run.bat
```

Choose option **[4] Enhanced Web Interface** - it now shows:
- 📊 Plagiarism analysis
- 🤖 AI detection results
- ⚠️ Risk assessment
- 💡 Recommendations

## Detection Methods

| Method | Speed | Accuracy | Notes |
|--------|-------|----------|-------|
| **Neural** | Medium | High | Uses fine-tuned model |
| **Statistical** | Fast | Medium | Analyzes word patterns |
| **Linguistic** | Fast | Medium | Checks for AI phrases |
| **Ensemble** | Medium | Very High | Combines all methods ⭐ |

**Use Ensemble (default)** for best results!

## Understanding AI Detection Confidence

- **0-30%** - Likely human-written
- **30-60%** - Ambiguous (mixed or complex writing)
- **60-100%** - Likely AI-generated

**Note:** Even human-written formal text may score 40-50%. Use full context for interpretation.

## What Does It Detect?

✅ **Detects:**
- ChatGPT, Claude, Gemini outputs
- Paraphrased AI content
- AI-assisted writing (partial AI)
- Formal/technical AI writing

❌ **Cannot Detect:**
- Highly edited AI content
- Mixed human+AI (requires context)
- Creative writing that mimics AI patterns
- Adversarially modified AI text

## Risk Assessment Explained

When analyzing two documents, you get a **combined risk score**:

```
Risk = (Plagiarism Score × 0.6) + (AI Detection Score × 0.4)
```

### Risk Levels:

| Level | What It Means | Action |
|-------|-------------|--------|
| **CRITICAL** | Severe integrity issues | Investigate immediately |
| **HIGH** | Significant concerns | Review with author |
| **MODERATE** | Some concerns | Manual verification |
| **LOW** | Minor concerns | Likely acceptable |
| **MINIMAL** | No concerns | Approve |

## Examples

### Example 1: Pure Plagiarism

```
Document 1: "The quick brown fox jumps over the lazy dog"
Document 2: "The quick brown fox jumps over the lazy dog"

Result:
- Plagiarism: 100% ❌
- AI Doc1: 10% ✅
- AI Doc2: 10% ✅
- Risk: CRITICAL (Exact copy)
```

### Example 2: AI-Generated Content

```
Document 1: "Artificial intelligence has undergone significant evolution..."
Document 2: (reference text)

Result:
- Plagiarism: 5% ✅
- AI Detection: 85% ❌
- Risk: HIGH (AI-generated)
```

### Example 3: AI + Plagiarism (Worst Case)

```
Document 1: (AI-generated text)
Document 2: (AI-generated text with minor changes)

Result:
- Plagiarism: 75% ❌
- AI Doc1: 90% ❌
- AI Doc2: 85% ❌
- Risk: CRITICAL
```

### Example 4: Clean Document

```
Document 1: (Original human writing)
Document 2: (Different original writing)

Result:
- Plagiarism: 10% ✅
- AI Doc1: 20% ✅
- AI Doc2: 25% ✅
- Risk: MINIMAL
```

## Using AI Detection in Your Code

### Method 1: Simple Detection

```python
from graphplag.detection.ai_detector import AIDetector

detector = AIDetector()

# Quick check
is_ai = detector.detect_ai_content(text)['is_ai']
if is_ai:
    print("Warning: Text appears to be AI-generated")
```

### Method 2: Detailed Analysis

```python
result = detector.detect_ai_content(text, method="ensemble")

print("AI Detection Results:")
print(f"  Is AI: {result['is_ai']}")
print(f"  Confidence: {result['confidence']:.2%}")
print(f"  Individual Scores:")
for method, score in result['scores'].items():
    print(f"    - {method}: {score:.2%}")
```

### Method 3: Batch Processing

```python
from graphplag.detection.ai_detector import AIDetector

detector = AIDetector()
documents = ["text1", "text2", "text3"]

results = []
for doc in documents:
    analysis = detector.detect_ai_content(doc)
    results.append({
        "text": doc[:50],
        "is_ai": analysis['is_ai'],
        "confidence": analysis['confidence']
    })

# Show summary
for r in results:
    status = "⚠️ AI" if r['is_ai'] else "✅ Human"
    print(f"{status}: {r['text']}... ({r['confidence']:.0%})")
```

## Frequently Asked Questions

**Q: How accurate is AI detection?**
A: Using the ensemble method, accuracy is typically 80-85% on clearly AI-generated text. Real-world performance depends on the sophistication of the writing.

**Q: Can it detect GPT-4 vs GPT-3 vs Claude?**
A: No, it detects "AI-like" patterns rather than specific models. Different AI models produce different patterns, making source identification difficult.

**Q: Will it flag formal human writing as AI?**
A: Possibly. Highly formal, technical, or academic human writing may score 40-50%. Always consider the context.

**Q: What about translated text?**
A: Translations often score higher on AI detection (40-60%) due to structural patterns. This is expected behavior.

**Q: Can I disable AI detection?**
A: Yes, use plagiarism detection only:
```python
from graphplag.detection.detector import PlagiarismDetector
detector = PlagiarismDetector()
```

**Q: How does it compare to other AI detectors?**
A: GraphPlag uses academic-grade graph kernels for plagiarism + multiple AI detection methods. It's designed for educational integrity checking, not absolute truth.

## Integration with Your System

### Educational Institutions

```python
# Check student submissions
for student_file in submitted_files:
    result = detector.analyze(student_file, rubric_text)
    
    if result['risk_assessment']['overall_risk_level'] in ['CRITICAL', 'HIGH']:
        flag_for_review(student_file)
```

### Content Verification

```python
# Verify user-generated content
if detector.detect_ai_content(user_text)['confidence'] > 0.7:
    require_human_verification(user_text)
```

### Academic Integrity Checks

```python
# Pre-submission screening
integrity_report = detector.analyze(submission, assignment_instructions)
if integrity_report['risk_assessment']['overall_risk_level'] != 'MINIMAL':
    send_alert_to_instructor(integrity_report)
```

## Advanced: Fine-Tuning Detection

### Adjust AI Detection Confidence Threshold

```python
detector = AIDetector()
result = detector.detect_ai_content(text)

# Custom threshold
THRESHOLD = 0.65  # 65% confidence
if result['confidence'] > THRESHOLD:
    print("Likely AI-generated")
```

### Focus on Specific Detection Methods

```python
# Use only statistical analysis (fast)
stat_result = detector.detect_ai_content(text, method="statistical")

# Use only neural detection (accurate)
neural_result = detector.detect_ai_content(text, method="neural")

# Use ensemble for best balance
ensemble_result = detector.detect_ai_content(text, method="ensemble")
```

## Troubleshooting

**Q: Neural model fails to load**
A: The AI detection model requires ~500MB download. Check internet connection and disk space.

**Q: Getting "Cannot read properties of undefined" error**
A: Ensure text is not empty. Minimum 50 characters recommended for accurate detection.

**Q: Ensemble detection is slow**
A: Use `method="statistical"` for faster processing without neural model.

**Q: Getting NaN (Not a Number) scores**
A: Input text may be too short or malformed. Try with longer text (>100 characters).

## Next Steps

1. **Run Examples**: `python examples/ai_detection_examples.py`
2. **Try Web Interface**: `.\run.bat` → Choose option [4]
3. **Read Full Docs**: See DOCUMENTATION.md for comprehensive guide
4. **Integrate**: Use in your application with `IntegratedDetector`

---

**Need Help?** See DOCUMENTATION.md or run examples to learn more!
