# GraphPlag AI Detection - Quick Reference

## One-Line Detection

```python
from graphplag.detection.ai_detector import AIDetector
is_ai = AIDetector().detect_ai_content(text)['is_ai']
```

## Common Use Cases

### 1. Check if document is AI-generated

```python
from graphplag.detection.ai_detector import AIDetector

detector = AIDetector()
result = detector.detect_ai_content(document_text)

if result['is_ai']:
    print(f"⚠️ AI-generated (confidence: {result['confidence']:.0%})")
```

### 2. Check plagiarism + AI simultaneously

```python
from graphplag.detection.integrated_detector import IntegratedDetector

detector = IntegratedDetector()
results = detector.analyze(doc1, doc2)

print(f"Plagiarism: {results['plagiarism_results']['similarity_score']:.0%}")
print(f"AI Risk: {results['risk_assessment']['overall_risk_level']}")
```

### 3. Get recommendations automatically

```python
for rec in results['recommendations']:
    print(f"→ {rec}")

# Output examples:
# → ACCEPT: Low integrity risk detected
# → REVIEW: Significant integrity concerns require manual review
# → ESCALATE: Refer to institutional review board
```

### 4. Batch check multiple documents

```python
from graphplag.detection.ai_detector import AIDetector

detector = AIDetector()
for document in documents:
    result = detector.detect_ai_content(document)
    if result['is_ai']:
        print(f"📌 {document[:30]}... - AI-generated")
```

### 5. Generate a detailed report

```python
detector = IntegratedDetector()
report = detector.generate_report(doc1, doc2, output_format="text")
print(report)

# Or save as HTML
html = detector.generate_report(doc1, doc2, output_format="html")
with open("report.html", "w") as f:
    f.write(html)
```

## Detection Methods Comparison

| Need | Use Method | Command |
|------|-----------|---------|
| **Fast screening** | Statistical | `method="statistical"` |
| **Accurate detection** | Neural | `method="neural"` |
| **Best results** | Ensemble | `method="ensemble"` (default) |
| **Explainable** | Linguistic | `method="linguistic"` |

## Understanding Results

```python
result = detector.detect_ai_content(text)

# Main result
result['is_ai']              # True/False
result['confidence']         # 0.0-1.0 (0-100%)

# Detailed scores
result['scores']             # {'statistical': 0.65, ...}

# Extra info
result['details']            # Method-specific details
```

## Risk Levels at a Glance

```
CRITICAL (>80%)  → 🔴 REJECT
HIGH (60-80%)    → 🟠 REVIEW  
MODERATE (40-60%)→ 🟡 VERIFY
LOW (20-40%)     → 🟢 OK
MINIMAL (<20%)   → ✅ APPROVE
```

## Confidence Thresholds

```python
# Conservative (strict)
if result['confidence'] > 0.75:
    print("Very likely AI-generated")

# Standard (balanced)
if result['confidence'] > 0.60:
    print("Likely AI-generated")

# Lenient (permissive)
if result['confidence'] > 0.50:
    print("Possibly AI-generated")
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Slow neural detection | Use `method="statistical"` |
| Model download fails | Check internet, ~500MB disk space |
| Getting NaN scores | Use text >100 characters |
| False positives | Increase confidence threshold |
| Model not found | Run `python -c "from transformers import pipeline; pipeline('text-classification', model='openai-community/roberta-base-openai-detector')"` |

## Files & Documentation

📚 **Documentation Files**:
- `DOCUMENTATION.md` - Complete guide (80+ pages)
- `AI_DETECTION_GUIDE.md` - AI-specific guide
- `AI_DETECTION_IMPLEMENTATION.md` - Technical details
- `README.md` - Project overview

💻 **Code Files**:
- `graphplag/detection/ai_detector.py` - Core AI detection
- `graphplag/detection/integrated_detector.py` - Combined analysis
- `examples/ai_detection_examples.py` - 6 working examples

## Running Web Interface with AI Detection

```powershell
.\run.bat
# Choose option [4] for Enhanced Web Interface
```

Opens http://localhost:7860 with:
- Document upload
- Real-time plagiarism checking
- AI content analysis
- Risk assessment
- Recommendations
- Report download

## Installation Check

```python
# Test AI detection works
from graphplag.detection.ai_detector import AIDetector
detector = AIDetector()
result = detector.detect_ai_content("Test text")
print("✅ AI Detection Ready!")

# Test integrated detector works
from graphplag.detection.integrated_detector import IntegratedDetector
print("✅ Integrated Detector Ready!")
```

## API Endpoints (when extended)

```bash
# Detect AI content
POST /analyze/ai
{
    "text": "Your text here"
}
→ Returns: { "is_ai": bool, "confidence": float, ... }

# Integrated analysis
POST /analyze/integrated
{
    "doc1": "Document 1",
    "doc2": "Document 2"
}
→ Returns: Plagiarism + AI + Risk assessment
```

## Python Integration

```python
# In your existing code
from graphplag.detection.integrated_detector import IntegratedDetector

detector = IntegratedDetector()

# Single line detection
is_ai = detector.ai_detector.detect_ai_content(text)['is_ai']

# Full analysis
results = detector.analyze(submission, rubric)
if results['risk_assessment']['overall_risk_level'] in ['CRITICAL', 'HIGH']:
    flag_submission(results['recommendations'])
```

## Performance Tips

```python
# For single document - use neural (most accurate)
result = detector.detect_ai_content(text, method="neural")

# For batch processing - use statistical (faster)
result = detector.detect_ai_content(text, method="statistical")

# For critical decisions - use ensemble (balanced)
result = detector.detect_ai_content(text, method="ensemble")

# For large documents - truncate first
text = text[:512]  # Most models have 512 token limit
result = detector.detect_ai_content(text)
```

## Examples by Use Case

### Educational Institution
```python
# Check student submissions
for submission in student_submissions:
    result = detector.analyze(submission, assignment_text)
    if result['risk_assessment']['overall_risk_level'] in ['HIGH', 'CRITICAL']:
        notify_instructor(submission, result['recommendations'])
```

### Content Platform
```python
# Verify user content
if detector.ai_detector.detect_ai_content(user_text)['confidence'] > 0.8:
    require_human_verification(user_text)
```

### Publishing
```python
# Pre-publication checks
report = detector.generate_report(article, previous_articles, output_format="html")
if "AI_ASSISTED_PLAGIARISM" in report['risk_factors']:
    request_author_clarification()
```

---

**Need more?** See `DOCUMENTATION.md` or `AI_DETECTION_GUIDE.md`
