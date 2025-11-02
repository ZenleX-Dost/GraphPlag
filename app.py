#!/usr/bin/env python
"""
GraphPlag Interactive Web Interface
Built with Gradio for plagiarism detection
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Tuple, Optional
import time

from graphplag import PlagiarismDetector
from graphplag.detection.report_generator import ReportGenerator
from graphplag.utils.file_parser import FileParser


# Global detector instance
detector = None
report_gen = ReportGenerator()
file_parser = FileParser()


def initialize_detector(method: str, threshold: float, language: str) -> str:
    """Initialize the plagiarism detector with settings"""
    global detector
    try:
        detector = PlagiarismDetector(
            method=method,
            threshold=threshold,
            language=language
        )
        return f"✅ Detector initialized: {method.upper()} method, threshold={threshold:.2f}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def extract_text_from_file(file) -> str:
    """Extract text from uploaded file"""
    if file is None:
        return ""
    
    try:
        # Get file path
        file_path = file.name if hasattr(file, 'name') else str(file)
        
        # Parse the file
        text = file_parser.parse_file(file_path)
        return text
    except Exception as e:
        return f"❌ Error reading file: {str(e)}"


def compare_documents(doc1_text: str, doc1_file, doc2_text: str, doc2_file, 
                     method: str, threshold: float, language: str) -> Tuple[str, str, str]:
    """Compare two documents for plagiarism"""
    # Get document 1 content
    if doc1_file is not None:
        doc1 = extract_text_from_file(doc1_file)
        if doc1.startswith("❌"):
            return doc1, None, ""
    else:
        doc1 = doc1_text
    
    # Get document 2 content
    if doc2_file is not None:
        doc2 = extract_text_from_file(doc2_file)
        if doc2.startswith("❌"):
            return doc2, None, ""
    else:
        doc2 = doc2_text
    
    if not doc1 or not doc2:
        return "❌ Please provide both documents (either text or files)", None, ""
    
    try:
        # Initialize detector if needed
        global detector
        if detector is None or detector.threshold != threshold:
            detector = PlagiarismDetector(
                method=method,
                threshold=threshold,
                language=language
            )
        
        # Detect plagiarism
        start_time = time.time()
        report = detector.detect_plagiarism(doc1, doc2)
        elapsed = time.time() - start_time
        
        # Build result text
        result = f"""
#  Plagiarism Detection Results

**Method:** {method.upper()}
**Processing Time:** {elapsed:.2f}s

---

## Summary
- **Similarity Score:** {report.similarity_score:.2%}
- **Threshold:** {threshold:.2%}
- **Plagiarism Detected:** {'✅ YES' if report.is_plagiarism else '❌ NO'}

## Details
- **Document 1 Length:** {len(doc1)} characters
- **Document 2 Length:** {len(doc2)} characters
- **Method Used:** {report.method}
- **Confidence:** {report.confidence if hasattr(report, 'confidence') else 'N/A'}

---

### Interpretation
"""
        
        if report.similarity_score >= 0.9:
            result += "🔴 **Very High Similarity** - Strong evidence of plagiarism"
        elif report.similarity_score >= 0.7:
            result += "🟠 **High Similarity** - Likely plagiarism detected"
        elif report.similarity_score >= 0.5:
            result += "🟡 **Moderate Similarity** - Possible paraphrasing or common content"
        else:
            result += "🟢 **Low Similarity** - Documents appear original"
        
        # Create visualization
        fig = go.Figure()
        
        # Add similarity gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=report.similarity_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Similarity Score"},
            delta={'reference': threshold * 100},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 90], 'color': "orange"},
                    {'range': [90, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))
        
        # Generate detailed HTML report
        html_report = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px;">
            <h2>Detailed Analysis</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Metric</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Value</th>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">Similarity Score</td>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>{report.similarity_score:.2%}</strong></td>
                </tr>
                <tr style="background-color: #f9f9f9;">
                    <td style="padding: 10px; border: 1px solid #ddd;">Plagiarism Detected</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{'<span style="color: red;">YES</span>' if report.is_plagiarism else '<span style="color: green;">NO</span>'}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">Method</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{method.upper()}</td>
                </tr>
                <tr style="background-color: #f9f9f9;">
                    <td style="padding: 10px; border: 1px solid #ddd;">Processing Time</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{elapsed:.2f}s</td>
                </tr>
            </table>
        </div>
        """
        
        return result, fig, html_report
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nPlease check your input and try again."
        return error_msg, None, ""


def batch_compare(documents_text: str, method: str, threshold: float, 
                  language: str) -> Tuple[str, str, str]:
    """Compare multiple documents and generate similarity matrix"""
    if not documents_text:
        return "❌ Please provide documents", "", ""
    
    try:
        # Parse documents (one per line, separated by "---")
        docs = [doc.strip() for doc in documents_text.split("---") if doc.strip()]
        
        if len(docs) < 2:
            return "❌ Please provide at least 2 documents separated by '---'", "", ""
        
        # Initialize detector
        global detector
        if detector is None:
            detector = PlagiarismDetector(
                method=method,
                threshold=threshold,
                language=language
            )
        
        # Create similarity matrix
        start_time = time.time()
        n = len(docs)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    report = detector.detect_plagiarism(docs[i], docs[j])
                    similarity_matrix[i][j] = report.similarity_score
                    similarity_matrix[j][i] = report.similarity_score
        
        elapsed = time.time() - start_time
        
        # Find suspicious pairs
        suspicious_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] >= threshold:
                    suspicious_pairs.append((i + 1, j + 1, similarity_matrix[i][j]))
        
        # Build result
        result = f"""
#  Batch Comparison Results

**Documents Analyzed:** {n}
**Processing Time:** {elapsed:.2f}s
**Suspicious Pairs:** {len(suspicious_pairs)}

---

## Suspicious Pairs (Similarity ≥ {threshold:.0%})

"""
        
        if suspicious_pairs:
            for doc1, doc2, score in suspicious_pairs:
                result += f"- **Doc {doc1} ↔ Doc {doc2}:** {score:.2%}\n"
        else:
            result += "*No suspicious pairs found.*\n"
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=[f"Doc {i+1}" for i in range(n)],
            y=[f"Doc {i+1}" for i in range(n)],
            colorscale='RdYlGn_r',
            text=np.round(similarity_matrix * 100, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Similarity %")
        ))
        
        fig.update_layout(
            title="Document Similarity Matrix",
            xaxis_title="Documents",
            yaxis_title="Documents",
            height=500,
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Create HTML table
        df = pd.DataFrame(
            similarity_matrix * 100,
            columns=[f"Doc {i+1}" for i in range(n)],
            index=[f"Doc {i+1}" for i in range(n)]
        )
        html_table = df.to_html(float_format=lambda x: f'{x:.1f}%')
        
        html_report = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px;">
            <h2>Similarity Matrix</h2>
            {html_table}
        </div>
        """
        
        return result, fig, html_report
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, None, ""


# Create Gradio interface
with gr.Blocks(title="GraphPlag - Plagiarism Detection", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 🔍 GraphPlag - Advanced Plagiarism Detection System
    
    Detect plagiarism using state-of-the-art graph-based analysis and machine learning.
    
    **Features:**
    - Multiple detection methods (Graph Kernels, GNN, Ensemble)
    - Real-time similarity analysis
    - Batch document comparison
    - Interactive visualizations
    """)
    
    with gr.Tabs():
        
        # Tab 1: Single Comparison
        with gr.Tab(" Compare Two Documents"):
            gr.Markdown("""
            ### Compare two documents for plagiarism
            **Supports:** PDF, DOCX, TXT, MD files or direct text input
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Document 1")
                    doc1_file = gr.File(
                        label="Upload File (PDF, DOCX, TXT, MD)",
                        file_types=[".pdf", ".docx", ".txt", ".md", ".markdown"]
                    )
                    gr.Markdown("**OR enter text directly:**")
                    doc1_input = gr.Textbox(
                        label="",
                        placeholder="Paste or type document content here...",
                        lines=10
                    )
                with gr.Column():
                    gr.Markdown("#### Document 2")
                    doc2_file = gr.File(
                        label="Upload File (PDF, DOCX, TXT, MD)",
                        file_types=[".pdf", ".docx", ".txt", ".md", ".markdown"]
                    )
                    gr.Markdown("**OR enter text directly:**")
                    doc2_input = gr.Textbox(
                        label="",
                        placeholder="Paste or type document content here...",
                        lines=10
                    )
            
            with gr.Row():
                method_single = gr.Dropdown(
                    choices=["kernel", "gnn", "ensemble"],
                    value="kernel",
                    label="Detection Method"
                )
                threshold_single = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="Plagiarism Threshold"
                )
                language_single = gr.Dropdown(
                    choices=["en", "es", "fr", "de"],
                    value="en",
                    label="Language"
                )
            
            compare_btn = gr.Button(" Analyze", variant="primary", size="lg")
            
            with gr.Row():
                result_output = gr.Markdown(label="Results")
            
            with gr.Row():
                plot_output = gr.Plot(label="Similarity Visualization")
            
            with gr.Row():
                html_output = gr.HTML(label="Detailed Report")
            
            compare_btn.click(
                fn=compare_documents,
                inputs=[doc1_input, doc1_file, doc2_input, doc2_file, method_single, threshold_single, language_single],
                outputs=[result_output, plot_output, html_output]
            )
        
        # Tab 2: Batch Comparison
        with gr.Tab(" Batch Compare"):
            gr.Markdown("""
            ### Compare multiple documents at once
            
            **Instructions:** Enter your documents separated by `---` on a new line.
            
            **Example:**
            ```
            This is document 1.
            ---
            This is document 2.
            ---
            This is document 3.
            ```
            """)
            
            batch_input = gr.Textbox(
                label="Documents (separated by '---')",
                placeholder="Document 1\n---\nDocument 2\n---\nDocument 3",
                lines=15
            )
            
            with gr.Row():
                method_batch = gr.Dropdown(
                    choices=["kernel", "gnn", "ensemble"],
                    value="kernel",
                    label="Detection Method"
                )
                threshold_batch = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="Plagiarism Threshold"
                )
                language_batch = gr.Dropdown(
                    choices=["en", "es", "fr", "de"],
                    value="en",
                    label="Language"
                )
            
            batch_btn = gr.Button(" Analyze Batch", variant="primary", size="lg")
            
            with gr.Row():
                batch_result_output = gr.Markdown(label="Results")
            
            with gr.Row():
                batch_plot_output = gr.Plot(label="Similarity Matrix")
            
            with gr.Row():
                batch_html_output = gr.HTML(label="Detailed Table")
            
            batch_btn.click(
                fn=batch_compare,
                inputs=[batch_input, method_batch, threshold_batch, language_batch],
                outputs=[batch_result_output, batch_plot_output, batch_html_output]
            )
        
        # Tab 3: Examples & Help
        with gr.Tab(" Examples & Help"):
            gr.Markdown("""
            ## How to Use GraphPlag
            
            ### Detection Methods
            
            1. **Kernel Method** (Recommended)
               - Fast and accurate
               - Uses Weisfeiler-Lehman graph kernels
               - Best for: Quick comparisons, large batches
            
            2. **GNN Method**
               - Deep learning approach
               - Requires trained model
               - Best for: Complex paraphrasing detection
            
            3. **Ensemble Method**
               - Combines multiple methods
               - Most accurate but slower
               - Best for: High-stakes detection
            
            ### Understanding Results
            
            - **0-50% Similarity:** Documents are original
            - **50-70% Similarity:** Moderate similarity, possible common sources
            - **70-90% Similarity:** High similarity, likely plagiarism
            - **90-100% Similarity:** Very high similarity, strong evidence of copying
            
            ### Example Documents
            
            #### Example 1: High Similarity
            **Document 1:**
            ```
            Machine learning is a subset of artificial intelligence that enables 
            systems to learn from data and improve their performance over time 
            without being explicitly programmed.
            ```
            
            **Document 2:**
            ```
            Machine learning, a subset of AI, allows systems to learn from data 
            and enhance performance over time without explicit programming.
            ```
            
            #### Example 2: Low Similarity
            **Document 1:**
            ```
            The Python programming language was created by Guido van Rossum 
            and first released in 1991. It emphasizes code readability.
            ```
            
            **Document 2:**
            ```
            Climate change refers to long-term shifts in global temperatures 
            and weather patterns, primarily caused by human activities.
            ```
            
            ### Tips
            
            - Longer documents provide more accurate results
            - Adjust threshold based on your use case
            - Use batch comparison for efficient multiple document analysis
            - Review matched segments for context
            
            ### System Information
            
            - **Version:** GraphPlag v0.1.0
            - **Backend:** PyTorch, spaCy, GraKeL
            - **Language Support:** English, Spanish, French, German
            - **Processing Time:** ~0.3s per document pair
            """)
    
    gr.Markdown("""
    ---
    **GraphPlag** | Graph-based Plagiarism Detection System
    
    For more information, visit the [GitHub repository](https://github.com/ZenleX-Dost/GraphPlag)
    """)


if __name__ == "__main__":
    print(" Starting GraphPlag Interactive Interface...")
    print(" Loading models... (this may take a moment)")
    
    # Launch the app
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )
