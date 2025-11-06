#!/usr/bin/env python
"""
GraphPlag Enhanced Interactive Web Interface
Modern, interactive plagiarism detection with beautiful UI
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Tuple, Optional, Dict
import time
from datetime import datetime

from graphplag import PlagiarismDetector
from graphplag.detection.report_generator import ReportGenerator
from graphplag.utils.file_parser import FileParser


# Global state
detector = None
report_gen = ReportGenerator()
file_parser = FileParser()
comparison_history = []


# Custom CSS for modern styling
custom_css = """
/* Main container styling */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Header styling with gradient */
.header-gradient {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}

/* Card styling */
.custom-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    transition: transform 0.2s, box-shadow 0.2s;
}

.custom-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
}

/* Button styling */
.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 8px !important;
    transition: all 0.3s !important;
}

.primary-button:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4) !important;
}

/* Stats card */
.stats-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin: 0.5rem;
}

/* Pulse animation for loading */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 1.5s ease-in-out infinite;
}

/* File upload area */
.file-upload {
    border: 2px dashed #667eea;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s;
}

.file-upload:hover {
    border-color: #764ba2;
    background: rgba(102, 126, 234, 0.05);
}

/* Result badge */
.result-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    margin: 0.5rem;
}

.badge-success {
    background: #d4edda;
    color: #155724;
}

.badge-warning {
    background: #fff3cd;
    color: #856404;
}

.badge-danger {
    background: #f8d7da;
    color: #721c24;
}
"""


def get_text_stats(text: str) -> Dict[str, any]:
    """Get statistics about text"""
    if not text:
        return {
            'chars': 0,
            'words': 0,
            'lines': 0,
            'sentences': 0
        }
    
    words = len(text.split())
    lines = len(text.split('\n'))
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return {
        'chars': len(text),
        'words': words,
        'lines': lines,
        'sentences': max(sentences, 1)
    }


def format_stats_display(stats: Dict[str, any]) -> str:
    """Format statistics for display"""
    return f"""
    <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0;">
        <div class="stats-card">
            <div style="font-size: 1.5rem; font-weight: bold;">{stats['chars']:,}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Characters</div>
        </div>
        <div class="stats-card">
            <div style="font-size: 1.5rem; font-weight: bold;">{stats['words']:,}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Words</div>
        </div>
        <div class="stats-card">
            <div style="font-size: 1.5rem; font-weight: bold;">{stats['lines']:,}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Lines</div>
        </div>
        <div class="stats-card">
            <div style="font-size: 1.5rem; font-weight: bold;">{stats['sentences']:,}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Sentences</div>
        </div>
    </div>
    """


def update_text_stats(text: str) -> str:
    """Update text statistics in real-time"""
    stats = get_text_stats(text)
    return format_stats_display(stats)


def extract_text_from_file(file) -> Tuple[str, str]:
    """Extract text from uploaded file and return text + stats"""
    if file is None:
        return "", ""
    
    try:
        file_path = file.name if hasattr(file, 'name') else str(file)
        text = file_parser.parse_file(file_path)
        stats = get_text_stats(text)
        stats_html = format_stats_display(stats)
        return text, stats_html
    except Exception as e:
        return f"Error reading file: {str(e)}", ""


def compare_documents(doc1_text: str, doc1_file, doc2_text: str, doc2_file, 
                     method: str, threshold: float, language: str) -> Tuple[str, str, str, str]:
    """Compare two documents with enhanced visualization"""
    
    # Get document contents
    if doc1_file is not None:
        doc1, _ = extract_text_from_file(doc1_file)
        if doc1.startswith(""):
            return doc1, None, "", ""
    else:
        doc1 = doc1_text
    
    if doc2_file is not None:
        doc2, _ = extract_text_from_file(doc2_file)
        if doc2.startswith(""):
            return doc2, None, "", ""
    else:
        doc2 = doc2_text
    
    if not doc1 or not doc2:
        return "Please provide both documents (either text or files)", None, "", ""
    
    try:
        # Initialize detector
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
        
        # Add to history
        comparison_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'similarity': report.similarity_score,
            'method': method,
            'is_plagiarism': report.is_plagiarism
        })
        
        # Build result with enhanced HTML
        similarity_color = (
            "#dc3545" if report.similarity_score >= 0.9 else
            "#fd7e14" if report.similarity_score >= 0.7 else
            "#ffc107" if report.similarity_score >= 0.5 else
            "#28a745"
        )
        
        badge_class = (
            "badge-danger" if report.similarity_score >= 0.9 else
            "badge-warning" if report.similarity_score >= 0.7 else
            "badge-warning" if report.similarity_score >= 0.5 else
            "badge-success"
        )
        
        result = f"""
        <div style="padding: 2rem; background: white; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <h2 style="color: #667eea; margin-bottom: 1.5rem;">Analysis Results</h2>
            
            <div style="text-align: center; margin: 2rem 0;">
                <div style="font-size: 4rem; font-weight: bold; color: {similarity_color};">
                    {report.similarity_score:.1%}
                </div>
                <div style="font-size: 1.2rem; color: #666; margin-top: 0.5rem;">
                    Similarity Score
                </div>
                <div class="result-badge {badge_class}" style="margin-top: 1rem; font-size: 1.1rem;">
                    {'PLAGIARISM DETECTED' if report.is_plagiarism else 'NO PLAGIARISM'}
                </div>
            </div>
            
            <hr style="margin: 2rem 0; border: none; border-top: 2px solid #f0f0f0;">
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <div style="color: #666; font-size: 0.9rem;">Method</div>
                    <div style="font-weight: bold; font-size: 1.1rem; color: #667eea;">{method.upper()}</div>
                </div>
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <div style="color: #666; font-size: 0.9rem;">Processing Time</div>
                    <div style="font-weight: bold; font-size: 1.1rem; color: #667eea;">{elapsed:.2f}s</div>
                </div>
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <div style="color: #666; font-size: 0.9rem;">Threshold</div>
                    <div style="font-weight: bold; font-size: 1.1rem; color: #667eea;">{threshold:.0%}</div>
                </div>
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <div style="color: #666; font-size: 0.9rem;">Language</div>
                    <div style="font-weight: bold; font-size: 1.1rem; color: #667eea;">{language.upper()}</div>
                </div>
            </div>
            
            <hr style="margin: 2rem 0; border: none; border-top: 2px solid #f0f0f0;">
            
            <h3 style="color: #667eea; margin-bottom: 1rem;">Interpretation</h3>
            <div style="padding: 1.5rem; background: #f8f9fa; border-radius: 8px; border-left: 4px solid {similarity_color};">
        """
        
        if report.similarity_score >= 0.9:
            result += """
                <strong style="color: #dc3545;">Very High Similarity</strong>
                <p style="margin-top: 0.5rem; color: #666;">
                Strong evidence of plagiarism. The documents share extremely similar content and structure.
                This level of similarity typically indicates direct copying or minimal paraphrasing.
                </p>
            """
        elif report.similarity_score >= 0.7:
            result += """
                <strong style="color: #fd7e14;"> High Similarity</strong>
                <p style="margin-top: 0.5rem; color: #666;">
                Significant similarity detected. The documents likely share substantial content, possibly with some paraphrasing.
                Further investigation recommended to determine if proper attribution exists.
                </p>
            """
        elif report.similarity_score >= 0.5:
            result += """
                <strong style="color: #ffc107;"> Moderate Similarity</strong>
                <p style="margin-top: 0.5rem; color: #666;">
                Moderate content overlap detected. Could indicate shared sources, common domain terminology,
                or limited paraphrasing. Review the context to determine if this is acceptable.
                </p>
            """
        else:
            result += """
                <strong style="color: #28a745;"> Low Similarity</strong>
                <p style="margin-top: 0.5rem; color: #666;">
                The documents appear to be largely original with minimal overlap. This is typical of independent work
                or documents on different topics. No plagiarism concerns detected.
                </p>
            """
        
        result += """
            </div>
            
            <div style="margin-top: 2rem; padding: 1rem; background: #e7f3ff; border-radius: 8px; border-left: 4px solid #667eea;">
                <strong style="color: #667eea;">Tip:</strong>
                <span style="color: #666;">
                Adjust the threshold slider to change sensitivity. Higher thresholds require stronger evidence for plagiarism detection.
                </span>
            </div>
        </div>
        """
        
        # Create enhanced visualization
        fig = create_enhanced_similarity_gauge(report.similarity_score, threshold)
        
        # Create detailed comparison chart
        stats_fig = create_comparison_stats(doc1, doc2, report.similarity_score)
        
        # History display
        history_html = create_history_display()
        
        return result, fig, stats_fig, history_html
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"""
        <div style="padding: 2rem; background: #fff5f5; border-radius: 15px; border: 2px solid #fc8181;">
            <h3 style="color: #c53030;">Error Occurred</h3>
            <p style="color: #742a2a; margin-top: 1rem;"><strong>Message:</strong> {str(e)}</p>
            <details style="margin-top: 1rem;">
                <summary style="cursor: pointer; color: #c53030;">Show Technical Details</summary>
                <pre style="background: white; padding: 1rem; border-radius: 8px; margin-top: 0.5rem; overflow-x: auto;">
{error_details}
                </pre>
            </details>
        </div>
        """
        return error_msg, None, None, ""


def create_enhanced_similarity_gauge(similarity: float, threshold: float) -> go.Figure:
    """Create an enhanced similarity gauge with better visuals"""
    fig = go.Figure()
    
    # Main gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=similarity * 100,
        domain={'x': [0, 1], 'y': [0.2, 1]},
        title={'text': "<b>Similarity Score</b>", 'font': {'size': 24, 'color': '#667eea'}},
        delta={'reference': threshold * 100, 'increasing': {'color': "#dc3545"}},
        number={'suffix': "%", 'font': {'size': 48}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#667eea"},
            'bar': {'color': "#667eea", 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, 50], 'color': '#d4edda'},
                {'range': [50, 70], 'color': '#fff3cd'},
                {'range': [70, 90], 'color': '#f8d7da'},
                {'range': [90, 100], 'color': '#f5c6cb'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 6},
                'thickness': 0.85,
                'value': threshold * 100
            }
        }
    ))
    
    # Add annotation
    interpretation = (
        "Very High" if similarity >= 0.9 else
        "High" if similarity >= 0.7 else
        "Moderate" if similarity >= 0.5 else
        "Low"
    )
    
    fig.add_annotation(
        text=f"<b>{interpretation} Similarity</b>",
        x=0.5, y=0.1,
        showarrow=False,
        font=dict(size=20, color='#667eea')
    )
    
    fig.update_layout(
        height=450,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif'}
    )
    
    return fig


def create_comparison_stats(doc1: str, doc2: str, similarity: float) -> go.Figure:
    """Create comparison statistics visualization"""
    stats1 = get_text_stats(doc1)
    stats2 = get_text_stats(doc2)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Document Comparison', 'Similarity Breakdown'),
        specs=[[{'type': 'bar'}, {'type': 'pie'}]]
    )
    
    # Bar chart comparing document stats
    categories = ['Characters', 'Words', 'Lines', 'Sentences']
    doc1_values = [stats1['chars'], stats1['words'], stats1['lines'], stats1['sentences']]
    doc2_values = [stats2['chars'], stats2['words'], stats2['lines'], stats2['sentences']]
    
    fig.add_trace(
        go.Bar(name='Document 1', x=categories, y=doc1_values, marker_color='#667eea'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Document 2', x=categories, y=doc2_values, marker_color='#764ba2'),
        row=1, col=1
    )
    
    # Pie chart for similarity breakdown
    fig.add_trace(
        go.Pie(
            labels=['Similar Content', 'Unique Content'],
            values=[similarity * 100, (1 - similarity) * 100],
            marker=dict(colors=['#f8d7da', '#d4edda']),
            hole=0.4
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif'}
    )
    
    return fig


def create_history_display() -> str:
    """Create HTML display of comparison history"""
    if not comparison_history:
        return """
        <div style="padding: 2rem; text-align: center; color: #999;">
            <p>No comparison history yet. Start analyzing documents to see your history here!</p>
        </div>
        """
    
    html = """
    <div style="padding: 1.5rem; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea; margin-bottom: 1rem;">Recent Comparisons</h3>
        <div style="max-height: 400px; overflow-y: auto;">
    """
    
    for i, entry in enumerate(reversed(comparison_history[-10:])):  # Last 10 entries
        badge_color = "#dc3545" if entry['is_plagiarism'] else "#28a745"
        badge_text = "Plagiarism" if entry['is_plagiarism'] else "Clean"
        
        html += f"""
        <div style="padding: 1rem; margin-bottom: 0.5rem; background: #f8f9fa; border-radius: 8px; border-left: 4px solid {badge_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: #667eea;">#{len(comparison_history) - i}</strong>
                    <span style="color: #999; margin-left: 1rem; font-size: 0.9rem;">{entry['timestamp']}</span>
                </div>
                <div>
                    <span style="background: {badge_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85rem;">
                        {badge_text}
                    </span>
                </div>
            </div>
            <div style="margin-top: 0.5rem; color: #666;">
                Similarity: <strong>{entry['similarity']:.1%}</strong> | Method: <strong>{entry['method'].upper()}</strong>
            </div>
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html


def load_example(example_name: str) -> Tuple[str, str]:
    """Load example documents"""
    examples = {
        "High Similarity": (
            """Machine learning is a subset of artificial intelligence that provides systems 
            the ability to automatically learn and improve from experience without being 
            explicitly programmed. It focuses on the development of computer programs that 
            can access data and use it to learn for themselves.""",
            """Machine learning, a subset of AI, enables computer systems to learn and 
            improve from experience automatically without explicit programming. It emphasizes 
            developing programs that can access and utilize data to learn independently."""
        ),
        "Low Similarity": (
            """The Python programming language was created by Guido van Rossum and first 
            released in 1991. It emphasizes code readability with significant use of whitespace.""",
            """Climate change refers to long-term shifts in global temperatures and weather 
            patterns, primarily caused by human activities like burning fossil fuels."""
        ),
        "Academic Text": (
            """This research examines the impact of social media on adolescent mental health. 
            The study employed a mixed-methods approach, combining quantitative surveys with 
            qualitative interviews. Results indicate a correlation between social media usage 
            and anxiety levels among teenagers.""",
            """The present study investigates how social media affects the mental well-being 
            of adolescents. Using both quantitative and qualitative methods, we found a 
            relationship between time spent on social platforms and increased anxiety in teens."""
        )
    }
    
    return examples.get(example_name, ("", ""))


# Create the enhanced Gradio interface
with gr.Blocks(
    title="GraphPlag - Advanced Plagiarism Detection",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
    ),
    css=custom_css
) as app:
    
    # Header
    gr.HTML("""
    <div class="header-gradient">
        <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">
            GraphPlag
        </h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Advanced Graph-Based Plagiarism Detection System
        </p>
        <div style="margin-top: 1rem; display: flex; gap: 1rem; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                Real-time Analysis
            </span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                Multi-format Support
            </span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                High Accuracy
            </span>
        </div>
    </div>
    """)
    
    with gr.Tabs() as tabs:
        
        # Tab 1: Compare Documents
        with gr.Tab("Compare Documents"):
            gr.Markdown("""
            ### Upload documents or paste text to detect plagiarism
            Supports: **PDF**, **DOCX**, **TXT**, **Markdown**
            """)
            
            # Example selector
            with gr.Row():
                example_dropdown = gr.Dropdown(
                    choices=["High Similarity", "Low Similarity", "Academic Text"],
                    label="Try an Example",
                    info="Load pre-made examples to test the system"
                )
                load_example_btn = gr.Button("Load Example", variant="secondary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Document 1")
                    doc1_file = gr.File(
                        label="Upload File",
                        file_types=[".pdf", ".docx", ".txt", ".md", ".markdown"],
                        file_count="single"
                    )
                    doc1_input = gr.Textbox(
                        label="Or Paste Text",
                        placeholder="Enter document content here...",
                        lines=12,
                        max_lines=20
                    )
                    doc1_stats = gr.HTML(label="Document Statistics")
                    
                with gr.Column():
                    gr.Markdown("#### Document 2")
                    doc2_file = gr.File(
                        label="Upload File",
                        file_types=[".pdf", ".docx", ".txt", ".md", ".markdown"],
                        file_count="single"
                    )
                    doc2_input = gr.Textbox(
                        label="Or Paste Text",
                        placeholder="Enter document content here...",
                        lines=12,
                        max_lines=20
                    )
                    doc2_stats = gr.HTML(label="Document Statistics")
            
            # Settings
            with gr.Row():
                method_single = gr.Dropdown(
                    choices=["kernel", "gnn", "ensemble"],
                    value="kernel",
                    label="Detection Method",
                    info="Kernel: Fast | GNN: Advanced | Ensemble: Most Accurate"
                )
                threshold_single = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="Sensitivity Threshold",
                    info="Higher = Stricter detection"
                )
                language_single = gr.Dropdown(
                    choices=["en", "es", "fr", "de"],
                    value="en",
                    label="Language"
                )
            
            compare_btn = gr.Button(
                "Analyze Documents",
                variant="primary",
                size="lg",
                elem_classes="primary-button"
            )
            
            # Results
            with gr.Row():
                result_output = gr.HTML(label="Analysis Results")
            
            with gr.Row():
                with gr.Column():
                    plot_output = gr.Plot(label="Similarity Gauge")
                with gr.Column():
                    stats_plot = gr.Plot(label="Document Comparison")
            
            with gr.Row():
                history_output = gr.HTML(label="Comparison History")
            
            # Event handlers
            doc1_input.change(
                fn=update_text_stats,
                inputs=[doc1_input],
                outputs=[doc1_stats]
            )
            
            doc2_input.change(
                fn=update_text_stats,
                inputs=[doc2_input],
                outputs=[doc2_stats]
            )
            
            doc1_file.change(
                fn=extract_text_from_file,
                inputs=[doc1_file],
                outputs=[doc1_input, doc1_stats]
            )
            
            doc2_file.change(
                fn=extract_text_from_file,
                inputs=[doc2_file],
                outputs=[doc2_input, doc2_stats]
            )
            
            load_example_btn.click(
                fn=load_example,
                inputs=[example_dropdown],
                outputs=[doc1_input, doc2_input]
            )
            
            compare_btn.click(
                fn=compare_documents,
                inputs=[doc1_input, doc1_file, doc2_input, doc2_file, 
                       method_single, threshold_single, language_single],
                outputs=[result_output, plot_output, stats_plot, history_output]
            )
        
        # Tab 2: Batch Compare (keeping original functionality)
        with gr.Tab("Batch Compare"):
            gr.Markdown("""
            ### Compare multiple documents at once
            
            **Instructions:** Enter your documents separated by `---` on a new line.
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
            
            batch_btn = gr.Button("Analyze Batch", variant="primary", size="lg")
            
            with gr.Row():
                batch_result_output = gr.Markdown(label="Results")
            
            with gr.Row():
                batch_plot_output = gr.Plot(label="Similarity Matrix")
        
        # Tab 3: About & Help
        with gr.Tab("ℹAbout & Help"):
            gr.Markdown("""
            ## How to Use GraphPlag
            
            ### Quick Start
            1. **Upload** or **paste** your documents
            2. **Select** detection method and threshold
            3. Click **Analyze** to get results
            
            ### Detection Methods
            
            | Method | Speed | Accuracy | Best For |
            |--------|-------|----------|----------|
            | **Kernel** | Fast | Good | Quick checks, large batches |
            | **GNN** | Slow | Great | Complex paraphrasing |
            | **Ensemble** | Slower | Best | Critical analysis |
            
            ### Similarity Scores
            
            - **0-50%**  Low similarity - Documents are original
            - **50-70%**  Moderate similarity - Possible shared sources
            - **70-90%**  High similarity - Likely plagiarism
            - **90-100%** Very high similarity - Strong evidence of copying
            
            ### Supported File Formats
            
            - **PDF** - Portable Document Format
            - **DOCX** - Microsoft Word
            - **TXT** - Plain Text
            - **MD** - Markdown
            
            ### Tips for Best Results
            
            1. Use longer documents (100+ words) for better accuracy
            2. Ensure files are text-based (not scanned images)
            3. Adjust threshold based on your use case
            4. Try different methods for comparison
            5. Review context, not just scores
            
            ### System Information
            
            - **Version:** GraphPlag v0.1.0
            - **Engine:** PyTorch + spaCy + GraKeL
            - **Language Models:** Multilingual MPNET
            - **Graph Kernels:** Weisfeiler-Lehman
            
            ---
            
            **Need help?** Check our [documentation](https://github.com/ZenleX-Dost/GraphPlag) or report issues on GitHub.
            """)
    
    # Footer
    gr.HTML("""
    <div style="margin-top: 3rem; padding: 2rem; background: #f8f9fa; border-radius: 12px; text-align: center;">
        <p style="color: #667eea; font-weight: 600; margin-bottom: 0.5rem;">
            GraphPlag - Graph-Based Plagiarism Detection
        </p>
        <p style="color: #999; font-size: 0.9rem;">
            Powered by AI • Open Source • Academic Research
        </p>
        <p style="margin-top: 1rem;">
            <a href="https://github.com/ZenleX-Dost/GraphPlag" style="color: #667eea; text-decoration: none;">
                Star on GitHub
            </a>
        </p>
    </div>
    """)


if __name__ == "__main__":
    print("Starting GraphPlag Enhanced Interface...")
    print("Loading models... (this may take a moment)")
    print("Open http://localhost:7860 in your browser")
    print("Press Ctrl+C to stop the server")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None
    )
