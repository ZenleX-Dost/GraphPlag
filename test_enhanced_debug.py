#!/usr/bin/env python
"""Debug script to test compare_documents function directly"""

from graphplag import PlagiarismDetector
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict, Any

def get_text_stats(text: str) -> Dict[str, Any]:
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


def create_empty_plot() -> go.Figure:
    """Create an empty plot figure for error cases"""
    fig = go.Figure()
    fig.update_layout(
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{
            'text': 'No data available',
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 16, 'color': '#999'}
        }],
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_enhanced_similarity_gauge(similarity: float, threshold: float) -> go.Figure:
    """Create an enhanced similarity gauge with better visuals"""
    fig = go.Figure()
    
    # Determine gauge color based on similarity
    gauge_color = (
        "#dc3545" if similarity >= 0.9 else
        "#fd7e14" if similarity >= 0.7 else
        "#ffc107" if similarity >= 0.5 else
        "#28a745"
    )
    
    # Main gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=similarity * 100,
        domain={'x': [0, 1], 'y': [0.15, 1]},
        title={'text': "<b>Similarity Score</b>", 'font': {'size': 26, 'color': '#333'}},
        number={'suffix': "%", 'font': {'size': 52, 'color': gauge_color, 'family': 'Arial Black'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 3, 'tickcolor': "#666", 'tickfont': {'size': 14}},
            'bar': {'color': gauge_color, 'thickness': 1.0},
            'bgcolor': "#f5f5f5",
            'borderwidth': 3,
            'bordercolor': "#ccc",
            'steps': [
                {'range': [0, 50], 'color': '#d4edda'},
                {'range': [50, 70], 'color': '#fff3cd'},
                {'range': [70, 90], 'color': '#ffd4da'},
                {'range': [90, 100], 'color': '#ffb8c8'}
            ],
            'threshold': {
                'line': {'color': "#000", 'width': 8},
                'thickness': 0.9,
                'value': threshold * 100
            }
        }
    ))
    
    # Add interpretation annotation
    interpretation = (
        "🚨 VERY HIGH" if similarity >= 0.9 else
        "⚠️ HIGH" if similarity >= 0.7 else
        "⚡ MODERATE" if similarity >= 0.5 else
        "✅ LOW"
    )
    
    fig.add_annotation(
        text=f"<b>{interpretation}</b>",
        x=0.5, y=0.05,
        showarrow=False,
        font=dict(size=24, color=gauge_color, family='Arial Black')
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=100, b=60),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'family': 'Arial, sans-serif'}
    )
    
    return fig


def create_comparison_stats(doc1: str, doc2: str, similarity: float) -> go.Figure:
    """Create comparison statistics visualization"""
    stats1 = get_text_stats(doc1)
    stats2 = get_text_stats(doc2)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '<b>Document Metrics Comparison</b>',
            '<b>Content Similarity Breakdown</b>'
        ),
        specs=[[{'type': 'bar'}, {'type': 'pie'}]]
    )
    
    # Bar chart comparing document stats
    categories = ['Characters', 'Words', 'Lines', 'Sentences']
    doc1_values = [stats1['chars'], stats1['words'], stats1['lines'], stats1['sentences']]
    doc2_values = [stats2['chars'], stats2['words'], stats2['lines'], stats2['sentences']]
    
    fig.add_trace(
        go.Bar(
            name='Document 1',
            x=categories,
            y=doc1_values,
            marker_color='#4e73df',
            text=doc1_values,
            textposition='outside'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            name='Document 2',
            x=categories,
            y=doc2_values,
            marker_color='#1cc88a',
            text=doc2_values,
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Pie chart for similarity breakdown
    similar_pct = similarity * 100
    unique_pct = (1 - similarity) * 100
    pie_colors = ['#f8d7da', '#d4edda'] if similarity >= 0.7 else ['#fff3cd', '#d4edda']
    
    fig.add_trace(
        go.Pie(
            labels=['Similar', 'Unique'],
            values=[similar_pct, unique_pct],
            marker=dict(colors=pie_colors, line=dict(color='white', width=3)),
            hole=0.5,
            textinfo='label+percent',
            textfont=dict(size=16, color='white', family='Arial Black'),
            hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add center text for donut chart
    fig.add_annotation(
        text=f"<b>{similarity:.0%}</b>",
        x=0.76, y=0.5,
        font=dict(size=28, color='#333', family='Arial Black'),
        showarrow=False,
        xref='paper', yref='paper'
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'family': 'Arial, sans-serif', 'size': 12}
    )
    
    fig.update_xaxes(title_text='<b>Metrics</b>', row=1, col=1)
    fig.update_yaxes(title_text='<b>Count</b>', row=1, col=1)
    
    return fig


# Test with sample texts
doc1 = """Machine learning is a subset of artificial intelligence that provides systems 
the ability to automatically learn and improve from experience without being 
explicitly programmed. It focuses on the development of computer programs that 
can access data and use it to learn for themselves."""

doc2 = """Machine learning, a subset of AI, enables computer systems to learn and 
improve from experience automatically without explicit programming. It emphasizes 
developing programs that can access and utilize data to learn independently."""

print("Testing plagiarism detection...")
detector = PlagiarismDetector(method='ensemble', threshold=0.95, language='en')
report = detector.detect_plagiarism(doc1, doc2)

print(f"Similarity Score: {report.similarity_score:.2%}")
print(f"Is Plagiarism: {report.is_plagiarism}")

print("\nGenerating gauge plot...")
gauge_fig = create_enhanced_similarity_gauge(report.similarity_score, 0.95)
print(f"Gauge figure type: {type(gauge_fig)}")
print(f"Gauge figure has traces: {len(gauge_fig.data)}")

print("\nGenerating stats plot...")
stats_fig = create_comparison_stats(doc1, doc2, report.similarity_score)
print(f"Stats figure type: {type(stats_fig)}")
print(f"Stats figure has traces: {len(stats_fig.data)}")

print("\n✅ All tests passed!")
