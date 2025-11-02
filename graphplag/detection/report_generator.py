"""
Report Generator Module

Generates comprehensive plagiarism detection reports with visualizations.
"""

from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from graphplag.core.models import PlagiarismReport


class ReportGenerator:
    """
    Generate plagiarism detection reports.
    
    Creates visual and textual reports from detection results.
    """
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_text_report(self, report: PlagiarismReport) -> str:
        """
        Generate text-based report.
        
        Args:
            report: PlagiarismReport object
            
        Returns:
            Formatted text report
        """
        return report.summary()
    
    def generate_html_report(
        self,
        report: PlagiarismReport,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate HTML report with visualizations.
        
        Args:
            report: PlagiarismReport object
            output_file: Optional output file path
            
        Returns:
            HTML content
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Plagiarism Detection Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                .status {{
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 5px;
                    font-weight: bold;
                    font-size: 18px;
                }}
                .plagiarism {{
                    background-color: #ffebee;
                    color: #c62828;
                    border-left: 5px solid #c62828;
                }}
                .no-plagiarism {{
                    background-color: #e8f5e9;
                    color: #2e7d32;
                    border-left: 5px solid #2e7d32;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 20px;
                    padding: 15px;
                    background-color: #f5f5f5;
                    border-radius: 5px;
                }}
                .metric-label {{
                    font-size: 12px;
                    color: #666;
                    text-transform: uppercase;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #333;
                }}
                .matches {{
                    margin-top: 30px;
                }}
                .match {{
                    background-color: #fff3e0;
                    padding: 10px;
                    margin: 10px 0;
                    border-left: 4px solid #ff9800;
                    border-radius: 3px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Plagiarism Detection Report</h1>
                
                <div class="status {'plagiarism' if report.is_plagiarism else 'no-plagiarism'}">
                    {'PLAGIARISM DETECTED' if report.is_plagiarism else 'NO PLAGIARISM DETECTED'}
                </div>
                
                <h2>Document Information</h2>
                <table>
                    <tr>
                        <th>Property</th>
                        <th>Document 1</th>
                        <th>Document 2</th>
                    </tr>
                    <tr>
                        <td>Document ID</td>
                        <td>{report.document1.doc_id or 'N/A'}</td>
                        <td>{report.document2.doc_id or 'N/A'}</td>
                    </tr>
                    <tr>
                        <td>Language</td>
                        <td>{report.document1.language.value}</td>
                        <td>{report.document2.language.value}</td>
                    </tr>
                    <tr>
                        <td>Number of Sentences</td>
                        <td>{len(report.document1.sentences)}</td>
                        <td>{len(report.document2.sentences)}</td>
                    </tr>
                </table>
                
                <h2>Similarity Metrics</h2>
                <div>
                    <div class="metric">
                        <div class="metric-label">Overall Similarity</div>
                        <div class="metric-value">{report.similarity_score:.1%}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Detection Method</div>
                        <div class="metric-value">{report.method.upper()}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Threshold</div>
                        <div class="metric-value">{report.threshold:.1%}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Processing Time</div>
                        <div class="metric-value">{report.processing_time:.2f}s</div>
                    </div>
                </div>
                
                {self._generate_kernel_scores_html(report) if report.kernel_scores else ''}
                {self._generate_matches_html(report) if report.matches else ''}
                
            </div>
        </body>
        </html>
        """
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return html_content
    
    def _generate_kernel_scores_html(self, report: PlagiarismReport) -> str:
        """Generate HTML for kernel scores section."""
        if not report.kernel_scores:
            return ""
        
        rows = ""
        for kernel, score in report.kernel_scores.items():
            rows += f"""
            <tr>
                <td>{kernel.upper()}</td>
                <td>{score:.4f}</td>
                <td>{score:.1%}</td>
            </tr>
            """
        
        return f"""
        <h2>Kernel Scores</h2>
        <table>
            <tr>
                <th>Kernel Type</th>
                <th>Score</th>
                <th>Percentage</th>
            </tr>
            {rows}
        </table>
        """
    
    def _generate_matches_html(self, report: PlagiarismReport) -> str:
        """Generate HTML for matches section."""
        if not report.matches:
            return ""
        
        matches_html = ""
        for i, match in enumerate(report.matches[:10], 1):  # Show top 10 matches
            sent1_idx = match.doc1_segment[0]
            sent2_idx = match.doc2_segment[0]
            
            sent1_text = report.document1.sentences[sent1_idx].text if sent1_idx < len(report.document1.sentences) else ""
            sent2_text = report.document2.sentences[sent2_idx].text if sent2_idx < len(report.document2.sentences) else ""
            
            matches_html += f"""
            <div class="match">
                <strong>Match {i}</strong> (Similarity: {match.similarity:.1%})<br>
                <strong>Doc 1, Sentence {sent1_idx+1}:</strong> {sent1_text[:200]}...<br>
                <strong>Doc 2, Sentence {sent2_idx+1}:</strong> {sent2_text[:200]}...
            </div>
            """
        
        return f"""
        <div class="matches">
            <h2>Top Matches Found: {len(report.matches)}</h2>
            {matches_html}
        </div>
        """
    
    def plot_similarity_heatmap(
        self,
        similarity_matrix: np.ndarray,
        labels: Optional[list] = None,
        output_file: Optional[str] = None
    ):
        """
        Plot similarity heatmap for multiple documents.
        
        Args:
            similarity_matrix: Similarity matrix (n x n)
            labels: Optional labels for documents
            output_file: Optional output file path
        """
        plt.figure(figsize=(10, 8))
        
        if labels is None:
            labels = [f"Doc {i+1}" for i in range(len(similarity_matrix))]
        
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=labels,
            yticklabels=labels,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Similarity Score'}
        )
        
        plt.title('Document Similarity Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def save_report(
        self,
        report: PlagiarismReport,
        filename: str = "report.html"
    ):
        """
        Save report to file.
        
        Args:
            report: PlagiarismReport object
            filename: Output filename
        """
        import os
        output_path = os.path.join(self.output_dir, filename)
        self.generate_html_report(report, output_file=output_path)
        print(f"Report saved to: {output_path}")
    
    def __repr__(self) -> str:
        return f"ReportGenerator(output_dir='{self.output_dir}')"
