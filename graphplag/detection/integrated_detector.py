"""
Combined Plagiarism and AI Detection Interface

This module provides a unified interface for detecting both plagiarism
and AI-generated content in documents.
"""

from typing import Optional, Union, Dict, List
import time

from graphplag.core.models import Document
from graphplag.detection.detector import PlagiarismDetector
from graphplag.detection.ai_detector import AIDetector


class IntegratedDetector:
    """
    Integrated system for detecting both plagiarism and AI-generated content.
    """
    
    def __init__(
        self,
        plagiarism_method: str = "ensemble",
        plagiarism_threshold: float = 0.7,
        language: str = "en",
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
        ai_detection_enabled: bool = True,
        ai_model_name: str = "openai-community/roberta-base-openai-detector"
    ):
        """
        Initialize integrated detector.
        
        Args:
            plagiarism_method: Method for plagiarism detection
            plagiarism_threshold: Plagiarism detection threshold
            language: Language for parsing
            embedding_model: Sentence embedding model
            ai_detection_enabled: Whether to enable AI detection
            ai_model_name: Model for AI detection
        """
        self.plagiarism_detector = PlagiarismDetector(
            method=plagiarism_method,
            threshold=plagiarism_threshold,
            language=language,
            embedding_model=embedding_model
        )
        
        self.ai_detector = AIDetector(model_name=ai_model_name) if ai_detection_enabled else None
        self.ai_detection_enabled = ai_detection_enabled
    
    def analyze(
        self,
        doc1: Union[str, Document],
        doc2: Union[str, Document],
        doc1_id: Optional[str] = None,
        doc2_id: Optional[str] = None,
        check_plagiarism: bool = True,
        check_ai: bool = True
    ) -> Dict:
        """
        Comprehensive analysis for plagiarism and AI content.
        
        Args:
            doc1: First document (text or Document object)
            doc2: Second document (text or Document object)
            doc1_id: Optional ID for first document
            doc2_id: Optional ID for second document
            check_plagiarism: Whether to check for plagiarism
            check_ai: Whether to check for AI-generated content
            
        Returns:
            Dictionary with:
            - plagiarism_results: Plagiarism detection results
            - ai_results: AI detection results
            - risk_assessment: Combined risk evaluation
            - recommendations: Action recommendations
        """
        start_time = time.time()
        
        results = {
            "analysis_metadata": {
                "document_1_id": doc1_id,
                "document_2_id": doc2_id,
                "timestamp": time.time()
            },
            "plagiarism_results": None,
            "ai_results": None,
            "risk_assessment": {},
            "recommendations": []
        }
        
        # Plagiarism detection
        if check_plagiarism:
            plagiarism_report = self.plagiarism_detector.detect_plagiarism(
                doc1, doc2, doc1_id=doc1_id, doc2_id=doc2_id
            )
            results["plagiarism_results"] = {
                "similarity_score": plagiarism_report.similarity_score,
                "is_plagiarized": plagiarism_report.is_plagiarized,
                "matches": len(plagiarism_report.matches) if plagiarism_report.matches else 0,
                "details": plagiarism_report.__dict__
            }
        
        # AI detection
        if check_ai and self.ai_detection_enabled:
            doc1_text = doc1 if isinstance(doc1, str) else doc1.text if hasattr(doc1, 'text') else str(doc1)
            doc2_text = doc2 if isinstance(doc2, str) else doc2.text if hasattr(doc2, 'text') else str(doc2)
            
            ai_result_1 = self.ai_detector.detect_ai_content(doc1_text, method="ensemble")
            ai_result_2 = self.ai_detector.detect_ai_content(doc2_text, method="ensemble")
            
            results["ai_results"] = {
                "document_1": ai_result_1,
                "document_2": ai_result_2,
                "both_ai": ai_result_1["is_ai"] and ai_result_2["is_ai"],
                "at_least_one_ai": ai_result_1["is_ai"] or ai_result_2["is_ai"]
            }
        
        # Risk assessment
        results["risk_assessment"] = self._assess_risk(results)
        
        # Recommendations
        results["recommendations"] = self._generate_recommendations(results)
        
        results["analysis_metadata"]["processing_time"] = time.time() - start_time
        
        return results
    
    def _assess_risk(self, results: Dict) -> Dict:
        """
        Assess overall integrity risk based on plagiarism and AI detection.
        """
        risk = {
            "overall_risk_level": "LOW",
            "risk_score": 0.0,
            "risk_factors": []
        }
        
        risk_score = 0.0
        
        # Plagiarism risk
        if results["plagiarism_results"]:
            plag_score = results["plagiarism_results"]["similarity_score"]
            risk_score += plag_score * 0.6  # 60% weight to plagiarism
            
            if plag_score > 0.9:
                risk["risk_factors"].append("CRITICAL_PLAGIARISM")
            elif plag_score > 0.7:
                risk["risk_factors"].append("HIGH_PLAGIARISM")
            elif plag_score > 0.5:
                risk["risk_factors"].append("MODERATE_PLAGIARISM")
        
        # AI detection risk
        if results["ai_results"]:
            ai_score_1 = results["ai_results"]["document_1"]["confidence"]
            ai_score_2 = results["ai_results"]["document_2"]["confidence"]
            avg_ai_score = (ai_score_1 + ai_score_2) / 2
            risk_score += avg_ai_score * 0.4  # 40% weight to AI detection
            
            if results["ai_results"]["both_ai"]:
                risk["risk_factors"].append("BOTH_DOCUMENTS_AI_GENERATED")
            elif results["ai_results"]["at_least_one_ai"]:
                risk["risk_factors"].append("AT_LEAST_ONE_AI_GENERATED")
        
        risk["risk_score"] = min(risk_score, 1.0)
        
        # Determine risk level
        if risk["risk_score"] > 0.8:
            risk["overall_risk_level"] = "CRITICAL"
        elif risk["risk_score"] > 0.6:
            risk["overall_risk_level"] = "HIGH"
        elif risk["risk_score"] > 0.4:
            risk["overall_risk_level"] = "MODERATE"
        elif risk["risk_score"] > 0.2:
            risk["overall_risk_level"] = "LOW"
        else:
            risk["overall_risk_level"] = "MINIMAL"
        
        return risk
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """
        Generate action recommendations based on analysis results.
        """
        recommendations = []
        
        risk = results.get("risk_assessment", {})
        
        if risk.get("overall_risk_level") == "CRITICAL":
            recommendations.append("REJECT: Critical integrity issues detected")
            recommendations.append("ESCALATE: Refer to institutional review board")
        elif risk.get("overall_risk_level") == "HIGH":
            recommendations.append("REVIEW: Significant integrity concerns require manual review")
            recommendations.append("CONTACT: Interview author to discuss findings")
        elif risk.get("overall_risk_level") == "MODERATE":
            recommendations.append("CAUTION: Moderate similarity detected, review specific sections")
            recommendations.append("CLARIFY: Request author clarification on borrowed content")
        else:
            recommendations.append("ACCEPT: Low integrity risk detected")
        
        # Specific recommendations based on risk factors
        if "BOTH_DOCUMENTS_AI_GENERATED" in risk.get("risk_factors", []):
            recommendations.append("ALERT: Both documents show signs of AI generation")
        
        if "CRITICAL_PLAGIARISM" in risk.get("risk_factors", []):
            recommendations.append("VERIFY: Check for proper attribution of sources")
        
        if "AT_LEAST_ONE_AI_GENERATED" in risk.get("risk_factors", []):
            recommendations.append("CLARIFY: Determine use of AI tools in writing process")
        
        return recommendations
    
    def generate_report(
        self,
        doc1: Union[str, Document],
        doc2: Union[str, Document],
        doc1_id: Optional[str] = None,
        doc2_id: Optional[str] = None,
        output_format: str = "dict"
    ) -> Union[Dict, str]:
        """
        Generate a comprehensive report in various formats.
        
        Args:
            doc1: First document
            doc2: Second document
            doc1_id: Optional ID for first document
            doc2_id: Optional ID for second document
            output_format: Format for output ('dict', 'json', 'text', 'html')
            
        Returns:
            Report in requested format
        """
        analysis = self.analyze(doc1, doc2, doc1_id, doc2_id)
        
        if output_format == "dict":
            return analysis
        elif output_format == "json":
            import json
            return json.dumps(analysis, indent=2, default=str)
        elif output_format == "text":
            return self._format_text_report(analysis)
        elif output_format == "html":
            return self._format_html_report(analysis)
        else:
            raise ValueError(f"Unknown output format: {output_format}")
    
    def _format_text_report(self, analysis: Dict) -> str:
        """Format analysis results as text report."""
        lines = [
            "=" * 70,
            "INTEGRATED PLAGIARISM AND AI DETECTION REPORT",
            "=" * 70,
            ""
        ]
        
        # Metadata
        lines.append(f"Analysis Time: {analysis['analysis_metadata'].get('timestamp')}")
        lines.append(f"Processing Time: {analysis['analysis_metadata'].get('processing_time'):.2f}s")
        lines.append("")
        
        # Plagiarism Results
        if analysis["plagiarism_results"]:
            lines.append("-" * 70)
            lines.append("PLAGIARISM DETECTION RESULTS")
            lines.append("-" * 70)
            plag = analysis["plagiarism_results"]
            lines.append(f"Similarity Score: {plag['similarity_score']:.2%}")
            lines.append(f"Is Plagiarized: {'YES' if plag['is_plagiarized'] else 'NO'}")
            lines.append(f"Number of Matches: {plag['matches']}")
            lines.append("")
        
        # AI Detection Results
        if analysis["ai_results"]:
            lines.append("-" * 70)
            lines.append("AI CONTENT DETECTION RESULTS")
            lines.append("-" * 70)
            ai = analysis["ai_results"]
            lines.append(f"Document 1 AI Score: {ai['document_1']['confidence']:.2%}")
            lines.append(f"Document 1 Is AI: {'YES' if ai['document_1']['is_ai'] else 'NO'}")
            lines.append(f"Document 2 AI Score: {ai['document_2']['confidence']:.2%}")
            lines.append(f"Document 2 Is AI: {'YES' if ai['document_2']['is_ai'] else 'NO'}")
            lines.append("")
        
        # Risk Assessment
        lines.append("-" * 70)
        lines.append("RISK ASSESSMENT")
        lines.append("-" * 70)
        risk = analysis["risk_assessment"]
        lines.append(f"Overall Risk Level: {risk['overall_risk_level']}")
        lines.append(f"Risk Score: {risk['risk_score']:.2%}")
        if risk.get("risk_factors"):
            lines.append("Risk Factors:")
            for factor in risk["risk_factors"]:
                lines.append(f"  - {factor}")
        lines.append("")
        
        # Recommendations
        lines.append("-" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        for rec in analysis["recommendations"]:
            lines.append(f"• {rec}")
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _format_html_report(self, analysis: Dict) -> str:
        """Format analysis results as HTML report."""
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .report { background: #f9f9f9; padding: 20px; border-radius: 8px; }
                .section { margin: 20px 0; padding: 15px; background: white; border-left: 4px solid #007bff; }
                .critical { border-left-color: #dc3545; }
                .high { border-left-color: #fd7e14; }
                .moderate { border-left-color: #ffc107; }
                .low { border-left-color: #28a745; }
                .score { font-size: 24px; font-weight: bold; }
                .metric { display: inline-block; margin-right: 30px; }
            </style>
        </head>
        <body>
        <div class="report">
            <h1>Plagiarism & AI Detection Report</h1>
        """
        
        risk = analysis["risk_assessment"]
        risk_class = risk["overall_risk_level"].lower()
        
        html += f"""
            <div class="section {risk_class}">
                <h2>Risk Assessment</h2>
                <p class="score">{risk['overall_risk_level']}</p>
                <p>Risk Score: {risk['risk_score']:.2%}</p>
        """
        
        if risk.get("risk_factors"):
            html += "<h3>Risk Factors:</h3><ul>"
            for factor in risk["risk_factors"]:
                html += f"<li>{factor}</li>"
            html += "</ul>"
        
        html += "</div>"
        
        # Plagiarism results
        if analysis["plagiarism_results"]:
            plag = analysis["plagiarism_results"]
            html += f"""
            <div class="section">
                <h2>Plagiarism Detection</h2>
                <div class="metric">Similarity: <strong>{plag['similarity_score']:.2%}</strong></div>
                <div class="metric">Matches: <strong>{plag['matches']}</strong></div>
            </div>
            """
        
        # AI results
        if analysis["ai_results"]:
            ai = analysis["ai_results"]
            html += f"""
            <div class="section">
                <h2>AI Content Detection</h2>
                <div class="metric">Doc 1 AI Score: <strong>{ai['document_1']['confidence']:.2%}</strong></div>
                <div class="metric">Doc 2 AI Score: <strong>{ai['document_2']['confidence']:.2%}</strong></div>
            </div>
            """
        
        # Recommendations
        html += "<div class='section'><h2>Recommendations</h2><ul>"
        for rec in analysis["recommendations"]:
            html += f"<li>{rec}</li>"
        html += "</ul></div>"
        
        html += """
        </div>
        </body>
        </html>
        """
        
        return html
