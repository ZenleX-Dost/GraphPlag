"""Detection module initialization."""

from graphplag.detection.detector import PlagiarismDetector
from graphplag.detection.report_generator import ReportGenerator

__all__ = [
    "PlagiarismDetector",
    "ReportGenerator",
]
