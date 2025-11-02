#!/usr/bin/env python
"""
GraphPlag Command Line Interface
Simple CLI for plagiarism detection
"""

import argparse
import sys
from pathlib import Path
import json
from typing import List

from graphplag import PlagiarismDetector
from graphplag.detection.report_generator import ReportGenerator


def read_file(file_path: str) -> str:
    """Read content from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        sys.exit(1)


def compare_documents(args):
    """Compare two documents"""
    # Read documents
    if args.file1:
        doc1 = read_file(args.file1)
        print(f"📄 Loaded Document 1: {args.file1}")
    else:
        doc1 = args.text1
    
    if args.file2:
        doc2 = read_file(args.file2)
        print(f"📄 Loaded Document 2: {args.file2}")
    else:
        doc2 = args.text2
    
    if not doc1 or not doc2:
        print("❌ Error: Both documents must be provided")
        sys.exit(1)
    
    # Initialize detector
    print(f"\n🔧 Initializing detector...")
    print(f"   Method: {args.method}")
    print(f"   Threshold: {args.threshold}")
    print(f"   Language: {args.language}")
    
    detector = PlagiarismDetector(
        method=args.method,
        threshold=args.threshold,
        language=args.language
    )
    
    # Detect plagiarism
    print("\n🔍 Analyzing documents...")
    report = detector.detect_plagiarism(doc1, doc2)
    
    # Display results
    print("\n" + "="*60)
    print("📊 PLAGIARISM DETECTION RESULTS")
    print("="*60)
    print(f"\n📈 Similarity Score: {report.similarity_score:.2%}")
    print(f"🎯 Threshold: {args.threshold:.2%}")
    print(f"⚠️  Plagiarism Detected: {'YES ✓' if report.is_plagiarism else 'NO ✗'}")
    print(f"🔬 Method Used: {report.method.upper()}")
    
    # Interpretation
    print("\n📝 Interpretation:")
    if report.similarity_score >= 0.9:
        print("   🔴 Very High Similarity - Strong evidence of plagiarism")
    elif report.similarity_score >= 0.7:
        print("   🟠 High Similarity - Likely plagiarism detected")
    elif report.similarity_score >= 0.5:
        print("   🟡 Moderate Similarity - Possible paraphrasing")
    else:
        print("   🟢 Low Similarity - Documents appear original")
    
    print("\n" + "="*60)
    
    # Save report if requested
    if args.output:
        report_gen = ReportGenerator()
        output_path = Path(args.output)
        
        if output_path.suffix == '.json':
            # Save as JSON
            report_data = {
                'similarity_score': report.similarity_score,
                'is_plagiarism': report.is_plagiarism,
                'method': report.method,
                'threshold': args.threshold,
                'doc1_length': len(doc1),
                'doc2_length': len(doc2)
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n💾 Report saved to: {output_path}")
        
        elif output_path.suffix == '.html':
            # Save as HTML
            report_gen.generate_html(report, str(output_path))
            print(f"\n💾 HTML report saved to: {output_path}")
        
        else:
            # Save as text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"GraphPlag Plagiarism Detection Report\n")
                f.write("="*60 + "\n\n")
                f.write(f"Similarity Score: {report.similarity_score:.2%}\n")
                f.write(f"Plagiarism Detected: {'YES' if report.is_plagiarism else 'NO'}\n")
                f.write(f"Method: {report.method}\n")
                f.write(f"Threshold: {args.threshold:.2%}\n")
            print(f"\n💾 Report saved to: {output_path}")


def batch_compare(args):
    """Compare multiple documents"""
    # Read documents from directory or file list
    documents = []
    doc_names = []
    
    if args.directory:
        # Read all text files from directory
        dir_path = Path(args.directory)
        if not dir_path.is_dir():
            print(f"❌ Error: {args.directory} is not a directory")
            sys.exit(1)
        
        for file_path in sorted(dir_path.glob('*.txt')):
            documents.append(read_file(str(file_path)))
            doc_names.append(file_path.name)
            print(f"📄 Loaded: {file_path.name}")
    
    elif args.files:
        # Read specific files
        for file_path in args.files:
            documents.append(read_file(file_path))
            doc_names.append(Path(file_path).name)
            print(f"📄 Loaded: {Path(file_path).name}")
    
    else:
        print("❌ Error: Provide either --directory or --files")
        sys.exit(1)
    
    if len(documents) < 2:
        print("❌ Error: Need at least 2 documents for comparison")
        sys.exit(1)
    
    # Initialize detector
    print(f"\n🔧 Initializing detector...")
    detector = PlagiarismDetector(
        method=args.method,
        threshold=args.threshold,
        language=args.language
    )
    
    # Compare all pairs
    print(f"\n🔍 Comparing {len(documents)} documents...")
    suspicious_pairs = []
    
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            report = detector.detect_plagiarism(documents[i], documents[j])
            if report.similarity_score >= args.threshold:
                suspicious_pairs.append((i, j, report.similarity_score))
    
    # Display results
    print("\n" + "="*60)
    print("📊 BATCH COMPARISON RESULTS")
    print("="*60)
    print(f"\n📚 Total Documents: {len(documents)}")
    print(f"🔍 Comparisons Made: {len(documents) * (len(documents) - 1) // 2}")
    print(f"⚠️  Suspicious Pairs: {len(suspicious_pairs)}")
    print(f"🎯 Threshold: {args.threshold:.2%}")
    
    if suspicious_pairs:
        print(f"\n🚨 Suspicious Pairs (Similarity ≥ {args.threshold:.0%}):")
        print("-" * 60)
        for i, j, score in sorted(suspicious_pairs, key=lambda x: x[2], reverse=True):
            print(f"   {doc_names[i]} ↔ {doc_names[j]}")
            print(f"   Similarity: {score:.2%}")
            print()
    else:
        print("\n✅ No suspicious pairs found!")
    
    print("="*60)
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        report_data = {
            'total_documents': len(documents),
            'comparisons': len(documents) * (len(documents) - 1) // 2,
            'threshold': args.threshold,
            'suspicious_pairs': [
                {
                    'doc1': doc_names[i],
                    'doc2': doc_names[j],
                    'similarity': score
                }
                for i, j, score in suspicious_pairs
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Report saved to: {output_path}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='GraphPlag - Graph-based Plagiarism Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two text strings
  python cli.py compare --text1 "First document" --text2 "Second document"
  
  # Compare two files
  python cli.py compare --file1 doc1.txt --file2 doc2.txt
  
  # Compare with custom settings
  python cli.py compare --file1 doc1.txt --file2 doc2.txt --method ensemble --threshold 0.8
  
  # Save report
  python cli.py compare --file1 doc1.txt --file2 doc2.txt --output report.html
  
  # Batch compare all files in directory
  python cli.py batch --directory ./documents
  
  # Batch compare specific files
  python cli.py batch --files doc1.txt doc2.txt doc3.txt --output results.json

For more information, visit: https://github.com/ZenleX-Dost/GraphPlag
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two documents')
    compare_parser.add_argument('--file1', help='Path to first document file')
    compare_parser.add_argument('--file2', help='Path to second document file')
    compare_parser.add_argument('--text1', help='First document as text')
    compare_parser.add_argument('--text2', help='Second document as text')
    compare_parser.add_argument('--method', choices=['kernel', 'gnn', 'ensemble'], 
                               default='kernel', help='Detection method (default: kernel)')
    compare_parser.add_argument('--threshold', type=float, default=0.7, 
                               help='Plagiarism threshold (default: 0.7)')
    compare_parser.add_argument('--language', choices=['en', 'es', 'fr', 'de'], 
                               default='en', help='Document language (default: en)')
    compare_parser.add_argument('--output', help='Save report to file (.txt, .json, .html)')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Compare multiple documents')
    batch_parser.add_argument('--directory', help='Directory containing documents')
    batch_parser.add_argument('--files', nargs='+', help='List of files to compare')
    batch_parser.add_argument('--method', choices=['kernel', 'gnn', 'ensemble'], 
                             default='kernel', help='Detection method (default: kernel)')
    batch_parser.add_argument('--threshold', type=float, default=0.7, 
                             help='Plagiarism threshold (default: 0.7)')
    batch_parser.add_argument('--language', choices=['en', 'es', 'fr', 'de'], 
                             default='en', help='Document language (default: en)')
    batch_parser.add_argument('--output', help='Save report to JSON file')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Print banner
    print("\n" + "="*60)
    print("🔍 GraphPlag - Plagiarism Detection System")
    print("="*60 + "\n")
    
    # Execute command
    if args.command == 'compare':
        compare_documents(args)
    elif args.command == 'batch':
        batch_compare(args)


if __name__ == '__main__':
    main()
