#!/usr/bin/env python
"""
GraphPlag Corpus Management Interface

Upload documents to a permanent library and search against the corpus.
"""

import gradio as gr
import os
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import corpus manager
try:
    from graphplag.corpus.corpus_manager import CorpusManager
    CORPUS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Corpus feature not available: {e}")
    CORPUS_AVAILABLE = False

# Global corpus manager
corpus_manager = None


def init_corpus():
    """Initialize corpus manager with database connections."""
    global corpus_manager
    
    try:
        postgres_url = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/graphplag")
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
        
        corpus_manager = CorpusManager(
            postgres_url=postgres_url,
            milvus_host=milvus_host,
            elasticsearch_host=es_host
        )
        
        return "✅ Corpus Manager initialized successfully!"
        
    except Exception as e:
        return f"❌ Error initializing Corpus Manager: {str(e)}\n\nMake sure full Docker stack is running!"


def add_to_corpus(file, tags_text: str, category: str):
    """Add document to corpus."""
    if not corpus_manager:
        return "❌ Please initialize Corpus Manager first!", None
    
    if file is None:
        return "❌ Please upload a file", None
    
    try:
        # Parse tags
        tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()] if tags_text else []
        
        # Add to corpus
        metadata = {
            'tags': tags,
            'category': category
        }
        
        doc_id = corpus_manager.add_document(file.name, metadata)
        
        # Get stats
        stats = corpus_manager.get_corpus_stats()
        stats_text = f"""
### ✅ Document Added Successfully!

**Document ID:** {doc_id}  
**File:** {os.path.basename(file.name)}  
**Tags:** {', '.join(tags) if tags else 'None'}  
**Category:** {category}

---

### 📊 Corpus Statistics
- **Total Documents:** {stats.get('total_documents', 0)}
- **Total Size:** {stats.get('total_size_bytes', 0) / 1024 / 1024:.2f} MB
- **Embeddings in Milvus:** {stats.get('milvus_embeddings', 0)}
- **Indexed in Elasticsearch:** {stats.get('elasticsearch_indexed', 0)}
"""
        
        return stats_text, get_corpus_table()
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None


def search_corpus_fn(query_file, query_text: str, top_k: int, search_mode: str):
    """Search corpus for similar documents."""
    if not corpus_manager:
        return "❌ Please initialize Corpus Manager first!"
    
    # Get query text
    if query_file is not None:
        try:
            from graphplag.utils.file_parser import FileParser
            parser = FileParser()
            query = parser.parse_file(query_file.name)
        except Exception as e:
            return f"❌ Error parsing file: {str(e)}"
    elif query_text:
        query = query_text
    else:
        return "❌ Please provide either a file or text query"
    
    try:
        # Search corpus
        results = corpus_manager.search_corpus(query, top_k, search_mode)
        
        if not results:
            return "### No matching documents found"
        
        # Format results
        output = f"### 🔍 Found {len(results)} Matching Documents\n\n"
        
        for i, match in enumerate(results, 1):
            output += f"""
#### {i}. {match.get('file_name', 'Unknown')}
- **Similarity Score:** {match['score']:.2%}
- **Category:** {match.get('category', 'general')}
- **Tags:** {', '.join(match.get('tags', [])) if match.get('tags') else 'None'}
- **Preview:** {match.get('content_preview', 'N/A')[:200]}...

---
"""
        
        return output
        
    except Exception as e:
        return f"❌ Error searching corpus: {str(e)}"


def get_corpus_table():
    """Get all corpus documents as a table."""
    if not corpus_manager:
        return None
    
    try:
        docs = corpus_manager.get_all_documents(limit=50)
        
        if not docs:
            return None
        
        # Format as list of lists for Gradio DataFrame
        data = []
        for doc in docs:
            data.append([
                doc['doc_id'],
                doc['file_name'],
                doc.get('category', 'general'),
                ', '.join(doc.get('tags', [])) if doc.get('tags') else '',
                f"{doc['file_size'] / 1024:.1f} KB" if doc.get('file_size') else 'N/A',
                str(doc.get('added_at', ''))[:19] if doc.get('added_at') else ''
            ])
        
        return data
        
    except Exception as e:
        print(f"Error getting corpus table: {e}")
        return None


def delete_doc(doc_id: int):
    """Delete document from corpus."""
    if not corpus_manager:
        return "❌ Please initialize Corpus Manager first!", None
    
    try:
        success = corpus_manager.delete_document(doc_id)
        
        if success:
            return f"✅ Deleted document ID: {doc_id}", get_corpus_table()
        else:
            return f"❌ Failed to delete document ID: {doc_id}", None
            
    except Exception as e:
        return f"❌ Error: {str(e)}", None


def get_stats():
    """Get corpus statistics."""
    if not corpus_manager:
        return "❌ Please initialize Corpus Manager first!"
    
    try:
        stats = corpus_manager.get_corpus_stats()
        
        output = f"""
# 📊 Corpus Statistics

## Overview
- **Total Documents:** {stats.get('total_documents', 0)}
- **Total Size:** {stats.get('total_size_bytes', 0) / 1024 / 1024:.2f} MB
- **Average Size:** {stats.get('avg_size_bytes', 0) / 1024:.2f} KB

## Storage Status
- **PostgreSQL Documents:** {stats.get('total_documents', 0)}
- **Milvus Embeddings:** {stats.get('milvus_embeddings', 0)}
- **Elasticsearch Indexed:** {stats.get('elasticsearch_indexed', 0)}

## Categories
"""
        
        categories = stats.get('categories', {})
        if categories:
            for cat, count in categories.items():
                output += f"- **{cat}:** {count} documents\n"
        else:
            output += "*No categories*\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="GraphPlag Corpus Manager", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 📚 GraphPlag - Document Corpus Manager
    
    Build a permanent document library and search for plagiarism against your entire corpus.
    
    **⚠️ Requirements:** Full Docker stack must be running (PostgreSQL + Milvus + Elasticsearch)
    """)
    
    with gr.Row():
        init_btn = gr.Button("🔌 Initialize Corpus Manager", variant="primary")
        init_output = gr.Textbox(label="Initialization Status", interactive=False)
    
    init_btn.click(fn=init_corpus, outputs=init_output)
    
    with gr.Tabs():
        
        # Tab 1: Add Documents
        with gr.Tab("📁 Add to Corpus"):
            gr.Markdown("### Upload documents to build your corpus")
            
            file_input = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".docx", ".txt", ".md", ".markdown"]
            )
            
            with gr.Row():
                tags_input = gr.Textbox(
                    label="Tags (comma-separated)",
                    placeholder="e.g., student, essay, 2024"
                )
                category_input = gr.Dropdown(
                    choices=["general", "academic", "research", "assignment", "thesis"],
                    value="general",
                    label="Category"
                )
            
            add_btn = gr.Button("➕ Add to Corpus", variant="primary")
            add_output = gr.Markdown()
            
            gr.Markdown("### Current Corpus")
            corpus_table = gr.Dataframe(
                headers=["ID", "File Name", "Category", "Tags", "Size", "Added"],
                label="Documents in Corpus"
            )
            
            add_btn.click(
                fn=add_to_corpus,
                inputs=[file_input, tags_input, category_input],
                outputs=[add_output, corpus_table]
            )
        
        # Tab 2: Search Corpus
        with gr.Tab("🔍 Search Corpus"):
            gr.Markdown("### Find similar documents in your corpus")
            
            with gr.Row():
                with gr.Column():
                    search_file = gr.File(
                        label="Upload Query Document",
                        file_types=[".pdf", ".docx", ".txt", ".md"]
                    )
                    gr.Markdown("**OR**")
                    search_text = gr.Textbox(
                        label="Enter Query Text",
                        placeholder="Paste text to search for...",
                        lines=5
                    )
                
                with gr.Column():
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of Results"
                    )
                    search_mode = gr.Radio(
                        choices=["hybrid", "vector", "fulltext"],
                        value="hybrid",
                        label="Search Mode"
                    )
            
            search_btn = gr.Button("🔎 Search", variant="primary", size="lg")
            search_output = gr.Markdown()
            
            search_btn.click(
                fn=search_corpus_fn,
                inputs=[search_file, search_text, top_k, search_mode],
                outputs=search_output
            )
        
        # Tab 3: Manage Corpus
        with gr.Tab("⚙️ Manage Corpus"):
            gr.Markdown("### View and manage your document corpus")
            
            with gr.Row():
                stats_btn = gr.Button("📊 Get Statistics")
                refresh_btn = gr.Button("🔄 Refresh Table")
            
            stats_output = gr.Markdown()
            
            manage_table = gr.Dataframe(
                headers=["ID", "File Name", "Category", "Tags", "Size", "Added"],
                label="All Documents"
            )
            
            with gr.Row():
                delete_id = gr.Number(label="Document ID to Delete", precision=0)
                delete_btn = gr.Button("🗑️ Delete", variant="stop")
            
            delete_output = gr.Textbox(label="Delete Status")
            
            stats_btn.click(fn=get_stats, outputs=stats_output)
            refresh_btn.click(fn=get_corpus_table, outputs=manage_table)
            delete_btn.click(
                fn=delete_doc,
                inputs=delete_id,
                outputs=[delete_output, manage_table]
            )
    
    gr.Markdown("""
    ---
    **GraphPlag Corpus Manager** | Powered by PostgreSQL + Mill + Elasticsearch
    """)


if __name__ == "__main__":
    if not CORPUS_AVAILABLE:
        print("ERROR: Corpus feature dependencies not available!")
        print("Please ensure graphplag.corpus package is installed.")
        exit(1)
    
    print("🚀 Starting GraphPlag Corpus Manager...")
    print("⚠️  Make sure full Docker stack is running!")
    print("   Run: docker-compose -f docker-compose-scalable.yml up -d")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from main app
        share=False,
        show_error=True
    )
