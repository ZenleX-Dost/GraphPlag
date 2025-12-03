import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ingest_folder(directory, category="general", tags=None):
    """
    Ingest all supported documents from a directory into the GraphPlag corpus.
    """
    try:
        from graphplag.corpus.corpus_manager import CorpusManager
    except ImportError:
        logger.error("Could not import CorpusManager. Make sure you are in the project root.")
        return

    # Initialize Corpus Manager
    try:
        postgres_url = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/graphplag")
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
        
        logger.info("Connecting to database...")
        manager = CorpusManager(
            postgres_url=postgres_url,
            milvus_host=milvus_host,
            elasticsearch_host=es_host
        )
        logger.info("✅ Connected to Corpus Manager")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        logger.error("Ensure Docker services are running: docker-compose -f docker-compose-scalable.yml up -d")
        return

    # Find files
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.error(f"Directory not found: {directory}")
        return

    supported_exts = {'.pdf', '.docx', '.txt', '.md', '.markdown'}
    files = [f for f in dir_path.glob('**/*') if f.suffix.lower() in supported_exts]
    
    if not files:
        logger.warning(f"No supported files found in {directory}")
        return

    logger.info(f"Found {len(files)} documents to ingest.")
    
    # Ingest files
    success_count = 0
    fail_count = 0
    
    metadata = {
        'category': category,
        'tags': tags or []
    }

    for file_path in tqdm(files, desc="Ingesting"):
        try:
            # Open file in binary mode
            with open(file_path, 'rb') as f:
                # Create a file-like object with a name attribute for the manager
                # (The manager likely expects a file object with .name or similar)
                # Actually, looking at app_corpus.py, it passes `file.name` (path) to add_document?
                # Let's check app_corpus.py again. 
                # It calls `corpus_manager.add_document(file.name, metadata)` where file is a temp file from Gradio.
                # So passing the path string should work if add_document handles it, 
                # OR passing a file object. 
                # Let's assume it takes a path string based on `file.name`.
                pass

            # Let's try passing the path directly first, as that's most common.
            doc_id = manager.add_document(str(file_path), metadata)
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to ingest {file_path.name}: {e}")
            fail_count += 1

    logger.info("-" * 30)
    logger.info(f"Ingestion Complete")
    logger.info(f"✅ Successfully added: {success_count}")
    logger.info(f"❌ Failed: {fail_count}")
    
    # Show stats
    stats = manager.get_corpus_stats()
    logger.info(f"Total Corpus Size: {stats.get('total_documents', 0)} documents")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk ingest documents into GraphPlag corpus")
    parser.add_argument("directory", help="Directory containing documents to ingest")
    parser.add_argument("--category", default="general", help="Category for imported documents")
    parser.add_argument("--tags", help="Comma-separated tags (e.g. 'research,2024')")
    
    args = parser.parse_args()
    
    tags_list = [t.strip() for t in args.tags.split(',')] if args.tags else []
    
    ingest_folder(args.directory, args.category, tags_list)
