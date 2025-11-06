"""
Large file optimization utilities.
Provides chunking, streaming, and memory-efficient processing for large documents.
"""
from typing import List, Iterator, Optional, Callable
from dataclasses import dataclass
import math
from pathlib import Path
import time


@dataclass
class ChunkInfo:
    """Information about a document chunk."""
    chunk_id: int
    start_sentence: int
    end_sentence: int
    text: str
    num_sentences: int
    size_bytes: int


class DocumentChunker:
    """
    Chunks large documents into manageable pieces for processing.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1000,  # sentences per chunk
        overlap: int = 50,  # sentences overlap between chunks
        max_memory_mb: int = 100
    ):
        """
        Initialize document chunker.
        
        Args:
            max_chunk_size: Maximum sentences per chunk
            overlap: Number of overlapping sentences between chunks
            max_memory_mb: Maximum memory per chunk in MB
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
    
    def chunk_sentences(
        self,
        sentences: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Iterator[ChunkInfo]:
        """
        Chunk sentences into manageable pieces.
        
        Args:
            sentences: List of sentence strings
            progress_callback: Optional callback(current, total)
            
        Yields:
            ChunkInfo objects
        """
        total_sentences = len(sentences)
        num_chunks = math.ceil(total_sentences / (self.max_chunk_size - self.overlap))
        
        start_idx = 0
        chunk_id = 0
        
        while start_idx < total_sentences:
            # Calculate end index
            end_idx = min(start_idx + self.max_chunk_size, total_sentences)
            
            # Get chunk sentences
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)
            chunk_size = len(chunk_text.encode('utf-8'))
            
            # Create chunk info
            chunk = ChunkInfo(
                chunk_id=chunk_id,
                start_sentence=start_idx,
                end_sentence=end_idx,
                text=chunk_text,
                num_sentences=len(chunk_sentences),
                size_bytes=chunk_size
            )
            
            yield chunk
            
            # Progress callback
            if progress_callback:
                progress_callback(chunk_id + 1, num_chunks)
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.overlap if end_idx < total_sentences else end_idx
            chunk_id += 1
    
    def estimate_chunks(self, num_sentences: int) -> int:
        """Estimate number of chunks for a document."""
        return math.ceil(num_sentences / (self.max_chunk_size - self.overlap))


class StreamingFileParser:
    """
    Memory-efficient streaming parser for large files.
    """
    
    def __init__(self, chunk_size_kb: int = 64):
        """
        Initialize streaming parser.
        
        Args:
            chunk_size_kb: Size of read chunks in KB
        """
        self.chunk_size = chunk_size_kb * 1024
    
    def parse_large_text_file(
        self,
        file_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Iterator[str]:
        """
        Parse large text file in streaming fashion.
        
        Args:
            file_path: Path to text file
            progress_callback: Optional callback(bytes_read, total_bytes)
            
        Yields:
            Lines from the file
        """
        path = Path(file_path)
        total_size = path.stat().st_size
        bytes_read = 0
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            buffer = ""
            
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                
                buffer += chunk
                bytes_read += len(chunk.encode('utf-8'))
                
                # Split into lines
                lines = buffer.split('\n')
                
                # Keep last incomplete line in buffer
                buffer = lines[-1]
                
                # Yield complete lines
                for line in lines[:-1]:
                    if line.strip():
                        yield line
                
                # Progress callback
                if progress_callback:
                    progress_callback(bytes_read, total_size)
            
            # Yield remaining buffer
            if buffer.strip():
                yield buffer


class ProgressTracker:
    """
    Track and display progress for long-running operations.
    """
    
    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            description: Description of the operation
        """
        self.total_items = total_items
        self.current_item = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current_item += increment
        current_time = time.time()
        
        # Update every 0.5 seconds
        if current_time - self.last_update > 0.5 or self.current_item == self.total_items:
            self._display_progress()
            self.last_update = current_time
    
    def _display_progress(self):
        """Display progress bar."""
        percent = (self.current_item / self.total_items) * 100
        elapsed = time.time() - self.start_time
        
        if self.current_item > 0:
            eta = (elapsed / self.current_item) * (self.total_items - self.current_item)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        bar_length = 40
        filled = int(bar_length * self.current_item / self.total_items)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r{self.description}: [{bar}] {percent:.1f}% ({self.current_item}/{self.total_items}) {eta_str}", end='', flush=True)
        
        if self.current_item == self.total_items:
            print()  # New line when complete
    
    def complete(self):
        """Mark as complete."""
        self.current_item = self.total_items
        self._display_progress()


class MemoryMonitor:
    """
    Monitor memory usage during processing.
    """
    
    def __init__(self, max_memory_mb: int = 1000):
        """
        Initialize memory monitor.
        
        Args:
            max_memory_mb: Maximum allowed memory in MB
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory = 0
    
    def check_memory(self) -> bool:
        """
        Check if memory limit would be exceeded.
        
        Returns:
            True if within limits, False if exceeded
        """
        try:
            import psutil
            process = psutil.Process()
            self.current_memory = process.memory_info().rss
            return self.current_memory < self.max_memory_bytes
        except ImportError:
            # psutil not available, assume OK
            return True
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.current_memory / (1024 * 1024)


class BatchProcessor:
    """
    Process items in batches to control memory usage.
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items per batch
        """
        self.batch_size = batch_size
    
    def process_in_batches(
        self,
        items: List,
        processor_func: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            processor_func: Function to process each batch
            progress_callback: Optional progress callback
            
        Returns:
            List of processed results
        """
        results = []
        total_items = len(items)
        
        for i in range(0, total_items, self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = processor_func(batch)
            results.extend(batch_results)
            
            if progress_callback:
                progress_callback(min(i + self.batch_size, total_items), total_items)
        
        return results
