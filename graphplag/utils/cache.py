"""
Embedding cache for optimizing repeated computations.
Stores sentence embeddings on disk to avoid recomputation.
"""
import os
import pickle
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
from datetime import datetime, timedelta


class EmbeddingCache:
    """
    Disk-based cache for sentence embeddings.
    Uses content hashing to identify cached embeddings.
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache/embeddings",
        max_age_days: int = 30,
        max_size_mb: int = 500
    ):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_days: Maximum age of cached items in days
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / "metadata.json"
        self.max_age = timedelta(days=max_age_days)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {"items": {}, "total_size": 0}
        return {"items": {}, "total_size": 0}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _compute_hash(self, text: str, model_name: str) -> str:
        """
        Compute hash for text and model combination.
        
        Args:
            text: Input text
            model_name: Name of embedding model
            
        Returns:
            Hash string
        """
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path for cache file."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(
        self,
        text: str,
        model_name: str
    ) -> Optional[np.ndarray]:
        """
        Get cached embedding.
        
        Args:
            text: Input text
            model_name: Name of embedding model
            
        Returns:
            Cached embedding array or None if not found
        """
        cache_key = self._compute_hash(text, model_name)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        # Check if cache item is expired
        if cache_key in self.metadata["items"]:
            item_data = self.metadata["items"][cache_key]
            created_at = datetime.fromisoformat(item_data["created_at"])
            
            if datetime.now() - created_at > self.max_age:
                # Cache expired, remove it
                self.remove(cache_key)
                return None
        
        try:
            with open(cache_path, 'rb') as f:
                embedding = pickle.load(f)
            
            # Update access time
            if cache_key in self.metadata["items"]:
                self.metadata["items"][cache_key]["last_accessed"] = datetime.now().isoformat()
                self.metadata["items"][cache_key]["access_count"] += 1
                self._save_metadata()
            
            return embedding
        except Exception:
            # If loading fails, remove corrupted cache
            self.remove(cache_key)
            return None
    
    def put(
        self,
        text: str,
        model_name: str,
        embedding: np.ndarray
    ):
        """
        Store embedding in cache.
        
        Args:
            text: Input text
            model_name: Name of embedding model
            embedding: Embedding array to cache
        """
        cache_key = self._compute_hash(text, model_name)
        cache_path = self._get_cache_path(cache_key)
        
        # Check if we need to clean up cache
        if self.metadata["total_size"] > self.max_size_bytes:
            self._cleanup_old_items()
        
        # Store embedding
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
            
            # Update metadata
            file_size = cache_path.stat().st_size
            
            if cache_key in self.metadata["items"]:
                # Update existing item
                old_size = self.metadata["items"][cache_key]["size"]
                self.metadata["total_size"] -= old_size
            
            self.metadata["items"][cache_key] = {
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "size": file_size,
                "access_count": 0,
                "model_name": model_name,
                "text_length": len(text)
            }
            
            self.metadata["total_size"] += file_size
            self._save_metadata()
            
        except Exception as e:
            print(f"Warning: Failed to cache embedding: {e}")
    
    def remove(self, cache_key: str):
        """Remove item from cache."""
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            cache_path.unlink()
        
        if cache_key in self.metadata["items"]:
            self.metadata["total_size"] -= self.metadata["items"][cache_key]["size"]
            del self.metadata["items"][cache_key]
            self._save_metadata()
    
    def _cleanup_old_items(self):
        """Remove least recently used items to free space."""
        # Sort items by last access time
        items = [
            (key, data)
            for key, data in self.metadata["items"].items()
        ]
        
        items.sort(key=lambda x: x[1]["last_accessed"])
        
        # Remove oldest items until we're under 80% of max size
        target_size = self.max_size_bytes * 0.8
        
        for cache_key, _ in items:
            if self.metadata["total_size"] <= target_size:
                break
            self.remove(cache_key)
    
    def clear(self):
        """Clear all cached items."""
        for cache_key in list(self.metadata["items"].keys()):
            self.remove(cache_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_items": len(self.metadata["items"]),
            "total_size_mb": self.metadata["total_size"] / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
            "items": [
                {
                    "model": data["model_name"],
                    "size_kb": data["size"] / 1024,
                    "created": data["created_at"],
                    "access_count": data["access_count"],
                    "text_length": data["text_length"]
                }
                for data in list(self.metadata["items"].values())[:10]  # Show first 10
            ]
        }


class SentenceSplitterCache:
    """
    Cache for sentence splitting results.
    Useful for documents that are parsed multiple times.
    """
    
    def __init__(self, cache_dir: str = ".cache/sentences"):
        """Initialize sentence cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_hash(self, text: str) -> str:
        """Compute hash for text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[List[str]]:
        """Get cached sentences."""
        cache_key = self._compute_hash(text)
        cache_path = self.cache_dir / f"{cache_key}.json"
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def put(self, text: str, sentences: List[str]):
        """Store sentences in cache."""
        cache_key = self._compute_hash(text)
        cache_path = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(sentences, f, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to cache sentences: {e}")
    
    def clear(self):
        """Clear all cached sentences."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
