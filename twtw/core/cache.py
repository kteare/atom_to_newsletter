"""
Cache management for TWTW.
"""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

# Cache configuration
CACHE_DIR = Path("cache")
CACHE_DB = CACHE_DIR / "article_cache.db"
CACHE_DURATION = timedelta(days=7)  # Cache articles for 7 days

class CacheManager:
    """
    Manages caching of article content and features to avoid redundant downloads.
    """
    def __init__(self):
        self._init_cache_dir()
        self._init_db()

    def _init_cache_dir(self):
        """Initialize the cache directory."""
        CACHE_DIR.mkdir(exist_ok=True)

    def _init_db(self):
        """Initialize the SQLite database for caching."""
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    url TEXT PRIMARY KEY,
                    content TEXT,
                    features TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def get(self, url: str) -> Optional[Dict]:
        """
        Get cached article content if it exists and is fresh.
        
        Args:
            url: The URL of the article to retrieve from cache
            
        Returns:
            Dict containing content and features if found and fresh, None otherwise
        """
        with sqlite3.connect(CACHE_DB) as conn:
            cursor = conn.execute(
                """
                SELECT content, features, timestamp 
                FROM articles 
                WHERE url = ?
                """, 
                (url,)
            )
            result = cursor.fetchone()
            
            if result:
                content, features, timestamp = result
                cache_time = datetime.fromisoformat(timestamp)
                if datetime.now() - cache_time < CACHE_DURATION:
                    return {
                        'content': content,
                        'features': json.loads(features)
                    }
                else:
                    # Clean up expired cache entry
                    conn.execute("DELETE FROM articles WHERE url = ?", (url,))
                    conn.commit()
            return None

    def set(self, url: str, content: str, features: Dict):
        """
        Cache article content and features.
        
        Args:
            url: The URL of the article to cache
            content: The article content to cache
            features: The article features to cache
        """
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO articles (url, content, features, timestamp)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (url, content, json.dumps(features))
            ) 