"""
Article data model for TWTW.
"""
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class Article:
    """
    Represents an article with its metadata and content.
    """
    title: str
    url: str
    image_url: Optional[str]
    author: Optional[str]
    source: Optional[str]
    summary: str
    read_more_url: str
    published: Optional[str] = None
    domain: Optional[str] = None
    reading_time: Optional[int] = None
    category_score: Optional[Dict[str, float]] = None
    full_text: Optional[str] = None
    content_features: Optional[Dict] = None 