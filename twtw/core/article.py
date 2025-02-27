"""
Article data model for TWTW.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    content: Optional[str] = None
    processed_content: Optional[str] = None  # Content processed by OpenAI
    subcategory: Optional[str] = None
    entities: Optional[str] = None  # Key people, companies, technologies
    features: Optional[Dict] = field(default_factory=dict)
    category: str = "Unclassified" 