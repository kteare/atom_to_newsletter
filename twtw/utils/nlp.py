"""
NLP utilities for TWTW.
"""
import re
from typing import Dict, List, Optional, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import logging

logger = logging.getLogger(__name__)

# Define category patterns
TECH_PATTERNS = [
    r'\b(?:technology|tech|software|hardware|digital|app|application|platform|device)\b',
    r'\b(?:computer|computing|code|programming|developer|development)\b',
    r'\b(?:internet|web|online|website|browser|cloud|server|database)\b'
]

AI_PATTERNS = [
    r'\b(?:artificial intelligence|AI|machine learning|ML|deep learning|neural network)\b',
    r'\b(?:NLP|natural language processing|computer vision|CV|robotics)\b',
    r'\b(?:algorithm|model|training|inference|prediction|classification)\b',
    r'\b(?:GPT|LLM|large language model|transformer|diffusion|generative)\b'
]

VC_PATTERNS = [
    r'\b(?:venture capital|VC|investor|investment|funding|fund|financing)\b',
    r'\b(?:startup|start-up|seed|series [A-Z]|valuation|exit|acquisition)\b',
    r'\b(?:million|billion|round|raise|raised|backed|portfolio|angel)\b'
]

BUSINESS_PATTERNS = [
    r'\b(?:business|company|corporation|enterprise|firm|industry|market)\b',
    r'\b(?:CEO|executive|leadership|management|strategy|growth|revenue)\b',
    r'\b(?:profit|earnings|quarterly|fiscal|financial|stock|shares)\b'
]

POLICY_PATTERNS = [
    r'\b(?:policy|regulation|regulatory|law|legal|legislation|compliance)\b',
    r'\b(?:government|governance|privacy|security|data protection|GDPR)\b',
    r'\b(?:antitrust|monopoly|competition|ruling|court|lawsuit|settlement)\b'
]

class TextAnalyzer:
    """
    Analyzes text content using NLP techniques.
    """
    def __init__(self, language: str = 'english'):
        """
        Initialize the TextAnalyzer.
        
        Args:
            language: Language for NLP processing
        """
        self.language = language
        self._ensure_nltk_data()
        
        # Define category patterns
        self.category_patterns = {
            "Technology": TECH_PATTERNS,
            "AI & Machine Learning": AI_PATTERNS,
            "Venture Capital": VC_PATTERNS,
            "Business & Markets": BUSINESS_PATTERNS,
            "Policy & Regulation": POLICY_PATTERNS
        }
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
    
    def extract_features(self, text: str) -> Dict:
        """
        Extract features from text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict of extracted features
        """
        if not text:
            return {}
        
        # Clean up the text
        clean_text = re.sub(r'\s+', ' ', text).strip()
        
        # Extract basic features
        features = {
            'length': len(clean_text),
            'has_code': bool(re.search(r'```|\{|\}|function|class|def|return|import', clean_text)),
            'has_quotes': clean_text.count('"') + clean_text.count('"') + clean_text.count('"'),
            'sentence_count': len(sent_tokenize(clean_text)),
            'word_count': len(clean_text.split()),
            'reading_time': len(clean_text.split()) // 200,  # Assuming 200 words per minute
            'technical_terms': sum(1 for term in [
                'algorithm', 'database', 'framework', 'api', 'cloud', 
                'infrastructure', 'protocol', 'architecture'
            ] if term in clean_text.lower())
        }
        
        # Add sentiment analysis
        blob = TextBlob(clean_text)
        features['sentiment'] = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
        # Add category pattern matches
        features['category_matches'] = self._match_category_patterns(clean_text)
        
        return features
    
    def _match_category_patterns(self, text: str) -> Dict[str, int]:
        """
        Match text against category patterns.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict of category names and match counts
        """
        matches = {}
        text_lower = text.lower()
        
        for category, patterns in self.category_patterns.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, text, re.IGNORECASE))
            matches[category] = count
            
        return matches
    
    def summarize(self, text: str, sentences_count: int = 3) -> str:
        """
        Generate a summary of the text.
        
        Args:
            text: The text to summarize
            sentences_count: Number of sentences in the summary
            
        Returns:
            Summary text
        """
        if not text or len(text) < 100:
            return text
        
        try:
            # Parse the text
            parser = PlaintextParser.from_string(text, Tokenizer(self.language))
            
            # Create the summarizer
            stemmer = Stemmer(self.language)
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words(self.language)
            
            # Generate the summary
            summary_sentences = summarizer(parser.document, sentences_count)
            summary = ' '.join(str(sentence) for sentence in summary_sentences)
            
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback to first few sentences
            sentences = sent_tokenize(text)
            return ' '.join(sentences[:min(sentences_count, len(sentences))])
    
    def classify_text(self, text: str, title: str = "", source: str = "") -> Tuple[str, Dict[str, float]]:
        """
        Classify text into predefined categories.
        
        Args:
            text: The text to classify
            title: The article title (optional)
            source: The article source (optional)
            
        Returns:
            Tuple of (category_name, scores_dict)
        """
        if not text and not title:
            return "Unclassified", {}
        
        # Combine title and text for better classification
        combined_text = f"{title} {text}" if title else text
        
        # Extract features
        features = self.extract_features(combined_text)
        
        # Calculate scores for each category
        scores = {}
        
        # Score based on pattern matches
        category_matches = features.get('category_matches', {})
        total_matches = sum(category_matches.values()) or 1  # Avoid division by zero
        
        for category, match_count in category_matches.items():
            # Calculate normalized score (0-1)
            scores[category] = min(match_count / total_matches * 2, 1.0)
        
        # Adjust scores based on source
        if source:
            source_lower = source.lower()
            
            # VC and investment sources
            if any(s in source_lower for s in ['techcrunch', 'crunchbase', 'venturebeat', 'pitchbook']):
                scores['Venture Capital'] = scores.get('Venture Capital', 0) + 0.2
                
            # Business sources
            if any(s in source_lower for s in ['wsj', 'bloomberg', 'ft.com', 'forbes', 'business']):
                scores['Business & Markets'] = scores.get('Business & Markets', 0) + 0.2
                
            # Tech sources
            if any(s in source_lower for s in ['wired', 'techcrunch', 'verge', 'ars', 'tech']):
                scores['Technology'] = scores.get('Technology', 0) + 0.2
                
            # AI sources
            if any(s in source_lower for s in ['ai', 'machine', 'learning', 'deepmind', 'openai']):
                scores['AI & Machine Learning'] = scores.get('AI & Machine Learning', 0) + 0.2
        
        # Cap scores at 1.0
        scores = {k: min(v, 1.0) for k, v in scores.items()}
        
        # Find the category with the highest score
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])
            if best_category[1] > 0.3:  # Threshold for classification
                return best_category[0], scores
        
        return "Unclassified", scores
    
    def get_subcategory(self, text: str, category: str) -> str:
        """
        Determine a subcategory within a main category.
        
        Args:
            text: The text to analyze
            category: The main category
            
        Returns:
            Subcategory name
        """
        if not text or category == "Unclassified":
            return ""
            
        text_lower = text.lower()
        
        # Technology subcategories
        if category == "Technology":
            if re.search(r'\b(?:mobile|phone|android|ios|iphone|app)\b', text_lower):
                return "Mobile"
            if re.search(r'\b(?:web|browser|internet|website|online)\b', text_lower):
                return "Web"
            if re.search(r'\b(?:hardware|device|gadget|computer|laptop|pc)\b', text_lower):
                return "Hardware"
            if re.search(r'\b(?:software|application|program|code|development)\b', text_lower):
                return "Software"
                
        # AI subcategories
        elif category == "AI & Machine Learning":
            if re.search(r'\b(?:llm|gpt|language model|chat|claude|gemini)\b', text_lower):
                return "Large Language Models"
            if re.search(r'\b(?:vision|image|video|diffusion|stable|midjourney|dall-e)\b', text_lower):
                return "Computer Vision & Generative AI"
            if re.search(r'\b(?:robot|robotics|automation|autonomous)\b', text_lower):
                return "Robotics & Automation"
                
        # VC subcategories
        elif category == "Venture Capital":
            if re.search(r'\b(?:seed|early|series a)\b', text_lower):
                return "Early Stage"
            if re.search(r'\b(?:series [b-z]|growth|late|ipo|exit|acquisition)\b', text_lower):
                return "Growth & Late Stage"
            if re.search(r'\b(?:fund|raise|billion|million)\b', text_lower):
                return "Fundraising"
                
        # Business subcategories
        elif category == "Business & Markets":
            if re.search(r'\b(?:earnings|revenue|profit|quarterly|fiscal)\b', text_lower):
                return "Earnings & Finance"
            if re.search(r'\b(?:strategy|leadership|management|ceo|executive)\b', text_lower):
                return "Leadership & Strategy"
            if re.search(r'\b(?:market|industry|sector|trend|analysis)\b', text_lower):
                return "Market Analysis"
                
        # Policy subcategories
        elif category == "Policy & Regulation":
            if re.search(r'\b(?:privacy|data|protection|gdpr|ccpa)\b', text_lower):
                return "Privacy & Data Protection"
            if re.search(r'\b(?:antitrust|competition|monopoly|regulation)\b', text_lower):
                return "Antitrust & Competition"
            if re.search(r'\b(?:security|cyber|hack|breach|vulnerability)\b', text_lower):
                return "Cybersecurity"
                
        return "" 