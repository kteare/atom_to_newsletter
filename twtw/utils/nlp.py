import re
import markdown
from bs4 import BeautifulSoup
import mistune

#!/usr/bin/env python3
"""
convert.py

A standalone Python script to convert the most recent organized_newsletter_YYYYMMDD.md
Markdown file to HTML and create a reading list.

Usage:
    python convert.py

This script should be placed in the same directory as your Markdown files.
"""

import os
import sys
import datetime

def find_latest_markdown_file(directory):
    """
    Scans the specified directory for files matching the pattern
    organized_newsletter_YYYYMMDD.md and returns the path to the most recent file.

    Args:
        directory (str): The directory to scan.

    Returns:
        str: The path to the most recent Markdown file.
             Returns None if no matching file is found.
    """
    pattern = re.compile(r'organized_newsletter_(\d{8})\.md$')
    latest_date = None
    latest_file = None

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            file_date_str = match.group(1)
            try:
                file_date = datetime.datetime.strptime(file_date_str, '%Y%m%d').date()
                if (latest_date is None) or (file_date > latest_date):
                    latest_date = file_date
                    latest_file = filename
            except ValueError:
                # Skip files with invalid dates
                continue

    if latest_file:
        return os.path.join(directory, latest_file)
    else:
        return None

def replace_metadata(match):
    metadata_text = match.group(1)
    # Split on tabs to preserve horizontal layout
    metadata_pairs = [pair.strip() for pair in metadata_text.split('\t') if pair.strip()]
    
    # Process each pair
    metadata_items = []
    for pair in metadata_pairs:
        if ':' in pair:
            label, value = pair.split(':', 1)
            # Clean up the label and value
            label = label.replace('**', '').strip()
            value = value.replace('**', '').strip()
            metadata_items.append(f'<strong>{label}:</strong> {value}')
    
    # Use non-breaking spaces and middots for better horizontal preservation
    metadata_html = '&nbsp;&nbsp;•&nbsp;&nbsp;'.join(metadata_items)
    return f'<hr/>\n<div class="metadata">{metadata_html}</div>\n<hr/>'

def convert_markdown_to_html(markdown_file_path, html_file_path):
    """
    Converts a Markdown file to an HTML file using the markdown library.

    Args:
        markdown_file_path (str): Path to the input Markdown file.
        html_file_path (str): Path to the output HTML file.
    """
    try:
        with open(markdown_file_path, 'r', encoding='utf-8') as md_file:
            md_text = md_file.read()

        # Find and process all metadata sections
        md_text = re.sub(
            r'<div class="metadata">\n(.*?)\n</div>',
            replace_metadata,
            md_text,
            flags=re.DOTALL
        )

        # Convert the rest of the markdown using mistune
        md_parser = mistune.create_markdown(escape=False)
        html = md_parser(md_text)
        
        # Create a BeautifulSoup object to parse the HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Suppress DeprecationWarning for the find_all call
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            # Find all text nodes that contain Markdown bold syntax
            for element in soup.find_all(string=lambda s: s and '**' in s):
                # Convert Markdown bold syntax to HTML strong tags
                converted = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', element)
                element.replace_with(BeautifulSoup(converted, 'html.parser'))

        # Convert the modified content back to string
        html = str(soup)

        # Optional: Add basic HTML structure
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="generator" content="TWTW Newsletter Generator">
    <meta name="date" content="{datetime.datetime.now().strftime('%Y-%m-%d')}">
    <title>{os.path.basename(html_file_path).replace('.html', '')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}

        h1, h2, h3 {{
            color: #1a1a1a;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }}

        a {{
            color: #0066cc;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        .article-metadata {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            background-color: #f8f9fa;
            padding: 16px 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 4px solid #0066cc;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .article-metadata p {{
            margin: 0 10px 0 0;
            font-size: 0.95em;
            line-height: 1.4;
            border-right: 1px solid #e9ecef;
            padding-right: 10px;
        }}

        .article-metadata p:last-child {{
            border-right: none;
            padding-right: 0;
        }}

        .article-author {{
            font-size: 1em;
            color: #1a1a1a;
            margin: 0 10px 0 0;
        }}

        .article-source {{
            color: #2d3748;
        }}

        .article-date, .article-time, .article-domain {{
            color: #4a5568;
        }}

        blockquote {{
            border-left: 4px solid #0066cc;
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #f8f9fa;
            color: #2d3748;
        }}

        pre, code {{
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 12px;
            overflow-x: auto;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        }}

        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            margin: 20px 0;
        }}

        hr {{
            border: none;
            border-top: 1px solid #e9ecef;
            margin: 30px 0;
        }}

        .metadata {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin: 1rem 0;
            padding: 0.75rem 1rem;
            background-color: #f8f9fa;
            border-radius: 6px;
            font-size: 0.95em;
            line-height: 1.4;
        }}

        .metadata span {{
            color: #666;
            display: flex;
            align-items: center;
            white-space: nowrap;
        }}

        .metadata span:not(:last-child) {{
            border-right: 1px solid #ddd;
            padding-right: 1rem;
        }}

        .metadata strong {{
            color: #333;
            margin-right: 0.25rem;
            font-weight: 600;
        }}
    </style>
</head>
<body>
{html}
</body>
</html>"""

        with open(html_file_path, 'w', encoding='utf-8') as html_file:
            html_file.write(html_content)

        print(f"Successfully converted '{markdown_file_path}' to '{html_file_path}'.")
    
    except FileNotFoundError:
        print(f"Error: The file '{markdown_file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred during Markdown to HTML conversion: {e}")

def main():
    """Convert markdown files to HTML."""
    # Get the current working directory
    current_directory = os.getcwd()
    output_dir = os.getenv('TWTW_OUTPUT_DIR')
    if output_dir:
        current_directory = os.path.join(current_directory, output_dir)
    
    # Input files
    newsletter_file = os.path.join(current_directory, "newsletter.md")
    reading_file = os.path.join(current_directory, "reading.md")
    snippets_file = os.path.join(current_directory, "snippets.md")
    original_file = os.path.join(current_directory, "original.md")
    contents_file = os.path.join(current_directory, "contents.md")
    
    # Output files
    newsletter_html = os.path.join(current_directory, "newsletter.html")
    reading_html = os.path.join(current_directory, "reading.html")
    snippets_html = os.path.join(current_directory, "snippets.html")
    original_html = os.path.join(current_directory, "original.html")
    contents_html = os.path.join(current_directory, "contents.html")
    
    # Convert newsletter.md to HTML
    if os.path.exists(newsletter_file):
        convert_markdown_to_html(newsletter_file, newsletter_html)
    else:
        print(f"No newsletter.md file found in {current_directory}")
    
    # Convert reading.md to HTML if it exists
    if os.path.exists(reading_file):
        convert_markdown_to_html(reading_file, reading_html)
        
    # Convert snippets.md to HTML if it exists
    if os.path.exists(snippets_file):
        convert_markdown_to_html(snippets_file, snippets_html)
        
    # Convert original.md to HTML if it exists
    if os.path.exists(original_file):
        convert_markdown_to_html(original_file, original_html)

    # Convert contents.md to HTML if it exists
    if os.path.exists(contents_file):
        convert_markdown_to_html(contents_file, contents_html)

if __name__ == "__main__":
    main()

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