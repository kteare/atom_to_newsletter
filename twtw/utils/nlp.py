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
Text analysis utilities for TWTW.
"""
# Remove dependency on markdown
# import markdown
import re
from typing import Tuple, Optional, List, Dict, Any

class TextAnalyzer:
    """
    Simplified text analyzer that uses rule-based approaches instead of NLP libraries.
    """
    def __init__(self):
        """Initialize the analyzer."""
        # Define category keywords
        self.category_keywords = {
            'Tech News': ['tech', 'technology', 'apple', 'google', 'microsoft', 'amazon', 'facebook', 'meta', 
                         'social media', 'iphone', 'android', 'device', 'hardware', 'software', 'app', 
                         'smartphone', 'gadget', 'internet', 'online', 'website', 'mobile'],
            
            'AI': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning', 'neural network', 
                  'gpt', 'chatgpt', 'openai', 'llm', 'large language model', 'generative ai', 'transformer',
                  'computer vision', 'natural language processing', 'nlp', 'claude', 'anthropic'],
            
            'Industry Analysis': ['industry', 'market', 'trend', 'analysis', 'forecast', 'report', 'growth', 
                                 'decline', 'strategy', 'competition', 'competitive', 'sector', 'economy',
                                 'economic', 'business model', 'disruption', 'transformation'],
            
            'Venture Capital': ['vc', 'venture capital', 'investor', 'investment', 'funding', 'series a', 
                               'series b', 'series c', 'startup funding', 'valuation', 'exit', 'acquisition',
                               'ipo', 'portfolio', 'angel investor', 'seed funding', 'venture'],
            
            'Programming': ['code', 'coding', 'programming', 'developer', 'software development', 'engineer', 
                           'github', 'git', 'python', 'javascript', 'java', 'c++', 'rust', 'go', 'golang',
                           'framework', 'library', 'api', 'sdk', 'algorithm', 'database', 'sql'],
            
            'Science': ['science', 'scientific', 'research', 'discovery', 'physics', 'chemistry', 'biology', 
                       'astronomy', 'climate', 'environment', 'medicine', 'medical', 'health', 'disease',
                       'vaccine', 'experiment', 'laboratory', 'study'],
            
            'Startups': ['startup', 'founder', 'entrepreneurship', 'entrepreneur', 'launch', 'early stage', 
                        'seed', 'incubator', 'accelerator', 'y combinator', 'techstars', 'product market fit',
                        'pivot', 'scale', 'scaling', 'growth hacking'],
            
            'Essays': ['essay', 'opinion', 'perspective', 'thought', 'view', 'analysis', 'reflection', 
                      'philosophy', 'argument', 'critique', 'review', 'editorial', 'column', 'think piece']
        }
        
        # Define subcategory keywords
        self.subcategory_keywords = {
            'Tech News': {
                'Mobile': ['mobile', 'iphone', 'android', 'smartphone', 'ios', 'app', 'tablet', 'ipad'],
                'Cloud Computing': ['cloud', 'aws', 'azure', 'google cloud', 'saas', 'paas', 'iaas'],
                'Social Media': ['social media', 'facebook', 'twitter', 'instagram', 'tiktok', 'linkedin'],
                'Hardware': ['hardware', 'device', 'chip', 'processor', 'gpu', 'cpu', 'semiconductor'],
                'Software': ['software', 'app', 'application', 'platform', 'operating system', 'windows', 'macos']
            },
            'AI': {
                'LLMs': ['llm', 'large language model', 'gpt', 'chatgpt', 'claude', 'generative ai', 'text generation'],
                'Computer Vision': ['computer vision', 'image recognition', 'object detection', 'facial recognition'],
                'NLP': ['nlp', 'natural language processing', 'text analysis', 'sentiment analysis'],
                'AI Ethics': ['ethics', 'bias', 'fairness', 'responsible ai', 'ai safety', 'alignment'],
                'AI Applications': ['application', 'use case', 'implementation', 'adoption', 'ai solution']
            },
            'Programming': {
                'Web Development': ['web', 'javascript', 'html', 'css', 'frontend', 'backend', 'fullstack'],
                'Data Science': ['data science', 'data analysis', 'pandas', 'numpy', 'jupyter', 'visualization'],
                'DevOps': ['devops', 'ci/cd', 'pipeline', 'docker', 'kubernetes', 'container', 'deployment'],
                'Security': ['security', 'cybersecurity', 'encryption', 'vulnerability', 'authentication'],
                'Languages': ['language', 'python', 'javascript', 'java', 'c++', 'rust', 'go', 'typescript']
            }
            # Other categories follow similar patterns
        }
        
    def classify_text(self, text: str, title: str = "", source: str = "") -> Tuple[str, float]:
        """
        Classify text into predefined categories.
        
        Args:
            text: The text to classify
            title: The title of the article (optional)
            source: The source of the article (optional)
            
        Returns:
            Tuple of (category, confidence_score)
        """
        if not text and not title:
            return "Other", 0.0
            
        # Combine text and title for classification, with title weighted more
        combined_text = (title + " " + title + " " + (text or "")).lower()
        
        # Count keyword matches for each category
        scores = {}
        for category, keywords in self.category_keywords.items():
            count = 0
            for keyword in keywords:
                count += combined_text.count(keyword.lower())
            scores[category] = count
            
        # If source is provided, boost certain categories based on source
        if source:
            source_lower = source.lower()
            
            # Tech news sources
            if any(s in source_lower for s in ['techcrunch', 'wired', 'verge', 'engadget', 'cnet']):
                scores['Tech News'] = scores.get('Tech News', 0) + 3
                
            # AI sources
            if any(s in source_lower for s in ['ai', 'artificial intelligence', 'openai', 'deepmind']):
                scores['AI'] = scores.get('AI', 0) + 3
                
            # Programming sources
            if any(s in source_lower for s in ['github', 'stackoverflow', 'dev.to', 'hackernews']):
                scores['Programming'] = scores.get('Programming', 0) + 3
                
            # VC sources
            if any(s in source_lower for s in ['techcrunch', 'venture', 'vc', 'funded', 'crunchbase']):
                scores['Venture Capital'] = scores.get('Venture Capital', 0) + 3
        
        # If no clear category, default to "Other"
        if not scores or max(scores.values(), default=0) == 0:
            return "Other", 0.0
            
        # Find the category with the highest score
        best_category = max(scores.items(), key=lambda x: x[1])
        
        # Calculate a confidence score (0-1)
        total_score = sum(scores.values())
        confidence = best_category[1] / total_score if total_score > 0 else 0.0
        
        return best_category[0], min(1.0, confidence)
        
    def get_subcategory(self, text: str, category: str) -> Optional[str]:
        """
        Get the subcategory for a given text and main category.
        
        Args:
            text: The text to classify
            category: The main category of the text
            
        Returns:
            Subcategory name or None if not found
        """
        if not text or not category or category not in self.subcategory_keywords:
            return None
            
        text_lower = text.lower()
        
        # Look for subcategory keywords within the main category
        scores = {}
        for subcategory, keywords in self.subcategory_keywords.get(category, {}).items():
            count = 0
            for keyword in keywords:
                count += text_lower.count(keyword.lower())
            scores[subcategory] = count
            
        # Find the subcategory with the highest score
        if not scores or max(scores.values(), default=0) == 0:
            return None
            
        best_subcategory = max(scores.items(), key=lambda x: x[1])
        
        # Only return if we have a reasonable match
        if best_subcategory[1] > 0:
            return best_subcategory[0]
            
        return None

    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract main keywords from text.
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        if not text:
            return []
            
        # Simple word frequency approach (a real NLP system would be more sophisticated)
        words = re.findall(r'\b[a-zA-Z]{3,15}\b', text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'is', 'in', 'to', 'a', 'of', 'for', 'with', 'on', 'that', 
                     'this', 'it', 'as', 'by', 'from', 'was', 'were', 'are', 'be', 'have', 'has'}
        filtered_words = [w for w in words if w not in stop_words]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Sort by frequency and take top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, count in sorted_words[:max_keywords]] 