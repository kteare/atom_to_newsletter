"""
Markdown formatting utilities for TWTW.
"""
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import copy

from twtw.core.article import Article
from bs4 import BeautifulSoup, NavigableString
import logging

# Configure logging
logger = logging.getLogger(__name__)

class MarkdownFormatter:
    """
    Formats articles into Markdown content.
    """
    def __init__(self):
        """
        Initialize the MarkdownFormatter.
        """
        self.today = datetime.now().strftime("%B %d, %Y")
        self.classification_confidence_threshold = 0.7
    
    def _extract_paragraphs_with_limit(self, html_content, word_limit=500):
        """
        Extract paragraphs from HTML content with a word limit.
        
        This method preserves HTML formatting including links, bold text, and other elements.
        
        Args:
            html_content: HTML content to extract paragraphs from
            word_limit: Maximum number of words to extract
            
        Returns:
            Extracted paragraphs as HTML
        """
        # Safety check - if content is None or empty
        if not html_content:
            return ""
            
        # Pre-process step: Check for content that is split character by character
        # This can happen with some content feeds where newlines are inserted between characters
        # Look for patterns like "A\nm\na\nz\no\nn" or "A m a z o n"
        if '\n' in html_content and len(html_content.strip()) > 20:
            # Check for single characters or spaces separated by newlines
            lines = html_content.split('\n')
            stripped_lines = [line.strip() for line in lines]
            
            # If more than 60% of non-empty lines are single characters or spaces, it's likely character-by-character splitting
            non_empty_lines = [line for line in stripped_lines if line]
            single_char_lines = [line for line in non_empty_lines if len(line) <= 1]
            
            if len(non_empty_lines) > 5 and len(single_char_lines) / len(non_empty_lines) > 0.6:
                logger.warning("Detected character-by-character splitting with newlines, joining characters")
                # Join all characters, ignoring newlines
                joined_text = ''.join(non_empty_lines)
                # Remove any HTML tags
                clean_text = re.sub(r'<[^>]*>', ' ', joined_text)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                
                # Create a paragraph with the cleaned text
                if clean_text:
                    # Count words for the log
                    word_count = len(clean_text.split())
                    logger.info(f"Created {word_count} words from character-by-character text")
                    # Return up to word_limit words
                    words = clean_text.split()
                    if len(words) > word_limit:
                        return f"<p>{' '.join(words[:word_limit])}...</p>"
                    return f"<p>{clean_text}</p>"
                return ""
        
        # Check for content with spaces between individual characters (like "A m a z o n")
        if re.search(r'\b(\w \w \w \w)\b', html_content):
            logger.warning("Detected character-by-character splitting with spaces, joining characters")
            # This pattern identifies text where individual characters are separated by spaces
            joined_text = re.sub(r'(\w) (\w) (\w)', r'\1\2\3', html_content)
            joined_text = re.sub(r'(\w) (\w)', r'\1\2', joined_text)  # Run twice to catch overlapping patterns
            joined_text = re.sub(r'(\w) (\w)', r'\1\2', joined_text)
            
            # If HTML tags are mixed with spaced characters, clean them
            clean_text = re.sub(r'<[^>]*>', ' ', joined_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            # Create a paragraph with the cleaned text
            if clean_text:
                # Count words for the log
                word_count = len(clean_text.split())
                logger.info(f"Created {word_count} words from space-separated character text")
                # Return up to word_limit words
                words = clean_text.split()
                if len(words) > word_limit:
                    return f"<p>{' '.join(words[:word_limit])}...</p>"
                return f"<p>{clean_text}</p>"
            return ""
            
        # Special case detection for character-by-character splitting with newlines
        # This pattern is common when text has been poorly processed
        if re.search(r'(<[^>]*>)?\s*\n\s*\w\s*\n\s*\w\s*\n\s*\w', html_content):
            logger.warning("Detected complex character-by-character splitting pattern with newlines, fixing...")
            # Join all the characters that are split across lines
            lines = html_content.split('\n')
            joined_text = ''.join([line.strip() for line in lines])
            # Remove any HTML tags
            clean_text = re.sub(r'<[^>]*>', ' ', joined_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            # Create a paragraph with the cleaned text
            if clean_text:
                # Count words for the log
                word_count = len(clean_text.split())
                logger.info(f"Created {word_count} words from character-by-character text")
                # Return up to word_limit words
                words = clean_text.split()
                if len(words) > word_limit:
                    return f"<p>{' '.join(words[:word_limit])}...</p>"
                return f"<p>{clean_text}</p>"
            return ""
            
        # Check if content already has HTML tags - look for complete tags, not just angle brackets
        is_html = bool(re.search(r'<[a-zA-Z][^>]*>(.*?)</[a-zA-Z][^>]*>', html_content, re.DOTALL))
        
        if is_html:
            try:
                # Clean up potential raw HTML fragments that shouldn't be rendered
                html_content = re.sub(r'<\s*!DOCTYPE.*?>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
                html_content = re.sub(r'<html.*?>.*?</html>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
                html_content = re.sub(r'<\s*meta.*?>', '', html_content, flags=re.IGNORECASE)
                html_content = re.sub(r'<\s*link.*?>', '', html_content, flags=re.IGNORECASE)
                html_content = re.sub(r'<\s*script.*?>.*?</\s*script\s*>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
                html_content = re.sub(r'<\s*style.*?>.*?</\s*style\s*>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
                
                # Check for content that might be HTML-like but not properly formed
                # This happens when angle brackets are present but the content is not proper HTML
                # For example: <p>T<h>i<s> <i>s> <n>o<t> <p>r<o>p<e>r <H>T<M>L</p>
                if re.search(r'<[a-zA-Z][^>]{0,1}>[^<]{0,1}<', html_content):
                    logger.warning("Detected malformed HTML with character-by-character splitting, treating as plain text")
                    # If this pattern is found, treat as plain text instead
                    is_html = False
                    # Clean the content by removing the angle brackets
                    html_content = re.sub(r'<[^>]*>', ' ', html_content)
                    html_content = re.sub(r'\s+', ' ', html_content).strip()
                else:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove script and style elements
                    for element in soup(['script', 'style']):
                        element.decompose()
                    
                    # Split content by double newlines if no block elements found
                    block_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'ul', 'ol', 'div'])
                    if not block_elements:
                        # No block elements found, split by double newlines and create paragraphs
                        text_content = soup.get_text(separator=' ')
                        paragraphs = re.split(r'\n\s*\n', text_content)
                        total_words = 0
                        result = ""
                        for para in paragraphs:
                            if para.strip():
                                para_text = para.strip()
                                para_words = len(para_text.split())
                                if total_words + para_words <= word_limit:
                                    result += f"<p>{para_text}</p>\n\n"
                                    total_words += para_words
                                else:
                                    # If we would exceed the word limit, truncate the paragraph
                                    words = para_text.split()
                                    remaining_words = word_limit - total_words
                                    if remaining_words > 0:
                                        result += f"<p>{' '.join(words[:remaining_words])}...</p>\n\n"
                                    break
                        return result
                    
                    # Extract block elements
                    elements = []
                    for tag in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'ul', 'ol', 'div']:
                        elements.extend(soup.find_all(tag))
                    
                    # Sort elements by their position in the document
                    elements.sort(key=lambda x: x.sourceline or 0)
                    
                    # Extract content up to the word limit
                    total_words = 0
                    result = ""
                    for element in elements:
                        element_text = element.get_text(' ', strip=True)
                        element_words = len(element_text.split())
                        
                        if total_words + element_words <= word_limit:
                            result += str(element) + "\n\n"
                            total_words += element_words
                        else:
                            # If we would exceed the word limit, truncate the element
                            if element.name in ['p', 'div', 'blockquote']:
                                words = element_text.split()
                                remaining_words = word_limit - total_words
                                if remaining_words > 0:
                                    # Create a new element with truncated text
                                    truncated_element = copy.copy(element)
                                    truncated_element.string = ' '.join(words[:remaining_words]) + '...'
                                    result += str(truncated_element) + "\n\n"
                            break
                    
                    logger.info(f"Extracted {total_words} words from HTML content")
                    return result
            
            except Exception as e:
                logger.error(f"Error extracting paragraphs: {e}")
                # Fallback to simple text extraction
                return self._truncate_html_by_words(html_content, word_limit)
        
        # If we reach here, either the content is not HTML or we've converted it to plain text
        # Content is not HTML, treat it as plain text and wrap in paragraph tags
        # First, clean up any HTML fragments that might be in the text
        text = re.sub(r'<\s*!DOCTYPE.*?>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<html.*?>.*?</html>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<\s*meta.*?>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*link.*?>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*script.*?>.*?</\s*script\s*>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<\s*style.*?>.*?</\s*style\s*>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove any remaining angle brackets
        text = re.sub(r'<[^>]*>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return ""
        
        # Split by double newlines to identify paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        total_words = 0
        result = ""
        
        for para in paragraphs:
            if para.strip():
                para_text = para.strip()
                para_words = len(para_text.split())
                
                if total_words + para_words <= word_limit:
                    result += f"<p>{para_text}</p>\n\n"
                    total_words += para_words
                else:
                    # If we would exceed the word limit, truncate the paragraph
                    words = para_text.split()
                    remaining_words = word_limit - total_words
                    if remaining_words > 0:
                        result += f"<p>{' '.join(words[:remaining_words])}...</p>\n\n"
                    break
        
        logger.info(f"Created {total_words} words of paragraphs from plain text")
        return result
    
    def _truncate_html_by_words(self, html, word_limit):
        """
        Truncate HTML content to a specific word count while preserving HTML tags.
        
        Args:
            html: HTML content to truncate
            word_limit: Maximum number of words to include
            
        Returns:
            Truncated HTML string
        """
        soup = BeautifulSoup(html, 'html.parser')
        words_seen = 0
        
        # Recursive function to process nodes
        def process_node(node):
            nonlocal words_seen
            
            # If we've reached our limit, stop processing
            if words_seen >= word_limit:
                return False
                
            # Process text nodes
            if isinstance(node, NavigableString):
                text = node.string
                words = text.split()
                
                if words_seen + len(words) <= word_limit:
                    # Keep the entire text
                    words_seen += len(words)
                    return True
                else:
                    # Keep only some of the words
                    words_to_keep = word_limit - words_seen
                    node.replace_with(' '.join(words[:words_to_keep]))
                    words_seen = word_limit
                    return False
            
            # Process element nodes
            elif node.name:
                # Make a copy of children since we might modify during iteration
                children = list(node.children)
                for child in children:
                    if not process_node(child):
                        # If child processing indicates we've hit the limit,
                        # remove all subsequent siblings
                        next_sibling = child.next_sibling
                        while next_sibling:
                            temp = next_sibling.next_sibling
                            next_sibling.extract()
                            next_sibling = temp
                        return False
                return True
                
            return True
            
        # Start processing from the root
        process_node(soup)
        
        # Return the truncated HTML
        return str(soup)
    
    def format_article_metadata(self, article: Article) -> str:
        """
        Format article metadata with consistent layout.
        
        Args:
            article: The article to format metadata for
            
        Returns:
            Formatted metadata string
        """
        metadata = []
        
        # Build metadata parts in consistent order
        if article.author:
            metadata.append(f"**Author:** {article.author}")
        if article.source:
            metadata.append(f"**Source:** {article.source}")
        if article.published:
            try:
                published_date = datetime.strptime(article.published, "%Y-%m-%dT%H:%M:%SZ").strftime("%B %d, %Y")
                metadata.append(f"**Published:** {published_date}")
            except (ValueError, TypeError):
                if article.published:
                    metadata.append(f"**Published:** {article.published}")
        if hasattr(article, 'category') and article.category:
            metadata.append(f"**Classification:** {article.category}")
        if hasattr(article, 'subcategory') and article.subcategory:
            metadata.append(f"**Subcategory:** {article.subcategory}")
        if article.domain:
            metadata.append(f"**Domain:** {article.domain}")
        
        # Return metadata separated by spaces for better formatting
        return " ".join(metadata)
    
    def format_article(self, article: Article) -> str:
        """
        Format an article for display in the newsletter.
        
        Args:
            article: The article to format
            
        Returns:
            Formatted article as a string
        """
        output = []
        
        # Add title with link
        output.append(f"### [{article.title}]({article.url})")
        output.append("")
        
        # Add metadata
        meta = self.format_article_metadata(article)
        output.append(meta)
        output.append("")
        
        # Check for image
        if article.image_url:
            output.append(f"![{article.title}]({article.image_url})")
            output.append("")
        
        # Add content with 1500-word limit - RULE #5: Use the processed content from OpenAI
        if hasattr(article, 'processed_content') and article.processed_content:
            # Use the processed content from OpenAI directly
            output.append(article.processed_content)
        elif article.content:
            # Fallback to extracting from HTML if processed_content is not available
            extracted = self._extract_paragraphs_with_limit(article.content, word_limit=1500)
            if extracted:
                output.append(extracted)
            elif article.full_text:
                # Fallback to full_text if content extraction failed
                extracted = self._extract_paragraphs_with_limit(article.full_text, word_limit=1500)
                if extracted:
                    output.append(extracted)
                else:
                    # Last resort, use summary
                    output.append(article.summary if article.summary else "*No content available*")
            else:
                # If no content, use summary
                output.append(article.summary if article.summary else "*No content available*")
        else:
            # If no content, use summary
            output.append(article.summary if article.summary else "*No content available*")
        
        # Add "Read more" link at the end
        output.append("")
        output.append(f"[Read more]({article.url})")
        output.append("")
        
        return "\n".join(output)
    
    def format_article_original(self, article: Article) -> str:
        """
        Format a single article with original content as Markdown.
        
        Args:
            article: The article to format
            
        Returns:
            Formatted article with original content as Markdown
        """
        output = []
        
        # Add title with link
        output.append(f"## [{article.title}]({article.url})\n")
        
        # Add metadata in a div for special processing
        metadata = self.format_article_metadata(article)
        if metadata:
            output.append('<div class="metadata">\n')
            output.append(metadata)
            output.append('\n</div>\n')
        
        # Add full text if available
        if article.full_text:
            # Process HTML content to preserve paragraph structure
            formatted_content = self._extract_paragraphs_with_limit(article.full_text, 500)
            output.append(f"\n{formatted_content}\n")
        else:
            # Fallback to summary
            if article.summary:
                output.append(f"\n{article.summary}\n")
        
        # Add read more link
        if article.read_more_url:
            output.append(f"\n[Read more]({article.read_more_url})\n")
        
        # Add separator
        output.append("\n---\n\n")
        
        return ''.join(output)
    
    def format_newsletter(self, articles: List[Article], categorized_articles: Dict[str, List[Article]], sorted_categories: List[str]) -> str:
        """
        Format articles into a newsletter.
        
        Args:
            articles: List of articles to format
            categorized_articles: Dict of articles grouped by category
            sorted_categories: List of categories in sorted order
            
        Returns:
            Formatted newsletter content
        """
        # Start with the header
        content = [
            "# That Was The Week",
            f"Generated on {self.today}",
            "",
            "## In This Issue"
        ]
        
        # Add table of contents
        for category in sorted_categories:
            category_articles = categorized_articles[category]
            content.append(f"- {category}: {len(category_articles)} articles")
        
        content.append("---")
        content.append("")
        
        # Add articles by category
        for category in sorted_categories:
            category_articles = categorized_articles[category]
            
            # Add category header
            content.append(f"## {category}")
            
            # Group by subcategory if available
            subcategorized = {}
            for article in category_articles:
                subcategory = getattr(article, 'subcategory', '')
                if subcategory:
                    if subcategory not in subcategorized:
                        subcategorized[subcategory] = []
                    subcategorized[subcategory].append(article)
            
            # Add subcategory sections if available
            if subcategorized:
                for subcategory, subcategory_articles in subcategorized.items():
                    content.append(f"### {subcategory}")
                    for article in subcategory_articles:
                        # Extend the content list with the lines from _format_article
                        content.extend(self._format_article(article))
                        content.append("---")
                        content.append("")
                
                # Add articles without subcategory
                uncategorized = [a for a in category_articles if not getattr(a, 'subcategory', '')]
                if uncategorized:
                    content.append("### Other")
                    for article in uncategorized:
                        # Extend the content list with the lines from _format_article
                        content.extend(self._format_article(article))
                        content.append("---")
                        content.append("")
            else:
                # No subcategories, just add all articles
                for article in category_articles:
                    # Extend the content list with the lines from _format_article
                    content.extend(self._format_article(article))
                    content.append("---")
                    content.append("")
        
        return "\n".join(content)
    
    def format_reading_list(self, articles: List[Article]) -> str:
        """
        Format articles into a reading list.
        
        Args:
            articles: List of articles to format
            
        Returns:
            Formatted reading list content
        """
        content = [
            "# That Was The Week - Reading List",
            "",
            f"Generated on {self.today}",
            ""
        ]
        
        # Count paywalled articles
        paywalled_count = sum(1 for article in articles if getattr(article, 'is_paywalled', False))
        if paywalled_count > 0:
            content.append(f"*Note: {paywalled_count} article{'s' if paywalled_count > 1 else ''} may be behind a paywall.*")
            content.append("")
        
        # Group by category
        categorized = {}
        for article in articles:
            category = getattr(article, 'category', 'Unclassified')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(article)
        
        # Add articles by category
        sorted_categories = sorted(categorized.keys())
        
        if len(sorted_categories) > 1:  # Only use categories if there's more than one
            for category in sorted_categories:
                content.append(f"## {category}")
                for article in categorized[category]:
                    paywall_indicator = " 🔒" if getattr(article, 'is_paywalled', False) else ""
                    content.append(f"- [{article.title}]({article.url}){paywall_indicator}")
                content.append("")
        else:
            # Just a flat list if only one category
            for article in articles:
                paywall_indicator = " 🔒" if getattr(article, 'is_paywalled', False) else ""
                content.append(f"- [{article.title}]({article.url}){paywall_indicator}")
        
        return "\n".join(content)
    
    def _format_article(self, article: Article) -> List[str]:
        """
        Format a single article in Markdown format.
        Returns a list of lines representing the article content.
        """
        lines = []
        
        # Add title with header formatting
        lines.append(f"### {article.title}")
        lines.append("")
        
        # Add metadata div
        lines.append('<div class="metadata">')
        
        # Source
        if article.source:
            lines.append(f"**Source:** {article.source}")
            
        # Date - check if it exists, use published instead of date
        if article.published:
            try:
                published_date = datetime.strptime(article.published, "%Y-%m-%dT%H:%M:%SZ").strftime("%B %d, %Y")
                lines.append(f"**Date:** {published_date}")
            except (ValueError, TypeError):
                lines.append(f"**Date:** {article.published}")
            
        # Category and subcategory
        if article.category:
            category_text = article.category
            if article.subcategory:
                category_text += f" / {article.subcategory}"
            lines.append(f"**Category:** {category_text}")
            
        # Reading time
        if article.reading_time:
            lines.append(f"**Reading time:** {article.reading_time} min")
            
        lines.append('</div>')
        lines.append("")
        
        # Add main image if available
        if hasattr(article, 'features') and article.features and 'main_image' in article.features and article.features['main_image']:
            lines.append(f"![{article.title}]({article.features['main_image']})")
            lines.append("")
        elif article.image_url:
            lines.append(f"![{article.title}]({article.image_url})")
            lines.append("")
        
        # Add content
        if article.processed_content:
            lines.append(article.processed_content)
        elif article.content:
            paragraphs = self._extract_paragraphs_with_limit(article.content, 1500)
            lines.extend(paragraphs)
        elif article.full_text:
            paragraphs = self._extract_paragraphs_with_limit(article.full_text, 1500)
            lines.extend(paragraphs)
        elif article.summary:
            lines.append(article.summary)
        else:
            lines.append("*No content available*")
        
        # Add read more link
        if article.url:
            lines.append("")
            lines.append(f"[Read more]({article.url})")
        
        # Add separator
        lines.append("")
        lines.append("---")
        lines.append("")
        
        return lines
    
    def format_snippets(self, articles: List[Article]) -> str:
        """
        Format articles with snippets as Markdown.
        
        Args:
            articles: List of articles to format
            
        Returns:
            Formatted articles with snippets as Markdown
        """
        formatted = "# That Was The Week - Extended Reading List\n\n"
        formatted += f"Generated on {self.today}\n\n"
        
        # Group articles by category
        categorized_articles = {}
        for article in articles:
            category = article.category if hasattr(article, 'category') and article.category else "Uncategorized"
            if category not in categorized_articles:
                categorized_articles[category] = []
            categorized_articles[category].append(article)
        
        # Sort categories
        sorted_categories = sorted(categorized_articles.keys())
        if "Uncategorized" in sorted_categories:
            sorted_categories.remove("Uncategorized")
            
        # Add articles by category
        for category in sorted_categories:
            formatted += f"## {category}\n\n"
            
            for article in categorized_articles[category]:
                # Add title and link
                formatted += f"### [{article.title}]({article.url})\n\n"
                
                # Add metadata
                metadata = []
                
                if article.source:
                    metadata.append(f"**Source:** {article.source}")
                
                if article.author:
                    metadata.append(f"**Author:** {article.author}")
                
                if article.published:
                    metadata.append(f"**Published:** {article.published}")
                
                if article.reading_time:
                    metadata.append(f"**Reading Time:** {article.reading_time} min")
                
                if article.domain:
                    metadata.append(f"**Domain:** {article.domain}")
                
                # Add metadata div
                if metadata:
                    formatted += "<div class=\"metadata\">\n"
                    formatted += "\t".join(metadata)
                    formatted += "\n</div>\n\n"
                
                # Add snippet
                if hasattr(article, 'content') and article.content:
                    # Extract a snippet with preserved paragraph structure
                    formatted += self._extract_paragraphs_with_limit(article.content, 500) + '\n\n'
                elif article.full_text:
                    # Use full_text if content is not available
                    formatted += self._extract_paragraphs_with_limit(article.full_text, 500) + '\n\n'
                elif article.summary:
                    # Fallback to summary
                    words = article.summary.split()
                    if len(words) > 500:
                        formatted += ' '.join(words[:500]) + '...\n\n'
                    else:
                        formatted += article.summary + '\n\n'
                
                # Add read more link
                formatted += f"[Read more]({article.url})\n\n"
                
                # Add separator
                formatted += "---\n\n"
        
        # Add uncategorized articles at the end
        if 'Uncategorized' in categorized_articles and categorized_articles['Uncategorized']:
            formatted += "## Uncategorized\n\n"
            
            for article in categorized_articles['Uncategorized']:
                # Add title and link
                formatted += f"### [{article.title}]({article.url})\n\n"
                
                # Add metadata
                metadata = []
                
                if article.source:
                    metadata.append(f"**Source:** {article.source}")
                
                if article.author:
                    metadata.append(f"**Author:** {article.author}")
                
                if article.published:
                    metadata.append(f"**Published:** {article.published}")
                
                if article.reading_time:
                    metadata.append(f"**Reading Time:** {article.reading_time} min")
                
                if article.domain:
                    metadata.append(f"**Domain:** {article.domain}")
                
                # Add metadata div
                if metadata:
                    formatted += "<div class=\"metadata\">\n"
                    formatted += "\t".join(metadata)
                    formatted += "\n</div>\n\n"
                
                # Add snippet
                if hasattr(article, 'content') and article.content:
                    # Extract a snippet with preserved paragraph structure
                    formatted += self._extract_paragraphs_with_limit(article.content, 500) + '\n\n'
                elif article.full_text:
                    # Use full_text if content is not available
                    formatted += self._extract_paragraphs_with_limit(article.full_text, 500) + '\n\n'
                elif article.summary:
                    # Fallback to summary
                    words = article.summary.split()
                    if len(words) > 500:
                        formatted += ' '.join(words[:500]) + '...\n\n'
                    else:
                        formatted += article.summary + '\n\n'
                
                # Add read more link
                formatted += f"[Read more]({article.url})\n\n"
                
                # Add separator
                formatted += "---\n\n"
        
        return formatted
    
    def format_contents_page(self, articles: List[Article]) -> str:
        """
        Format articles into a contents page.
        
        Args:
            articles: List of articles to format
            
        Returns:
            Formatted contents page
        """
        content = [
            "# That Was The Week - Contents",
            "",
            f"Generated on {self.today}",
            "",
            "## Table of Contents"
        ]
        
        # Group by category
        categorized = {}
        for article in articles:
            category = getattr(article, 'category', 'Unclassified')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(article)
        
        # Add categories to table of contents
        sorted_categories = sorted(categorized.keys())
        
        for i, category in enumerate(sorted_categories, 1):
            content.append(f"{i}. [{category}](#{category.lower().replace(' ', '-')})")
            
            # Add subcategories if available
            subcategorized = {}
            for article in categorized[category]:
                subcategory = getattr(article, 'subcategory', '')
                if subcategory:
                    if subcategory not in subcategorized:
                        subcategorized[subcategory] = []
                    subcategorized[subcategory].append(article)
            
            if subcategorized:
                for j, subcategory in enumerate(sorted(subcategorized.keys()), 1):
                    content.append(f"   {i}.{j}. [{subcategory}](#{subcategory.lower().replace(' ', '-')})")
        
        content.append("")
        
        # Add detailed contents
        for i, category in enumerate(sorted_categories, 1):
            content.append(f"## {i}. {category}")
            
            # Add subcategories if available
            subcategorized = {}
            for article in categorized[category]:
                subcategory = getattr(article, 'subcategory', '')
                if subcategory:
                    if subcategory not in subcategorized:
                        subcategorized[subcategory] = []
                    subcategorized[subcategory].append(article)
            
            if subcategorized:
                for j, subcategory in enumerate(sorted(subcategorized.keys()), 1):
                    content.append(f"### {i}.{j}. {subcategory}")
                    
                    for k, article in enumerate(subcategorized[subcategory], 1):
                        content.append(f"{i}.{j}.{k}. [{article.title}]({article.url})")
                    
                    content.append("")
                
                # Add articles without subcategory
                uncategorized = [a for a in categorized[category] if not getattr(a, 'subcategory', '')]
                if uncategorized:
                    content.append(f"### {i}.{len(subcategorized) + 1}. Other")
                    
                    for k, article in enumerate(uncategorized, 1):
                        content.append(f"{i}.{len(subcategorized) + 1}.{k}. [{article.title}]({article.url})")
                    
                    content.append("")
            else:
                # No subcategories, just add all articles
                for j, article in enumerate(categorized[category], 1):
                    content.append(f"{i}.{j}. [{article.title}]({article.url})")
                
                content.append("")
        
        return "\n".join(content)
    
    def _classify_article(self, article: Article) -> str:
        """
        Simple classification logic (placeholder).
        In a real implementation, this would use more sophisticated classification.
        
        Args:
            article: The article to classify
            
        Returns:
            Category name
        """
        # This is a simplified placeholder
        # In the real implementation, this would use the category_score
        if article.category_score:
            # Find the category with the highest score
            max_category = max(article.category_score.items(), key=lambda x: x[1])
            if max_category[1] >= self.classification_confidence_threshold:
                return max_category[0]
        
        # Default to Unclassified if no clear category
        return "Unclassified" 