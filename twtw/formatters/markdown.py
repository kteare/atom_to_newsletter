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
        # Check if content already has HTML tags
        is_html = bool(re.match(r'^\s*<\w+', html_content))
        
        if is_html:
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for element in soup(['script', 'style']):
                    element.decompose()
                
                # Split content by double newlines if no block elements found
                block_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'ul', 'ol', 'div'])
                if not block_elements:
                    # No block elements found, split by double newlines and create paragraphs
                    text_content = soup.get_text()
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
                    element_text = element.get_text()
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
        else:
            # Content is not HTML, treat it as plain text and wrap in paragraph tags
            text = html_content.strip()
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
        if article.reading_time:
            metadata.append(f"**Reading Time:** {article.reading_time} min")
        if article.domain:
            metadata.append(f"**Domain:** {article.domain}")
        
        # Return metadata separated by spaces for better formatting
        return " ".join(metadata)
    
    def format_article(self, article: Article) -> str:
        """
        Format a single article as Markdown.
        
        Args:
            article: The article to format
            
        Returns:
            Formatted article as Markdown
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
        
        # Add image if available - prioritize the main image from content features
        main_image = None
        if article.content_features and 'images' in article.content_features and article.content_features['images']:
            # Find the best image - prefer larger images that aren't icons
            best_image = None
            max_dimension = 0
            
            for img in article.content_features['images']:
                # Skip small images that are likely icons
                width = int(img.get('width', '0') or '0')
                height = int(img.get('height', '0') or '0')
                
                # Calculate image importance based on dimensions and position
                dimension_score = width * height if width > 0 and height > 0 else 0
                
                # If no dimensions, assume it's a regular image
                if dimension_score == 0:
                    dimension_score = 500 * 300  # Default assumed size
                
                # Prefer images with higher dimensions
                if dimension_score > max_dimension:
                    max_dimension = dimension_score
                    best_image = img
            
            if best_image:
                main_image = best_image
        
        # If we found a main image from content features, use it
        if main_image:
            width = main_image.get('width', '800') or '800'
            height = main_image.get('height', 'auto') or 'auto'
            output.append(f'\n<img width="{width}" src="{main_image["src"]}" height="{height}"><br>\n')
        # Fallback to the article's image_url if no content images found
        elif article.image_url:
            output.append(f'\n<img width="800" src="{article.image_url}" height="auto"><br>\n')
        
        # Add content with 500-word limit
        if article.content:
            # Process HTML content to preserve paragraph structure and limit to 500 words
            formatted_content = self._extract_paragraphs_with_limit(article.content, 500)
            output.append(f"\n{formatted_content}\n")
        elif article.full_text:
            # Process HTML content to preserve paragraph structure and limit to 500 words
            formatted_content = self._extract_paragraphs_with_limit(article.full_text, 500)
            output.append(f"\n{formatted_content}\n")
        # Fallback to summary
        elif article.summary:
            output.append(f"\n{article.summary}\n")
        
        # Add read more link
        if article.read_more_url:
            output.append(f"\n[Read more]({article.read_more_url})\n")
        
        # Add separator
        output.append("\n---\n\n")
        
        return ''.join(output)
    
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
                        content.append(self._format_article(article))
                        content.append("---")
                        content.append("")
                
                # Add articles without subcategory
                uncategorized = [a for a in category_articles if not getattr(a, 'subcategory', '')]
                if uncategorized:
                    content.append("### Other")
                    for article in uncategorized:
                        content.append(self._format_article(article))
                        content.append("---")
                        content.append("")
            else:
                # No subcategories, just add all articles
                for article in category_articles:
                    content.append(self._format_article(article))
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
    
    def _format_article(self, article: Article) -> str:
        """
        Format a single article as a string with metadata and content.
        
        This method creates a formatted version of the article with title, metadata, 
        and content, ensuring proper paragraph structure is maintained.
        
        Args:
            article: The article to format
            
        Returns:
            Formatted article as a string
        """
        output = []
        
        # Add title with link
        output.append(f"## [{article.title}]({article.url})\n")
        
        # Add metadata in a div for special processing
        metadata = self.format_article_metadata(article)
        if metadata:
            output.append(metadata + "\n")
        
        # Add image if available - prioritize the main image from content features
        main_image = None
        if article.content_features and 'images' in article.content_features and article.content_features['images']:
            # Find the best image - prefer larger images that aren't icons
            best_image = None
            max_dimension = 0
            
            for img in article.content_features['images']:
                # Skip small images that are likely icons
                width = int(img.get('width', '0') or '0')
                height = int(img.get('height', '0') or '0')
                
                # Calculate image importance based on dimensions and position
                dimension_score = width * height if width > 0 and height > 0 else 0
                
                # If no dimensions, assume it's a regular image
                if dimension_score == 0:
                    dimension_score = 500 * 300  # Default assumed size
                
                # Prefer images with higher dimensions
                if dimension_score > max_dimension:
                    max_dimension = dimension_score
                    best_image = img
            
            if best_image:
                main_image = best_image
        
        # If we found a main image from content features, use it
        if main_image:
            width = main_image.get('width', '800') or '800'
            height = main_image.get('height', 'auto') or 'auto'
            output.append(f'<p style="margin-bottom: 1em;"><img alt="{article.title}" src="{main_image["src"]}"/></p>\n')
        # Fallback to the article's image_url if no content images found
        elif article.image_url:
            output.append(f'<p style="margin-bottom: 1em;"><img alt="{article.title}" src="{article.image_url}"/></p>\n')
        
        # Process content to ensure proper paragraph structure
        article_content = ""
        if article.content:
            article_content = article.content
        elif article.full_text:
            article_content = article.full_text
        elif article.summary:
            article_content = article.summary
        
        if article_content:
            # Check if content is already HTML or plain text
            is_html = bool(re.match(r'^\s*<\w+', article_content))
            
            # Clean up common prefixes that might be duplicated
            common_prefixes = [
                "Listen to this post:",
                "Subscribe to receive"
            ]
            for prefix in common_prefixes:
                if article_content.count(prefix) > 1:
                    first_index = article_content.find(prefix)
                    second_index = article_content.find(prefix, first_index + 1)
                    if first_index >= 0 and second_index >= 0:
                        article_content = article_content[:second_index] + article_content[second_index + len(prefix):]
            
            if is_html:
                # For HTML content, use the extract_paragraphs method to limit words while preserving structure
                formatted_content = self._extract_paragraphs_with_limit(article_content, 500)
                output.append(formatted_content)
            else:
                # For plain text, we need to create proper paragraph structure
                paragraphs = []
                
                # Try to split by double newlines first
                if '\n\n' in article_content:
                    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', article_content) if p.strip()]
                # If that didn't work or resulted in just one paragraph, try to split by sentences
                if len(paragraphs) <= 1 and len(article_content) > 500 and '. ' in article_content:
                    # Split by sentence-ending punctuation followed by space and capital letter
                    sentences = re.split(r'([.!?]\s+)(?=[A-Z])', article_content)
                    current_para = ''
                    sentence_count = 0
                    paragraphs = []
                    
                    for i, part in enumerate(sentences):
                        if i % 2 == 0:  # Content part
                            current_para += part
                            sentence_count += 1
                        else:  # Separator (punctuation)
                            current_para += part
                            # Group roughly 2-3 sentences per paragraph
                            if sentence_count >= 2 and len(current_para) > 200:
                                paragraphs.append(current_para)
                                current_para = ''
                                sentence_count = 0
                
                    # Add any remaining content
                    if current_para:
                        paragraphs.append(current_para)
                
                # If we still don't have paragraphs, use the original content
                if not paragraphs:
                    paragraphs = [article_content]
                
                # Format each paragraph with proper HTML tags
                word_count = 0
                word_limit = 500
                formatted_paragraphs = []
                
                for paragraph in paragraphs:
                    if paragraph.strip():
                        para_words = len(paragraph.split())
                        if word_count + para_words <= word_limit:
                            formatted_paragraphs.append(f'<p style="margin-bottom: 1em;">{paragraph.strip()}</p>')
                            word_count += para_words
                        else:
                            # If we'd exceed the word limit, truncate the paragraph
                            words = paragraph.split()
                            remaining_words = word_limit - word_count
                            if remaining_words > 0:
                                formatted_paragraphs.append(f'<p style="margin-bottom: 1em;">{" ".join(words[:remaining_words])}...</p>')
                            break
                
                output.append('\n'.join(formatted_paragraphs) + '\n')
        
        # Add read more link
        read_more_url = article.read_more_url or article.url
        output.append(f'<p style="margin-bottom: 1em;"><a href="{read_more_url}">Read more...</a></p>\n')
        
        # Add separator
        output.append("<hr/>\n")
        
        return ''.join(output)
    
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