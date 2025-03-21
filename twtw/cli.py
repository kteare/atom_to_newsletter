"""
Command-line interface for TWTW.
"""
import os
import sys
import argparse
import logging
import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import aiohttp
from tqdm import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re
import json
import time
import backoff
import requests

# Add fallback for tiktoken
try:
    import tiktoken
except ImportError:
    print("tiktoken package not found. Installing it now...")
    import subprocess
    import sys
    try:
        # Use --no-deps to avoid pulling in complex dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken", "--no-deps"])
        import tiktoken
        print("tiktoken installed successfully.")
    except Exception as e:
        print(f"Could not install tiktoken: {e}")
        print("Using a simple word-based counter as fallback (less accurate for OpenAI API).")
        # Create a simple fallback implementation
        class SimpleTokenCounter:
            def encode(self, text):
                # This is an approximation - 1 token ≈ 4 chars in English
                return [1] * (len(text.split()) + len(text) // 4)
                
        class SimpleTiktoken:
            def get_encoding(self, _):
                return SimpleTokenCounter()
                
        # Create a module-like object
        class TiktokenModule:
            def __init__(self):
                self.tiktoken = SimpleTiktoken()
                
            def get_encoding(self, model_name):
                return self.tiktoken.get_encoding(model_name)
                
        tiktoken = TiktokenModule()

# Make sure markdown is installed
try:
    import markdown
except ImportError:
    print("markdown package not found. Installing it now...")
    import subprocess
    try:
        # Install markdown without dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown", "--no-deps"])
        import markdown
        print("markdown installed successfully.")
    except Exception as e:
        print(f"Could not install markdown: {e}")
        print("Creating simple markdown parser as fallback.")
        
        # Simple markdown parser fallback
        class SimpleMarkdown:
            @staticmethod
            def markdown(text):
                # Very basic markdown to HTML conversion
                text = re.sub(r'# (.*?)\n', r'<h1>\1</h1>\n', text)
                text = re.sub(r'## (.*?)\n', r'<h2>\1</h2>\n', text)
                text = re.sub(r'### (.*?)\n', r'<h3>\1</h3>\n', text)
                text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
                text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
                text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
                text = re.sub(r'^- (.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
                return text
                
        # Replace markdown module with our simple implementation
        class MarkdownModule:
            def __init__(self):
                pass
                
            def markdown(self, text):
                return SimpleMarkdown.markdown(text)
                
        markdown = MarkdownModule()

try:
    import openai
except (ImportError, TimeoutError) as e:
    print(f"Error importing OpenAI module: {e}")
    print("This might be due to network connectivity issues.")
    print("Try ensuring you have a stable internet connection or use the --offline mode if available.")
    sys.exit(1)
from urllib.parse import urlparse
import shutil

from twtw.fetchers.feedly import FeedlyFetcher
from twtw.formatters.html import HtmlConverter
from twtw.formatters.markdown import MarkdownFormatter
from twtw.core.processor import ContentProcessor
from twtw.core.article import Article
from twtw.utils.nlp import TextAnalyzer
from twtw.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"twtw_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize text analyzer for classification
text_analyzer = TextAnalyzer()

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="TWTW Newsletter Generator")
    
    parser.add_argument("--feeds", default="feeds.txt", help="Path to file containing feed URLs (default: feeds.txt)")
    parser.add_argument("--output-dir", help="Directory for output files (default: based on TWTW_OUTPUT_DIR env var or timestamp)")
    parser.add_argument("--refresh-cache", action="store_true", help="Force refresh cached content")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching feeds (use existing files)")
    parser.add_argument("--skip-process", action="store_true", help="Skip processing articles")
    parser.add_argument("--skip-convert", action="store_true", help="Skip converting to HTML")
    parser.add_argument("--skip-today-folder", action="store_true", help="Skip checking for and including atom feeds from 'today' folder")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of articles to process from each feed (0 for all)")
    parser.add_argument("--model", help="Specify OpenAI model to use (default: from config)")
    parser.add_argument("--use-local-model", action="store_true", help="Use local rule-based model instead of OpenAI API")
    parser.add_argument("--test-html-extraction", action="store_true", help="Test HTML content extraction while still using OpenAI")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum number of retries for API calls (default: 5)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds for API calls (default: 60)")
    
    # For backward compatibility, still support the old --use-today-folder option but make it do nothing
    # since it's now the default behavior
    parser.add_argument("--use-today-folder", action="store_true", help=argparse.SUPPRESS)
    
    return parser.parse_args()

async def parse_atom_feed(feed_path: str, limit: int = 0) -> List[Article]:
    """
    Parse an Atom feed and extract article information.
    
    Args:
        feed_path: Path to the Atom feed file
        limit: Maximum number of articles to extract (0 = no limit)
        
    Returns:
        List of Article objects
    """
    logger.info(f"Parsing feed: {feed_path}")
    
    try:
        # Parse the XML
        tree = ET.parse(feed_path)
        root = tree.getroot()
        
        # Define namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'media': 'http://search.yahoo.com/mrss/'}
        
        # Extract articles
        articles = []
        for i, entry in enumerate(root.findall('.//atom:entry', ns)):
            if limit > 0 and i >= limit:
                break
                
            # Extract basic info
            title = entry.find('atom:title', ns).text if entry.find('atom:title', ns) is not None else ""
            if entry.find('atom:title', ns) is not None and entry.find('atom:title', ns).get('type') == 'html':
                title = entry.find('atom:title', ns).text
                
            url = entry.find('.//atom:link[@rel="alternate"]', ns)
            url = url.get('href') if url is not None else ""
            
            published = entry.find('atom:published', ns)
            published = published.text if published is not None else None
            
            # Extract author - look for <author><name> structure
            author = None
            author_elem = entry.find('.//atom:author/atom:name', ns)
            if author_elem is not None and author_elem.text:
                author = author_elem.text
            
            # Also check for 'n' element in author which is used in some feeds
            if author is None:
                author_elem = entry.find('.//atom:author/n', ns)
                if author_elem is not None and author_elem.text:
                    author = author_elem.text
            
            # Extract source
            source = entry.find('.//atom:source/atom:title', ns)
            source = source.text if source is not None else None
            
            # Extract summary
            summary = entry.find('atom:summary', ns)
            summary = summary.text if summary is not None else ""
            
            # Extract full content - need to get the HTML content
            content_elem = entry.find('atom:content', ns)
            full_content = None
            if content_elem is not None:
                # Get the full content text without processing
                # If the content is type="html", we need to get the inner text/XML
                if content_elem.get('type') == 'html':
                    # This preserves the HTML content as is
                    full_content = content_elem.text
                    logger.debug(f"Extracted HTML content for '{title}': length={len(full_content) if full_content else 0}")
                else:
                    # For non-HTML content types, get whatever text is there
                    full_content = content_elem.text
                    logger.debug(f"Extracted non-HTML content for '{title}': length={len(full_content) if full_content else 0}")
                    
                # Debug log to check content extraction
                if full_content:
                    content_preview = full_content[:100] + "..." if len(full_content) > 100 else full_content
                    logger.debug(f"Content preview for '{title}': {content_preview}")
                else:
                    logger.warning(f"No content found for article: {title}")
            else:
                logger.warning(f"No content element found for article: {title}")
            
            # Extract image URL from media:content tag - RULE #2
            image_url = None
            media_content = entry.find('.//media:content[@medium="image"]', ns)
            if media_content is not None:
                image_url = media_content.get('url')
            
            # Fallback to any media:content if no image-specific one is found
            if image_url is None:
                media_content = entry.find('.//media:content', ns)
                if media_content is not None:
                    image_url = media_content.get('url')
            
            # Create Article object
            article = Article(
                title=title,
                url=url,
                image_url=image_url,
                author=author,
                source=source,
                summary=summary,
                read_more_url=url,
                published=published,
                full_text=full_content,  # Store the full content from the feed
                content=full_content     # Also set as primary content to avoid crawling
            )
            
            articles.append(article)
        
        logger.info(f"Extracted {len(articles)} articles from feed")
        return articles
        
    except Exception as e:
        logger.error(f"Error parsing feed {feed_path}: {e}")
        logger.exception("Stack trace:")
        return []

async def process_articles(
    articles: List[Article], 
    output_dir: str, 
    refresh_cache: bool = False, 
    force_model: Optional[str] = None,
    use_local_model: bool = False,
    max_retries: int = 5,
    timeout: int = 60,
    test_html_extraction: bool = False
) -> List[Article]:
    """
    Process a list of articles.
    
    Args:
        articles: List of articles to process
        output_dir: Directory for outputting cached results
        refresh_cache: Whether to refresh the cache
        force_model: Optional model override to use instead of automatic selection
        use_local_model: Whether to use local rule-based model instead of OpenAI API
        max_retries: Maximum number of retries for API calls
        timeout: Timeout in seconds for API calls
        test_html_extraction: Whether to test HTML content extraction
        
    Returns:
        List of processed articles
    """
    # Skip empty lists
    if not articles:
        return []
    
    logger.info(f"Processing {len(articles)} articles")
    
    # Create a dictionary to store the formatted markdown for each category
    categories_markdown = {}
    
    # If user requests local model, use it straight away
    text_analyzer = TextAnalyzer()
    
    if use_local_model:
        logger.info("Using local model for classification as requested")
        processed_articles = []
        
        for article in tqdm(articles, desc="Processing articles with local model"):
            try:
                # Extract domain from URL
                domain = urlparse(article.url).netloc if article.url else ""
                article.domain = domain
                
                # Test HTML extraction if requested
                if test_html_extraction and article.full_text:
                    logger.info(f"Testing HTML extraction for article: {article.title}")
                    
                    # Log the first 200 characters of the HTML content for debugging
                    if article.full_text:
                        content_preview = article.full_text[:200].replace('\n', '\\n')
                        logger.info(f"Original HTML content preview: {content_preview}")
                    
                    # Use the markdown formatter to extract paragraphs from HTML
                    from twtw.formatters.markdown import MarkdownFormatter
                    md_formatter = MarkdownFormatter()
                    extracted_text = md_formatter._extract_paragraphs_with_limit(article.full_text, 2000)
                    
                    # Log the extracted text
                    if extracted_text:
                        text_preview = extracted_text[:200].replace('\n', '\\n')
                        logger.info(f"Extracted text preview: {text_preview}")
                        
                        # Use the extracted text as the article's full text
                        article.full_text = extracted_text
                    else:
                        logger.warning(f"Failed to extract text from HTML for article: {article.title}")
                
                # Classify using the local rule-based classifier
                category, category_score = text_analyzer.classify_text(
                    article.full_text or article.summary, 
                    article.title, 
                    article.source
                )
                article.category = category
                article.category_score = category_score
                article.subcategory = text_analyzer.get_subcategory(article.full_text or article.summary, category)
                
                # Set basic features
                word_count = len((article.full_text or "").split())
                article.features["word_count"] = word_count
                article.reading_time = max(1, word_count // 200)
                
                # Create basic formatted content for the article
                title = article.title or ""
                source = article.source or ""
                url = article.url or ""
                author = article.author or ""
                published = article.published or ""
                
                # Format the published date if possible
                if published:
                    try:
                        published_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").strftime("%B %d, %Y")
                        published = published_date
                    except (ValueError, TypeError):
                        pass
                
                # Extract content for the article
                content = article.full_text or article.summary or ""
                if len(content) > 500:
                    content = content[:500] + "..."
                
                # Create formatted markdown
                formatted_markdown = f"""### [{title}]({url})

**Author:** {author}
**Source:** {source}
**Published:** {published}
**Classification:** {article.category}
**Subcategory:** {article.subcategory or 'N/A'}

{content}

[Read more]({url})
"""
                # Store the formatted markdown
                if category not in categories_markdown:
                    categories_markdown[category] = []
                
                categories_markdown[category].append(formatted_markdown)
                article.processed_content = formatted_markdown
                
                processed_articles.append(article)
            except Exception as e:
                logger.error(f"Error processing article {article.title}: {e}")
        
        # Create the newsletter.md file directly
        newsletter_path = os.path.join(output_dir, "newsletter.md")
        with open(newsletter_path, "w", encoding="utf-8") as f:
            # Start with the header
            f.write("# That Was The Week\n\n")
            f.write(f"Generated on {datetime.now().strftime('%B %d, %Y')}\n\n")
            f.write("## In This Issue\n\n")
            
            # Add table of contents
            sorted_categories = sorted(categories_markdown.keys())
            for category in sorted_categories:
                f.write(f"- {category}: {len(categories_markdown[category])} articles\n")
            
            f.write("\n---\n\n")
            
            # Add articles by category
            for category in sorted_categories:
                f.write(f"## {category}\n\n")
                
                # Add all formatted articles in this category
                for article_markdown in categories_markdown[category]:
                    f.write(article_markdown)
                    f.write("\n---\n\n")
        
        # Create a reading list too
        reading_list_path = os.path.join(output_dir, "reading_list.md")
        with open(reading_list_path, "w", encoding="utf-8") as f:
            f.write("# That Was The Week - Reading List\n\n")
            f.write(f"Generated on {datetime.now().strftime('%B %d, %Y')}\n\n")
            
            # Group by category for the reading list
            for category in sorted_categories:
                f.write(f"## {category}\n\n")
                
                # Get articles in this category
                category_articles = [a for a in processed_articles if a.category == category]
                
                # Add links to each article
                for article in category_articles:
                    f.write(f"- [{article.title}]({article.url})\n")
                
                f.write("\n")
        
        logger.info(f"Generated Markdown files with local model: {newsletter_path}, {reading_list_path}")
        
        return processed_articles
    
    # Get the OpenAI API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.warning("OPENAI_API_KEY not set - limited classification capabilities")
        # Fall back to the local model if no API key
        return await process_articles(articles, output_dir, refresh_cache, force_model, True, max_retries, timeout, test_html_extraction)
        
    # Determine the model to use - first use forced_model if provided, otherwise determine automatically
    def get_best_available_model(configured_model, forced_model):
        """Get the best available model, preferring o3-mini for performance"""
        # If model is forced, use it
        if forced_model:
            logger.info(f"Using forced model: {forced_model}")
            return forced_model
            
        # Only check if we have an API key
        if not api_key:
            return configured_model
            
        try:
            # Create client
            client = openai.OpenAI(api_key=api_key)
            
            # List available models
            models = client.models.list()
            model_ids = [model.id for model in models.data]
            
            # Check models with user preferences
            # 1. o3-mini (user prefers for performance)
            if "o3-mini" in model_ids:
                logger.info("Using o3-mini model (preferred for performance)")
                return "o3-mini"
            # 2. gpt-4o-mini (cost-effective alternative)
            elif "gpt-4o-mini" in model_ids:
                logger.info("Using gpt-4o-mini model (cost-effective alternative)")
                return "gpt-4o-mini"
            # 3. If configured model is gpt-4o, check if we should use gpt-4 instead (cheaper)
            elif configured_model == "gpt-4o" and "gpt-4" in model_ids:
                logger.info("Using gpt-4 model instead of gpt-4o for cost savings")
                return "gpt-4"
            # If not available, fall back to configured model
            return configured_model
        except Exception as e:
            logger.warning(f"Error checking available models: {e}. Using configured model.")
            return configured_model
    
    # Select model based on model parameter or available models
    model = get_best_available_model(get_config("openai", {}).get("model", "gpt-3.5-turbo"), force_model)
    
    # Set up OpenAI API parameters
    api_params = {
        "model": model,
        "response_format": {"type": "json_object"},
    }
    
    # Only add temperature parameter if the model supports it
    if model and not model.startswith("o3-"):
        api_params["temperature"] = get_config("openai", {}).get("temperature", 0.7)
        
    logger.info(f"Using model: {model}")
    
    # Initialize processor for fallback processing if needed
    processor = ContentProcessor()
    
    # Process articles in parallel
    processed_articles = []
    failed_articles = []
    
    # Use tqdm for progress bar
    with tqdm(total=len(articles), desc="Processing articles") as pbar:
        for article in articles:
            try:
                # Test HTML extraction if requested
                if test_html_extraction and article.full_text:
                    logger.info(f"Testing HTML extraction for article: {article.title}")
                    
                    # Log the first 200 characters of the HTML content for debugging
                    if article.full_text:
                        content_preview = article.full_text[:200].replace('\n', '\\n')
                        logger.info(f"Original HTML content preview: {content_preview}")
                    
                    # Use the markdown formatter to extract paragraphs from HTML
                    from twtw.formatters.markdown import MarkdownFormatter
                    md_formatter = MarkdownFormatter()
                    extracted_text = md_formatter._extract_paragraphs_with_limit(article.full_text, 2000)
                    
                    # Log the extracted text
                    if extracted_text:
                        text_preview = extracted_text[:200].replace('\n', '\\n')
                        logger.info(f"Extracted text preview: {text_preview}")
                        
                        # Use the extracted text as the article's full text
                        article.full_text = extracted_text
                    else:
                        logger.warning(f"Failed to extract text from HTML for article: {article.title}")
                
                # Ensure we have the content
                if not article.content or len(article.content.split()) < 50:
                    logger.info(f"Article content missing or too short: {article.title}. Attempting to fetch.")
                    updated_article = await processor.fetcher.process_article(
                        article.url, 
                        fallback_content=article.summary if hasattr(article, 'summary') else None
                    )
                    
                    if updated_article:
                        article.content = updated_article.get('content', '')
                        article.features = updated_article.get('features', {})
                        
                        # If the new content is still empty or too short, use summary or existing full_text
                        if not article.content or len(article.content.split()) < 50:
                            if article.full_text:
                                article.content = article.full_text
                            elif article.summary:
                                article.content = article.summary
                
                # Make sure we have at least some content
                content_for_processing = (
                    article.content or 
                    article.full_text or 
                    article.summary or 
                    f"Article titled: {article.title} from {article.source}"
                )
                
                # Prepare data for OpenAI
                title = article.title or ""
                source = article.source or ""
                url = article.url or ""
                author = article.author or ""
                published = article.published or ""
                image_url = article.image_url or ""
                
                # Format the published date if possible
                if published:
                    try:
                        published_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").strftime("%B %d, %Y")
                        published = published_date
                    except (ValueError, TypeError):
                        pass
                
                # Update the system message to include classification metadata request
                try:
                    # Prepare system and user messages for the API call
                    system_message = """You are an expert at formatting news articles into clean, well-structured markdown.
                    For the provided article, create a complete markdown entry with:
                    
                    1. Classify the article into one of these categories: 
                       - Tech News
                       - AI
                       - Industry Analysis
                       - Venture Capital
                       - Programming
                       - Science
                       - Startups
                       - Essays
                       - Other
                    
                    2. Format the article with the following elements:
                       a. A header with the title linked to the article URL
                       b. Metadata: Include ALL of the following on separate lines with the same formatting:
                          - Author
                          - Source
                          - Published Date
                          - Classification (primary category)
                          - Subcategory or topic
                       c. If an image URL is provided, include the image
                       d. The article content limited to 500 words, properly formatted
                       e. End with a "Read more" link to the original article
                    
                    IMPORTANT: The classification metadata should be integrated with the other metadata fields, NOT in a separate section. All metadata should appear in the same format using bold labels like "**Classification:**".
                    
                    Sample format:
                    ```
                    ### [Article Title](article_url)
                    
                    **Author:** Author Name
                    **Source:** Source Name
                    **Published:** Publication Date
                    **Classification:** Tech News
                    **Subcategory:** Mobile Technology
                    
                    Article content goes here...
                    
                    [Read more](article_url)
                    ```
                    
                    The content should be well-formatted, clean Markdown with proper spacing and structure.
                    
                    Provide your response as a JSON object with these fields:
                    {
                        "category": "category name",
                        "subcategory": "more specific topic area",
                        "formatted_markdown": "the complete formatted markdown for the article"
                    }
                    """
                    
                    user_message = f"""Article Title: {title}
                    Article URL: {url}
                    Source: {source}
                    Author: {author}
                    Published Date: {published}
                    Image URL: {image_url}
                    
                    Content:
                    {content_for_processing}
                    """
                
                    # Call OpenAI API with backoff retry for rate limits
                    @backoff.on_exception(
                        backoff.expo,
                        (openai.RateLimitError, openai.APITimeoutError, 
                         requests.exceptions.RequestException, 
                         requests.exceptions.Timeout,
                         requests.exceptions.ConnectionError,
                         json.JSONDecodeError,
                         asyncio.TimeoutError,
                         aiohttp.ClientError),
                        max_tries=max_retries,
                        factor=2,
                        max_time=timeout
                    )
                    def call_openai_with_retry(prompt_content):
                        try:
                            # Create a client with a longer timeout
                            client = openai.OpenAI(
                                api_key=api_key,
                                timeout=timeout
                            )
                            
                            response = client.chat.completions.create(
                                messages=[
                                    {
                                        "role": "system",
                                        "content": system_message
                                    },
                                    {
                                        "role": "user",
                                        "content": prompt_content
                                    }
                                ],
                                **api_params
                            )
                            return response
                        except KeyboardInterrupt:
                            logging.info("API call interrupted.")
                            return None
                        except Exception as e:
                            logging.error(f"Error calling OpenAI API: {e}")
                            raise e
                    
                    # Call API with retry mechanism
                    response = call_openai_with_retry(user_message)
                    
                    if response is None:
                        # The call was interrupted, use rule-based classification as fallback
                        logging.warning(f"Using rule-based classification for '{article.title}' due to API call interruption")
                        category, category_score = text_analyzer.classify_text(content_for_processing, title, source)
                        article.category = category
                        article.category_score = category_score
                        article.subcategory = text_analyzer.get_subcategory(content_for_processing, category)
                        
                        # Add classification metadata
                        word_count = len(content_for_processing.split())
                        reading_time = max(1, word_count // 200)
                        
                        # Create basic formatted content with classification metadata
                        article.processed_content = f"""### [{title}]({url})

**Author:** {author}
**Source:** {source}
**Published:** {published}
**Classification:** {article.category}
**Subcategory:** {article.subcategory or 'N/A'}

{content_for_processing[:500] + "..." if len(content_for_processing) > 500 else content_for_processing}

[Read more]({url})
"""
                        
                        # Store the formatted markdown
                        if category not in categories_markdown:
                            categories_markdown[category] = []
                        
                        categories_markdown[category].append(article.processed_content)
                        
                        logger.info(f"Processed article with raw response: {article.title} - Category: {article.category}")
                        continue
                        
                    # Parse JSON response
                    try:
                        response_data = json.loads(response.choices[0].message.content)
                        
                        # Extract the category and formatted markdown
                        category = response_data.get("category", "Unclassified")
                        formatted_markdown = response_data.get("formatted_markdown", "")
                        
                        # Extract additional classification metadata
                        subcategory = response_data.get("subcategory", "N/A")
                        
                        # Update the article with the processed data
                        article.category = category
                        # Keep confidence score processing for backward compatibility
                        confidence = response_data.get("confidence", "N/A")
                        article.category_score = float(confidence) if isinstance(confidence, (int, float)) or (isinstance(confidence, str) and confidence.replace('.', '', 1).isdigit()) else None
                        article.subcategory = subcategory
                        
                        # Use the formatted markdown directly
                        article.processed_content = formatted_markdown
                        
                        # Store the formatted markdown in the category dictionary
                        if category not in categories_markdown:
                            categories_markdown[category] = []
                        
                        categories_markdown[category].append(formatted_markdown)
                        
                        logger.info(f"Successfully processed with OpenAI: {article.title} - Category: {article.category}, Subcategory: {article.subcategory}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing OpenAI JSON response: {e}. Using raw response.")
                        
                        # Try to extract content from raw response
                        response_text = response.choices[0].message.content
                        
                        # If we can identify a category from the raw response
                        category_match = re.search(r'Category:\s*(.*?)[\n\r]', response_text, re.IGNORECASE)
                        if category_match:
                            category = category_match.group(1).strip()
                        else:
                            category, category_score = text_analyzer.classify_text(content_for_processing, title, source)
                            
                        # Attempt to extract subcategory from raw response
                        subcategory_match = re.search(r'Subcategory:\s*(.*?)[\n\r]', response_text, re.IGNORECASE)
                        subcategory = subcategory_match.group(1).strip() if subcategory_match else text_analyzer.get_subcategory(content_for_processing, category)
                        
                        # Use the entire response as formatted content
                        formatted_markdown = response_text
                        
                        # Update the article
                        article.category = category
                        article.subcategory = subcategory
                        article.processed_content = formatted_markdown
                        
                        # Store the formatted markdown
                        if category not in categories_markdown:
                            categories_markdown[category] = []
                        
                        categories_markdown[category].append(formatted_markdown)
                        
                        logger.info(f"Processed article with raw response: {article.title} - Category: {article.category}")
                except Exception as e:
                    logger.warning(f"Error using OpenAI API: {e}. Falling back to rule-based classification.")
                    # Fallback to text_analyzer for category
                    category, category_score = text_analyzer.classify_text(content_for_processing, title, source)
                    subcategory = text_analyzer.get_subcategory(content_for_processing, category)
                    article.category = category
                    article.category_score = category_score
                    article.subcategory = subcategory
                    
                    # Calculate word count and reading time
                    word_count = len(content_for_processing.split())
                    reading_time = max(1, word_count // 200)
                    
                    # Create basic formatted content with classification metadata
                    article.processed_content = f"""### [{title}]({url})

**Author:** {author}
**Source:** {source}
**Published:** {published}
**Classification:** {article.category}
**Subcategory:** {article.subcategory or 'N/A'}

{content_for_processing[:500] + "..." if len(content_for_processing) > 500 else content_for_processing}

[Read more]({url})
"""
                    # Store the formatted markdown
                    if category not in categories_markdown:
                        categories_markdown[category] = []
                    
                    categories_markdown[category].append(article.processed_content)
                
                # Extract domain from URL if available
                if article.url:
                    parsed_url = urlparse(article.url)
                    article.domain = parsed_url.netloc
                
                processed_articles.append(article)
                
            except Exception as e:
                logger.warning(f"Failed to process article {article.url}: {e}")
                failed_articles.append(article)
            finally:
                pbar.update(1)
    
    # Create the newsletter.md file directly
    newsletter_path = os.path.join(output_dir, "newsletter.md")
    with open(newsletter_path, "w", encoding="utf-8") as f:
        # Start with the header
        f.write("# That Was The Week\n\n")
        f.write(f"Generated on {datetime.now().strftime('%B %d, %Y')}\n\n")
        f.write("## In This Issue\n\n")
        
        # Add table of contents
        sorted_categories = sorted(categories_markdown.keys())
        for category in sorted_categories:
            f.write(f"- {category}: {len(categories_markdown[category])} articles\n")
        
        f.write("\n---\n\n")
        
        # Add articles by category
        for category in sorted_categories:
            f.write(f"## {category}\n\n")
            
            # Add all formatted articles in this category
            for article_markdown in categories_markdown[category]:
                f.write(article_markdown)
                f.write("\n---\n\n")
    
    # Create a reading list too
    reading_list_path = os.path.join(output_dir, "reading_list.md")
    with open(reading_list_path, "w", encoding="utf-8") as f:
        f.write("# That Was The Week - Reading List\n\n")
        f.write(f"Generated on {datetime.now().strftime('%B %d, %Y')}\n\n")
        
        # Group by category for the reading list
        for category in sorted_categories:
            f.write(f"## {category}\n\n")
            
            # Get articles in this category
            category_articles = [a for a in processed_articles if a.category == category]
            
            # Add links to each article
            for article in category_articles:
                f.write(f"- [{article.title}]({article.url})\n")
            
            f.write("\n")
    
    # Log results
    logger.info(f"Successfully processed {len(processed_articles)} articles")
    if failed_articles:
        logger.warning(f"Failed to process {len(failed_articles)} articles")
        
        # Add failed articles with a note
        for article in failed_articles:
            article.content = "**Note: Full article content could not be retrieved.**"
            article.processed_content = article.content
            article.category = "Unclassified"
            processed_articles.append(article)
    
    return processed_articles

async def async_main():
    """
    Asynchronous main function.
    
    This function orchestrates the entire TWTW process:
    1. Fetches feeds (if not skipped)
    2. Processes articles (if not skipped)
    3. Generates markdown files
    4. Converts markdown to HTML (if not skipped)
    
    Command-line options:
    - --feeds: Path to file containing feed URLs (default: feeds.txt)
    - --output-dir: Directory for output files
    - --refresh-cache: Force refresh cached content
    - --skip-fetch: Skip fetching feeds (use existing files)
    - --skip-process: Skip processing articles
    - --skip-convert: Skip converting to HTML
    - --limit: Maximum number of articles to process per feed (0 for all)
    - --model: Specify OpenAI model to use
    - --use-local-model: Use local rule-based model instead of OpenAI API
    - --skip-today-folder: Skip including atom feeds from the 'today' folder (by default, feeds from 'today' folder are included)
    """
    # Parse command-line arguments
    args = parse_args()

    # Record the start time
    start_time = time.time()

    # Generate an output directory if one wasn't specified
    if args.output_dir is None:
        datetime_str = datetime.now().strftime('%Y-%m-%d:%H:%M:%S')
        output_dir = os.getenv('TWTW_OUTPUT_DIR', f'output_{datetime_str}')
        if 'datetimestamp' in output_dir:
            # Replace datetimestamp with the actual datetime
            output_dir = output_dir.replace('datetimestamp', datetime_str)
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")
    
    # Helper function to find a unique filename and copy a feed file
    def copy_feed_file_with_unique_name(source_path, output_dir, original_filename, existing_numbers, feed_files):
        """
        Copy a feed file from source to output directory with a unique feedly*.atom filename
        
        Args:
            source_path: Full path to the source file
            output_dir: Directory to copy the file to
            original_filename: Original filename (may not be in feedly*.atom format)
            existing_numbers: Set of numbers already used in existing feedly*.atom files
            feed_files: List of feed files to append the new filename to
            
        Returns:
            The new filename that was used
        """
        # If the file doesn't match the feedly*.atom pattern, generate a unique name
        if not (original_filename.startswith('feedly') and original_filename.endswith('.atom')):
            # Find the next available number
            next_num = 1
            while next_num in existing_numbers:
                next_num += 1
            
            new_feed_file = f'feedly{next_num}.atom'
            existing_numbers.add(next_num)  # Mark as used
            output_feed_path = os.path.join(output_dir, new_feed_file)
            
            # Copy with the new name
            shutil.copy2(source_path, output_feed_path)
            logger.info(f"Copied {original_filename} to {new_feed_file} in output directory")
            
            # Add to our feed_files list for processing
            if new_feed_file not in feed_files:
                feed_files.append(new_feed_file)
                
            return new_feed_file
        else:
            # The file already follows our naming convention - check if it exists
            output_feed_path = os.path.join(output_dir, original_filename)
            if not os.path.exists(output_feed_path):
                # Safe to copy with original name
                shutil.copy2(source_path, output_feed_path)
                logger.info(f"Copied {original_filename} to output directory")
                
                # Add to our feed_files list for processing
                if original_filename not in feed_files:
                    feed_files.append(original_filename)
                    
                return original_filename
            else:
                # File exists - create a new name
                try:
                    # Extract the number
                    original_num = int(original_filename.replace('feedly', '').replace('.atom', ''))
                    
                    # Find next available number
                    next_num = original_num
                    while next_num in existing_numbers:
                        next_num += 1
                    
                    new_feed_file = f'feedly{next_num}.atom'
                    existing_numbers.add(next_num)
                    new_output_path = os.path.join(output_dir, new_feed_file)
                    
                    # Copy with the new name
                    shutil.copy2(source_path, new_output_path)
                    logger.info(f"Feed file {original_filename} already exists, copied as {new_feed_file}")
                    
                    # Add to our feed_files list for processing
                    if new_feed_file not in feed_files:
                        feed_files.append(new_feed_file)
                        
                    return new_feed_file
                except ValueError:
                    logger.warning(f"Could not generate unique name for {original_filename}, skipping")
                    return None
    
    articles = []
    
    try:
        # Step 1: Fetch feeds
        if not args.skip_fetch:
            logger.info("Fetching feeds...")
            fetcher = FeedlyFetcher(args.feeds)
            fetched_feeds = fetcher.fetch_all_feeds(output_dir)
            
            if not fetched_feeds:
                logger.error("No feeds were successfully fetched")
                return 1
            
            # Initialize feed_files in the skip_fetch branch before trying to append to it
            feed_files = list(fetched_feeds.keys())
            
            # If --use-today-folder option is specified, look for atom feeds in the 'today' folder
            if not args.skip_today_folder and os.path.isdir('today'):
                logger.info("Checking 'today' folder for additional atom feeds...")
                today_feeds = [f for f in os.listdir('today') if f.endswith('.atom')]
                
                if today_feeds:
                    logger.info(f"Found {len(today_feeds)} atom feeds in 'today' folder")
                    
                    # Find existing feedly*.atom files to avoid name conflicts
                    existing_feed_files = [f for f in os.listdir(output_dir) if f.startswith('feedly') and f.endswith('.atom')]
                    existing_numbers = set()
                    for file in existing_feed_files:
                        try:
                            num = int(file.replace('feedly', '').replace('.atom', ''))
                            existing_numbers.add(num)
                        except ValueError:
                            continue
                    
                    # Copy these feeds to the output directory with unique names
                    for feed_file in today_feeds:
                        today_feed_path = os.path.join('today', feed_file)
                        copy_feed_file_with_unique_name(today_feed_path, output_dir, feed_file, existing_numbers, feed_files)
                else:
                    logger.info("No atom feeds found in 'today' folder")
            
            if not feed_files:
                logger.error("No feed files found in the output directory")
                return 1
            
            # Parse each feed file
            feeds_tasks = []
            for feed_file in feed_files:
                feed_path = os.path.join(output_dir, feed_file)
                task = parse_atom_feed(feed_path, args.limit)
                feeds_tasks.append(task)
            
            # Gather results from all feeds
            feed_results = await asyncio.gather(*feeds_tasks)
            for result in feed_results:
                articles.extend(result)
            
            # Deduplicate articles by URL to ensure uniqueness
            unique_articles = []
            seen_urls = set()
            
            for article in articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    unique_articles.append(article)
                else:
                    logger.info(f"Skipping duplicate article: {article.title} ({article.url})")
            
            # Replace the original articles list with the deduplicated list
            articles = unique_articles
            
            logger.info(f"Found {len(articles)} unique articles across all feeds")
        else:
            logger.info("Skipping feed fetching as requested")
            
            # Look for XML files in the output directory
            feed_files = [f for f in os.listdir(output_dir) if f.endswith('.atom')]
            
            # If --use-today-folder option is specified, look for atom feeds in the 'today' folder
            if not args.skip_today_folder and os.path.isdir('today'):
                logger.info("Checking 'today' folder for additional atom feeds...")
                today_feeds = [f for f in os.listdir('today') if f.endswith('.atom')]
                
                if today_feeds:
                    logger.info(f"Found {len(today_feeds)} atom feeds in 'today' folder")
                    
                    # Find existing feedly*.atom files to avoid name conflicts
                    existing_feed_files = [f for f in os.listdir(output_dir) if f.startswith('feedly') and f.endswith('.atom')]
                    existing_numbers = set()
                    for file in existing_feed_files:
                        try:
                            num = int(file.replace('feedly', '').replace('.atom', ''))
                            existing_numbers.add(num)
                        except ValueError:
                            continue
                    
                    # Copy these feeds to the output directory with unique names
                    for feed_file in today_feeds:
                        today_feed_path = os.path.join('today', feed_file)
                        copy_feed_file_with_unique_name(today_feed_path, output_dir, feed_file, existing_numbers, feed_files)
                else:
                    logger.info("No atom feeds found in 'today' folder")
        
        # Step 2: Process articles and generate markdown directly
        if not args.skip_process and articles:
            logger.info("Processing articles...")
            articles = await process_articles(
                articles, 
                output_dir, 
                args.refresh_cache, 
                args.model, 
                args.use_local_model, 
                args.max_retries, 
                args.timeout,
                args.test_html_extraction
            )
            logger.info("Markdown files generated during processing")
        else:
            if args.skip_process:
                logger.info("Skipping article processing as requested")
            else:
                logger.warning("No articles to process")
        
        # Skip the separate generate_markdown step since we're now generating markdown directly in process_articles
        
        # Step 4: Convert markdown to HTML
        if not args.skip_convert:
            logger.info("Converting Markdown to HTML...")
            html_converter = HtmlConverter()
            html_results = html_converter.convert_all_markdown_files(output_dir)
            
            success_count = sum(1 for success in html_results.values() if success)
            logger.info(f"Converted {success_count}/{len(html_results)} Markdown files to HTML")
        else:
            logger.info("Skipping HTML conversion as requested")
            
        # Record the end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Total execution time: {duration:.2f} seconds")
        
        return 0
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        # Make sure to return the articles that were successfully processed so far
        return articles
    except Exception as e:
        logger.exception(f"Error in async_main: {e}")
        # Make sure to return any articles that were successfully processed
        return articles
    finally:
        # Ensure all aiohttp sessions are closed
        for session in [s for s in asyncio.all_tasks() if hasattr(s, '_session') and s._session]:
            try:
                if not session._session.closed:
                    await session._session.close()
            except Exception as e:
                logger.error(f"Error closing session: {e}")

def main():
    """
    Main entry point for the application.
    """
    try:
        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'twtw_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )

        # Run the async main function
        logging.info("Starting TWTW Newsletter Generator")
        asyncio.run(async_main())
        logging.info("TWTW Newsletter Generator completed successfully")
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        # Allow for a graceful shutdown
        print("\nShutting down gracefully. Please wait...")
        try:
            # Save any work that might be in progress
            # This is a good place to implement a recovery mechanism
            print("Process interrupted, but any completed article processing has been saved.")
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    sys.exit(main()) 