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
import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from twtw.fetchers.feedly import FeedlyFetcher
from twtw.formatters.html import HtmlConverter
from twtw.formatters.markdown import MarkdownFormatter
from twtw.core.processor import ContentProcessor
from twtw.core.article import Article
from twtw.utils.nlp import TextAnalyzer

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
    parser = argparse.ArgumentParser(description="That Was The Week - Newsletter Generator")
    parser.add_argument("--feeds", help="Path to feeds file", default="feeds.txt")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--refresh-cache", action="store_true", help="Force refresh cache")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching feeds")
    parser.add_argument("--skip-process", action="store_true", help="Skip processing articles")
    parser.add_argument("--skip-convert", action="store_true", help="Skip converting to HTML")
    parser.add_argument("--limit", type=int, help="Limit number of articles to process (0 = no limit)", default=0)
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
                # Check if it's HTML content
                if content_elem.get('type') == 'html':
                    # Get the full HTML content and ensure it's properly parsed
                    full_content = content_elem.text
                    # Wrap the content in a div to ensure proper parsing
                    if full_content and not full_content.strip().startswith('<'):
                        full_content = f'<div>{full_content}</div>'
                else:
                    # Just get the text
                    full_content = content_elem.text
                    
                # Clean up any potential issues with the content
                if full_content:
                    # Remove any duplicate "Listen to this post:" prefixes
                    full_content = full_content.replace("Listen to this post: Listen to this post:", "Listen to this post:")
            
            # Extract image URL from media:content tag
            image_url = None
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
                full_text=full_content
            )
            
            articles.append(article)
        
        logger.info(f"Extracted {len(articles)} articles from feed")
        return articles
        
    except Exception as e:
        logger.error(f"Error parsing feed {feed_path}: {e}")
        return []

async def process_articles(articles: List[Article], output_dir: str, refresh_cache: bool = False) -> List[Article]:
    """
    Process articles to extract content and features.
    
    Args:
        articles: List of articles to process
        output_dir: Output directory for cache
        refresh_cache: Whether to refresh the cache
        
    Returns:
        List of processed articles
    """
    logger.info(f"Processing {len(articles)} articles...")
    
    # Initialize processor with default parameters
    processor = ContentProcessor()
    
    # Process articles in parallel
    processed_articles = []
    failed_articles = []
    
    # Use tqdm for progress bar
    with tqdm.tqdm(total=len(articles), desc="Processing articles") as pbar:
        for article in articles:
            try:
                # Process article
                updated_article = await processor.fetcher.process_article(
                    article.url, 
                    fallback_content=article.summary if hasattr(article, 'summary') else None
                )
                
                if updated_article:
                    # Update article with content and features
                    article.content = updated_article.get('content', '')
                    article.features = updated_article.get('features', {})
                    article.images = updated_article.get('images', [])
                    
                    # Check if article is paywalled
                    if 'features' in updated_article and updated_article['features'].get('paywall_detected'):
                        article.is_paywalled = True
                    
                    # Classify article
                    title = article.title or ""
                    content = article.content or ""
                    source = article.source or ""
                    
                    # Use the improved classification system
                    category, scores = text_analyzer.classify_text(content, title, source)
                    article.category = category
                    
                    # Get subcategory if available
                    subcategory = text_analyzer.get_subcategory(content + " " + title, category)
                    if subcategory:
                        article.subcategory = subcategory
                    
                    # Calculate reading time
                    word_count = len(content.split())
                    article.reading_time = max(1, word_count // 200)  # Assume 200 words per minute
                    
                    # Extract domain from URL
                    if article.url:
                        from urllib.parse import urlparse
                        parsed_url = urlparse(article.url)
                        article.domain = parsed_url.netloc
                    
                    processed_articles.append(article)
                    logger.debug(f"Processed article: {article.title} - Category: {article.category}")
                else:
                    # If processing failed, add to failed articles
                    logger.warning(f"Failed to process article {article.url}: No content extracted")
                    failed_articles.append(article)
            except Exception as e:
                logger.warning(f"Failed to process article {article.url}: {e}")
                failed_articles.append(article)
            finally:
                pbar.update(1)
    
    # Log results
    logger.info(f"Successfully processed {len(processed_articles)} articles")
    if failed_articles:
        logger.warning(f"Failed to process {len(failed_articles)} articles")
        
        # Add failed articles with a note
        for article in failed_articles:
            article.content = "**Note: Full article content could not be retrieved.**"
            article.category = "Unclassified"
            article.is_paywalled = True  # Assume paywalled if we couldn't process it
            processed_articles.append(article)
    
    return processed_articles

async def generate_markdown(articles: List[Article], output_dir: str) -> Dict[str, str]:
    """
    Generate Markdown files for the processed articles.
    
    Args:
        articles: List of processed articles
        output_dir: Output directory for Markdown files
        
    Returns:
        Dict of generated file paths
    """
    logger.info("Generating Markdown files...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize formatter
    formatter = MarkdownFormatter()
    
    # Group articles by category
    categorized_articles = {}
    for article in articles:
        category = article.category
        if category not in categorized_articles:
            categorized_articles[category] = []
        categorized_articles[category].append(article)
    
    # Sort categories
    sorted_categories = sorted(categorized_articles.keys())
    
    # Generate newsletter
    newsletter_path = os.path.join(output_dir, "newsletter.md")
    with open(newsletter_path, "w", encoding="utf-8") as f:
        # Generate newsletter content
        newsletter_content = formatter.format_newsletter(
            articles=articles,
            categorized_articles=categorized_articles,
            sorted_categories=sorted_categories
        )
        f.write(newsletter_content)
    
    # Generate reading list
    reading_list_path = os.path.join(output_dir, "reading_list.md")
    with open(reading_list_path, "w", encoding="utf-8") as f:
        # Generate reading list content
        reading_list_content = formatter.format_reading_list(articles)
        f.write(reading_list_content)
    
    # Log results
    logger.info(f"Generated Markdown files: {newsletter_path}, {reading_list_path}")
    
    return {
        "newsletter": newsletter_path,
        "reading_list": reading_list_path
    }

async def async_main():
    """
    Main entry point for the application.
    """
    # Explicitly reload environment variables from .env file
    load_dotenv(override=True)
    
    # Parse command-line arguments
    args = parse_args()
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Format: YYYY-MM-DD:HH:MM:SS
        datetime_str = datetime.now().strftime('%Y-%m-%d:%H:%M:%S')
        env_output_dir = os.getenv('TWTW_OUTPUT_DIR', f'output_{datetime_str}')
        
        # Check if the environment variable contains the special placeholder
        if 'datetimestamp' in env_output_dir:
            # Replace the placeholder with the actual timestamp
            output_dir = env_output_dir.replace('datetimestamp', datetime_str)
        else:
            output_dir = env_output_dir
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set environment variable for other components
    os.environ['TWTW_OUTPUT_DIR'] = output_dir
    
    logger.info(f"Starting TWTW Newsletter Generator")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Fetch feeds
    if not args.skip_fetch:
        logger.info("Fetching feeds...")
        fetcher = FeedlyFetcher(feeds_file=args.feeds)
        successful_feeds = fetcher.fetch_all_feeds(output_dir)
        
        if not successful_feeds:
            logger.error("No feeds were successfully fetched. Exiting.")
            return 1
        
        logger.info(f"Successfully fetched {len(successful_feeds)} feeds")
    else:
        logger.info("Skipping feed fetching")
        # Find existing feed files
        successful_feeds = {}
        for file in os.listdir(output_dir):
            if file.endswith('.atom'):
                successful_feeds[file] = "existing"
    
    # Step 2: Process articles
    if not args.skip_process:
        logger.info("Processing articles...")
        
        # Parse feeds and extract articles
        all_articles = []
        for feed_file in successful_feeds:
            feed_path = os.path.join(output_dir, feed_file)
            articles = await parse_atom_feed(feed_path, limit=args.limit)
            all_articles.extend(articles)
        
        if not all_articles:
            logger.warning("No articles found in feeds")
        else:
            # Process articles
            processed_articles = await process_articles(all_articles, output_dir, args.refresh_cache)
            
            # Generate Markdown files
            if processed_articles:
                markdown_files = await generate_markdown(processed_articles, output_dir)
                logger.info(f"Generated {len(markdown_files)} Markdown files")
            else:
                logger.warning("No articles were successfully processed")
    else:
        logger.info("Skipping article processing")
    
    # Step 3: Convert to HTML
    if not args.skip_convert:
        logger.info("Converting to HTML...")
        converter = HtmlConverter()
        results = converter.convert_all_markdown_files(output_dir)
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Converted {success_count}/{len(results)} Markdown files to HTML")
    else:
        logger.info("Skipping HTML conversion")
    
    logger.info("TWTW Newsletter Generator completed successfully")
    return 0

def main():
    """
    Entry point for the command-line script.
    """
    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 