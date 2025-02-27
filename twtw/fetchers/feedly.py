"""
Feedly feed fetcher for TWTW.
"""
import os
import datetime
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urljoin
import requests
import logging

# Configure logging
logger = logging.getLogger(__name__)

class FeedlyFetcher:
    """
    Fetches feeds from Feedly.
    """
    def __init__(self, feeds_file: str = 'feeds.txt'):
        """
        Initialize the FeedlyFetcher.
        
        Args:
            feeds_file: Path to the file containing feed URLs
        """
        self.feeds_file = feeds_file
        self.enterprise_token = os.getenv('FEEDLY_ENTERPRISE_TOKEN')
        
    def read_feeds(self) -> Dict[str, str]:
        """
        Read feed URLs from a text file.
        
        Returns:
            Dict mapping feed indexes to feed URLs
        """
        feeds = {}
        try:
            with open(self.feeds_file, 'r') as f:
                for i, line in enumerate(f, 1):
                    url = line.strip()
                    if url and not url.startswith('#'):  # Skip empty lines and comments
                        feeds[i] = url
            return feeds
        except FileNotFoundError:
            logger.error(f"Error: {self.feeds_file} not found. Please create it with one feed URL per line.")
            return {}

    def fetch_feed(self, feed_url: str, output_filename: str) -> bool:
        """
        Fetch and save a single feed.
        
        Args:
            feed_url: The URL of the feed to fetch
            output_filename: The filename to save the feed content to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Set up headers for enterprise authentication if needed
            headers = {}
            if 'cloud.feedly.com/v3/enterprise' in feed_url:
                if not self.enterprise_token:
                    logger.error("Error: FEEDLY_ENTERPRISE_TOKEN environment variable not set")
                    return False
                headers['Authorization'] = f'Bearer {self.enterprise_token}'

            # Fetch the feed content without following redirects
            response = requests.get(feed_url, headers=headers, allow_redirects=False)
            if response.is_redirect or response.status_code in (301, 302):
                # Manually follow the redirect to get the final content
                redirect_url = response.headers.get('Location')
                if not redirect_url.startswith('http'):
                    # If the redirect URL is relative, construct the full URL
                    redirect_url = urljoin(feed_url, redirect_url)
                # Fetch the final content with the same headers
                response = requests.get(redirect_url, headers=headers)
                response.raise_for_status()
                feed_content = response.text
            else:
                response.raise_for_status()
                feed_content = response.text

            # Save the content to a file with the desired filename
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(feed_content)

            logger.info(f"Feed successfully saved to {output_filename}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred while fetching the feed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                if e.response.status_code == 401:
                    logger.error("Authentication failed. Please check your FEEDLY_ENTERPRISE_TOKEN")
            return False

    def fetch_all_feeds(self, output_directory: Optional[str] = None) -> Dict[str, str]:
        """
        Fetch all feeds and save them to the output directory.
        
        Args:
            output_directory: The directory to save the feeds to. If None, a dated directory will be created.
            
        Returns:
            Dict mapping filenames to feed URLs that were successfully fetched
        """
        # Read feeds from the configuration file
        feeds = self.read_feeds()
        if not feeds:
            logger.warning("No feeds found to process")
            return {}

        # Generate the output directory with the current date if not provided
        if output_directory is None:
            # Format: YYYY-MM-DD:HH:MM:SS
            datetime_str = datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S')
            output_directory = os.getenv('TWTW_OUTPUT_DIR', f'output_{datetime_str}')
        
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Get existing feed files in the output directory
        existing_feed_files = [f for f in os.listdir(output_directory) if f.startswith('feedly') and f.endswith('.atom')]
        
        # Extract existing numbers
        existing_numbers = set()
        for file in existing_feed_files:
            # Extract the number between "feedly" and ".atom"
            try:
                num = int(file.replace('feedly', '').replace('.atom', ''))
                existing_numbers.add(num)
            except ValueError:
                continue

        # Fetch each feed with unique filenames
        logger.info(f"Found {len(feeds)} feeds to process...")
        successful_feeds = {}
        
        for index, url in feeds.items():
            # Find the next available number
            next_num = index
            while next_num in existing_numbers:
                next_num += 1
            
            # Create filename
            filename = f'feedly{next_num}.atom'
            existing_numbers.add(next_num)  # Mark this number as used
            
            logger.info(f"Processing feed: {url}")
            output_file_path = os.path.join(output_directory, filename)
            if self.fetch_feed(url, output_file_path):
                successful_feeds[filename] = url
                
        return successful_feeds


def main():
    """
    Main entry point for the feedly fetcher.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the fetcher
    fetcher = FeedlyFetcher()
    fetcher.fetch_all_feeds()


if __name__ == '__main__':
    main() 