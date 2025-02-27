"""
Article processing functionality for TWTW.
"""
import asyncio
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import backoff
import aiohttp
import async_timeout
from bs4 import BeautifulSoup
import trafilatura
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from twtw.core.cache import CacheManager
from twtw.utils.http import RateLimiter, REQUEST_TIMEOUT, MAX_CONCURRENT_REQUESTS

class ArticleFetcher:
    """
    Fetches and processes article content from URLs.
    """
    def __init__(self):
        self.cache = CacheManager()
        self.rate_limiter = RateLimiter()
        self._session = None
        self.processed_urls = set()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
        }

    @property
    def session(self):
        """
        Lazy initialization of aiohttp session.
        
        Returns:
            aiohttp.ClientSession: The HTTP session
        """
        if self._session is None:
            self._session = aiohttp.ClientSession(headers=self.headers)
        return self._session

    async def close_session(self):
        """Close aiohttp session."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3
    )
    async def fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch URL content with retries and timeout.
        
        Args:
            url: The URL to fetch
            
        Returns:
            The HTML content as a string, or None if the fetch failed
        """
        domain = urlparse(url).netloc
        await self.rate_limiter.acquire(domain)
        
        try:
            async with async_timeout.timeout(REQUEST_TIMEOUT):
                async with self.session.get(url) as response:
                    response.raise_for_status()
                    content = await response.text()
                    # Report success to rate limiter
                    self.rate_limiter.report_success(domain)
                    return content
        except (aiohttp.ClientError, asyncio.TimeoutError, asyncio.CancelledError) as e:
            # Report failure to rate limiter
            self.rate_limiter.report_failure(domain)
            print(f"Error fetching {url}: {e}")
            return None
        except Exception as e:
            # Report failure to rate limiter
            self.rate_limiter.report_failure(domain)
            print(f"Unexpected error fetching {url}: {e}")
            return None

    async def process_article(self, url: str, fallback_content: Optional[str] = None) -> Optional[Dict]:
        """
        Process article URL with caching and fallback content handling.
        
        Args:
            url: The URL of the article to process
            fallback_content: Optional fallback content if URL fetch fails
            
        Returns:
            Dict containing content and features if successful, None otherwise
        """
        # Check cache first
        cached = self.cache.get(url)
        if cached:
            return cached

        # If there is no fallback content, fetch from URL
        content = None
        if not fallback_content:
            content = await self.fetch_url(url)
        
        # If URL fetch fails, use fallback content from feed (if available)
        if not content and fallback_content:
            content = fallback_content

        if not content:
            return None

        try:
            # Extract main content using trafilatura
            downloaded = trafilatura.extract(
                content,
                include_comments=False,
                include_tables=True,
                include_images=True,
                include_links=True,
                favor_recall=True,
                output_format='xml'  # Use XML to preserve images and formatting
            )

            # Extract images from the content
            images = []
            seen_image_urls = set()  # Track seen image URLs to avoid duplicates
            
            # Initialize extracted_text to hold our content in case image extraction fails
            extracted_text = None
            
            if downloaded:
                # First convert XML to markdown, so we have content even if image extraction fails
                extracted_text = trafilatura.extract(
                    downloaded,
                    include_images=True,
                    output_format='markdown'
                )
                
                try:
                    # Parse the XML to extract images - avoid importing BeautifulSoup here, we already have it
                    soup = BeautifulSoup(downloaded, 'xml')
                    img_tags = soup.find_all('img')
                    
                    # Process images in order of appearance
                    for img in img_tags:
                        if img.get('src'):
                            img_url = img.get('src')
                            
                            # Skip if we've already seen this image URL
                            if img_url in seen_image_urls:
                                continue
                                
                            # Add to seen set
                            seen_image_urls.add(img_url)
                            
                            # Skip tiny images (likely icons)
                            width = img.get('width', '')
                            height = img.get('height', '')
                            
                            try:
                                w = int(width) if width else 0
                                h = int(height) if height else 0
                                if w > 0 and h > 0 and w < 50 and h < 50:
                                    continue  # Skip tiny images
                            except ValueError:
                                pass  # If we can't parse dimensions, include the image
                            
                            # Add the image
                            images.append({
                                'src': img_url,
                                'alt': img.get('alt', ''),
                                'width': width,
                                'height': height
                            })
                    
                except Exception as e:
                    print(f"Error extracting images from XML: {e}")
                    # Continue with the extracted text we already have

            if not extracted_text:
                # Fallback to BeautifulSoup if trafilatura fails
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove unwanted elements
                for elem in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    elem.decompose()
                
                # Try to find main content
                article = (
                    soup.find('article') or
                    soup.find(class_=lambda x: x and any(
                        term in str(x).lower() for term in ['article', 'content', 'post', 'entry']
                    ))
                )
                
                # Extract images
                if article:
                    img_tags = article.find_all('img')
                    for img in img_tags:
                        if img.get('src'):
                            img_url = img.get('src')
                            
                            # Skip if we've already seen this image URL
                            if img_url in seen_image_urls:
                                continue
                                
                            # Add to seen set
                            seen_image_urls.add(img_url)
                            
                            # Skip tiny images (likely icons)
                            width = img.get('width', '')
                            height = img.get('height', '')
                            
                            try:
                                w = int(width) if width else 0
                                h = int(height) if height else 0
                                if w > 0 and h > 0 and w < 50 and h < 50:
                                    continue  # Skip tiny images
                            except ValueError:
                                pass  # If we can't parse dimensions, include the image
                            
                            # Add the image
                            images.append({
                                'src': img_url,
                                'alt': img.get('alt', ''),
                                'width': width,
                                'height': height
                            })
                    
                    # Extract text content as markdown-friendly format rather than raw HTML
                    paragraphs = []
                    for p in article.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                        paragraphs.append(p.get_text(strip=True))
                    
                    if paragraphs:
                        extracted_text = "\n\n".join(paragraphs)
                    else:
                        extracted_text = article.get_text(separator=' ', strip=True)
                else:
                    # Fallback to body content
                    img_tags = soup.find_all('img')
                    for img in img_tags[:5]:  # Limit to first 5 images
                        if img.get('src'):
                            img_url = img.get('src')
                            
                            # Skip if we've already seen this image URL
                            if img_url in seen_image_urls:
                                continue
                                
                            # Add to seen set
                            seen_image_urls.add(img_url)
                            
                            # Skip tiny images (likely icons)
                            width = img.get('width', '')
                            height = img.get('height', '')
                            
                            try:
                                w = int(width) if width else 0
                                h = int(height) if height else 0
                                if w > 0 and h > 0 and w < 50 and h < 50:
                                    continue  # Skip tiny images
                            except ValueError:
                                pass  # If we can't parse dimensions, include the image
                            
                            # Add the image
                            images.append({
                                'src': img_url,
                                'alt': img.get('alt', ''),
                                'width': width,
                                'height': height
                            })
                    
                    # Extract paragraphs from the body
                    paragraphs = []
                    for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                        if p.get_text(strip=True):  # Skip empty paragraphs
                            paragraphs.append(p.get_text(strip=True))
                    
                    if paragraphs:
                        extracted_text = "\n\n".join(paragraphs)
                    else:
                        # Last resort: Use soup.get_text but clean it up
                        raw_text = soup.get_text()
                        # Split by newlines and re-join non-empty lines
                        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
                        extracted_text = "\n\n".join(lines)
            
            # Use the extracted text - make sure we're not including raw HTML
            if extracted_text:
                # Clean up the text - remove any raw HTML-like content that might have been included
                text = re.sub(r'<\s*!DOCTYPE.*?>', '', extracted_text, flags=re.IGNORECASE | re.DOTALL)
                text = re.sub(r'<html.*?>.*?</html>', '', text, flags=re.IGNORECASE | re.DOTALL)
                text = re.sub(r'<\s*meta.*?>', '', text, flags=re.IGNORECASE)
                text = re.sub(r'<\s*link.*?>', '', text, flags=re.IGNORECASE)
                text = re.sub(r'<\s*script.*?>.*?</\s*script\s*>', '', text, flags=re.IGNORECASE | re.DOTALL)
                text = re.sub(r'<\s*style.*?>.*?</\s*style\s*>', '', text, flags=re.IGNORECASE | re.DOTALL)
                
                # Normalize whitespace
                text = re.sub(r'\s+', ' ', text).strip()
            else:
                text = ""
            
            # Check for paywall indicators
            paywall_detected = False
            paywall_terms = [
                'subscribe to read', 'subscribe to continue', 'subscription required',
                'to continue reading', 'create an account', 'sign up to read',
                'premium content', 'premium article', 'members only', 'paid subscribers only',
                'subscribe to unlock', 'subscribe for full access', 'subscribe now',
                'for subscribers only', 'for full access', 'unlock this article',
                'login to read', 'sign in to read', 'register to read', 'register to continue',
                'subscribe to', 'subscription', 'paywall', 'premium', 'subscribe for',
                'already subscribed', 'subscriber-only', 'subscribers only'
            ]
            
            # Check for common paywall domains
            paywall_domains = [
                'ft.com', 'wsj.com', 'nytimes.com', 'economist.com', 'bloomberg.com',
                'washingtonpost.com', 'newyorker.com', 'wired.com', 'thetimes.co.uk',
                'theinformation.com', 'stratechery.com', 'theatlantic.com', 'medium.com',
                'forbes.com', 'businessinsider.com', 'seekingalpha.com', 'barrons.com'
            ]
            
            # Check if URL domain is in known paywall domains
            if url:
                domain = url.split('://')[1].split('/')[0] if '://' in url else ''
                if any(pd in domain for pd in paywall_domains):
                    # For known paywall sites, set a higher suspicion level
                    paywall_suspicion = True
                else:
                    paywall_suspicion = False
            else:
                paywall_suspicion = False
            
            if text:
                lower_text = text.lower()
                
                # Check for paywall terms in the content
                if any(term in lower_text for term in paywall_terms):
                    paywall_detected = True
                    print(f"Paywall detected for {url}")
                # If we have a suspicious domain and the content is very short, it's likely paywalled
                elif paywall_suspicion and len(lower_text) < 1000:
                    paywall_detected = True
                    print(f"Likely paywall detected for {url} (known paywall site with limited content)")
            # If we have a suspicious domain and no content, it's likely paywalled
            elif paywall_suspicion:
                paywall_detected = True
                print(f"Likely paywall detected for {url} (known paywall site)")
            
            # Generate a better summary
            summary = ""
            if text:
                # Try to extract a good summary - first paragraph or first few sentences
                sentences = sent_tokenize(text)
                if sentences:
                    # If we have a short article, use the first 1-2 sentences
                    if len(sentences) <= 3:
                        summary = ' '.join(sentences)
                    else:
                        # For longer articles, use first paragraph or first few sentences
                        first_para = text.split('\n\n')[0] if '\n\n' in text else ' '.join(sentences[:3])
                        
                        # Limit summary length
                        if len(first_para) > 500:
                            summary = first_para[:500] + "..."
                        else:
                            summary = first_para
            
            # If we couldn't extract a good summary, use the first 500 chars
            if not summary and text:
                summary = text[:500] + "..." if len(text) > 500 else text
            
            # Extract features
            features = {
                'length': len(text),
                'has_code': bool(re.search(r'```|\{|\}|function|class|def|return|import', text)),
                'has_quotes': text.count('"') + text.count('"') + text.count('"'),
                'sentence_count': len(sent_tokenize(text)) if text else 0,
                'link_count': content.count('href='),
                'reading_time': max(1, len(text.split()) // 200) if text else 1,  # Ensure at least 1 minute
                'technical_terms': sum(1 for term in [
                    'algorithm', 'database', 'framework', 'api', 'cloud', 
                    'infrastructure', 'protocol', 'architecture'
                ] if term in text.lower()) if text else 0,
                'images': images,
                'summary': summary,
                'paywall_detected': paywall_detected
            }

            # Cache the results
            self.cache.set(url, text, features)

            return {
                'content': text,
                'features': features,
                'images': images
            }

        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None


class ContentProcessor:
    """
    Handles parallel processing of multiple articles.
    """
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_REQUESTS):
        self.fetcher = ArticleFetcher()
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_urls(self, urls: List[str]) -> Dict[str, Dict]:
        """
        Process multiple URLs in parallel with rate limiting.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            Dict mapping URLs to their processed content and features
        """
        async def process_with_semaphore(url: str) -> Tuple[str, Optional[Dict]]:
            async with self.semaphore:
                result = await self.fetcher.process_article(url)
                return url, result

        tasks = [process_with_semaphore(url) for url in urls]
        results = {}
        
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Fetching articles"
        ):
            url, result = await task
            if result:
                results[url] = result

        await self.fetcher.close_session()
        return results 