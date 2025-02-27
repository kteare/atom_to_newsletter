"""
HTML conversion utilities for TWTW.
"""
import re
import os
import datetime
import warnings
from typing import Dict, List, Optional
import mistune
from bs4 import BeautifulSoup
import logging

# Configure logging
logger = logging.getLogger(__name__)

class HtmlConverter:
    """
    Converts Markdown content to HTML.
    """
    def __init__(self, css_file: str = 'styles.css'):
        """
        Initialize the HtmlConverter.
        
        Args:
            css_file: Path to the CSS file to use for styling
        """
        self.css_file = css_file
        self.css_content = self._load_css()
    
    def _load_css(self) -> str:
        """
        Load CSS content from file.
        
        Returns:
            CSS content as string
        """
        try:
            with open(self.css_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"CSS file {self.css_file} not found. Using default styles.")
            return """
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
                color: #333;
            }

            /* Apply sans-serif to all text elements */
            p, h1, h2, h3, h4, h5, h6, span, a, li, td, th, div, blockquote, code, pre {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }

            h1, h2, h3 {
                color: #1a1a1a;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
            }

            h1 {
                font-size: 2.25em;
                font-weight: 700;
            }

            h2 {
                font-size: 1.75em;
                font-weight: 600;
            }

            h3 {
                font-size: 1.5em;
                font-weight: 600;
            }

            a {
                color: #0066cc;
                text-decoration: none;
            }

            a:hover {
                text-decoration: underline;
            }

            .article-metadata {
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                background-color: #f8f9fa;
                padding: 16px 20px;
                margin: 15px 0;
                border-radius: 8px;
                border-left: 4px solid #0066cc;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }

            .article-metadata p {
                margin: 0 10px 0 0;
                font-size: 0.95em;
                line-height: 1.4;
                border-right: 1px solid #e9ecef;
                padding-right: 10px;
            }

            .article-metadata p:last-child {
                border-right: none;
                padding-right: 0;
            }

            .article-author {
                font-size: 1em;
                color: #1a1a1a;
                margin: 0 10px 0 0;
            }

            .article-source {
                color: #2d3748;
            }

            .article-date, .article-time, .article-domain {
                color: #4a5568;
            }

            blockquote {
                border-left: 4px solid #0066cc;
                margin: 20px 0;
                padding: 10px 20px;
                background-color: #f8f9fa;
                color: #2d3748;
            }

            pre, code {
                background-color: #f8f9fa;
                border-radius: 4px;
                padding: 12px;
                overflow-x: auto;
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            }

            img {
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                margin: 20px 0;
            }

            hr {
                border: none;
                border-top: 1px solid #e9ecef;
                margin: 30px 0;
            }

            .metadata {
                margin: 1rem 0;
                padding: 0.75rem 1rem;
                background-color: #f8f9fa;
                border-radius: 6px;
                font-size: 0.95em;
                line-height: 1.4;
                color: #666;
            }

            .metadata strong {
                color: #333;
                margin-right: 0.25rem;
                font-weight: 600;
            }
            
            .twitter-handle {
                color: #1DA1F2;
                font-weight: normal;
            }
            
            .tweet-date {
                color: #657786;
                font-size: 0.9em;
                margin-left: 0.5em;
            }
            
            /* Style for lists to ensure sans-serif */
            ul, ol {
                padding-left: 1.5em;
                margin: 1em 0;
            }
            
            li {
                margin-bottom: 0.5em;
            }
            
            /* Force sans-serif globally */
            * {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
            }
            
            /* Exception for code elements */
            code, pre, .code {
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
            }
            """
    
    def replace_metadata(self, match) -> str:
        """
        Replace metadata div with formatted HTML.
        
        Args:
            match: Regex match object
            
        Returns:
            Formatted HTML for metadata
        """
        metadata_text = match.group(1)
        # Split on tabs or • to preserve horizontal layout
        metadata_pairs = [pair.strip() for pair in metadata_text.split('\t') if pair.strip()]
        if not metadata_pairs:
            metadata_pairs = [pair.strip() for pair in metadata_text.split('•') if pair.strip()]
        
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
    
    def convert_markdown_to_html(self, markdown_file_path: str, html_file_path: str) -> bool:
        """
        Converts a Markdown file to an HTML file.
        
        Args:
            markdown_file_path: Path to the input Markdown file
            html_file_path: Path to the output HTML file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(markdown_file_path, 'r', encoding='utf-8') as md_file:
                md_text = md_file.read()

            # Find and process all metadata sections
            md_text = re.sub(
                r'<div class="metadata">\n(.*?)\n</div>',
                self.replace_metadata,
                md_text,
                flags=re.DOTALL
            )
            
            # Fix Twitter handle format with date - pattern like: Person (@handle)[date]
            def format_twitter_handle(match):
                name = match.group(1)
                handle = match.group(2)
                date = match.group(3) if len(match.groups()) > 2 else ""
                
                if date:
                    return f'<strong>{name}</strong> <span class="twitter-handle">@{handle}</span> <span class="tweet-date">{date}</span>'
                else:
                    return f'<strong>{name}</strong> <span class="twitter-handle">@{handle}</span>'
                
            md_text = re.sub(
                r'(\w+)\s+\(@(\w+)\)(?:\[(.*?)\])?',
                format_twitter_handle,
                md_text
            )
            
            # Pre-process Markdown links to convert them to HTML links
            def convert_md_link(match):
                text = match.group(1)
                url = match.group(2)
                return f'<a href="{url}">{text}</a>'
            
            # Convert markdown links to HTML links
            md_text = re.sub(r'\[(.*?)\]\((.*?)\)', convert_md_link, md_text)
            
            # Convert footnote references
            md_text = re.sub(
                r'\[(\d+)\](?:\(#fn\d+-\d+\))?',
                r'<sup><a href="#footnote-\1">\1</a></sup>',
                md_text
            )

            # Convert the rest of the markdown using mistune
            md_parser = mistune.create_markdown(escape=False)
            html = md_parser(md_text)
            
            # Create a BeautifulSoup object to parse the HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Suppress DeprecationWarning for the find_all call
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                # Find all text nodes that contain Markdown bold syntax
                for element in soup.find_all(string=lambda s: s and '**' in s):
                    # Convert Markdown bold syntax to HTML strong tags
                    converted = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', element)
                    element.replace_with(BeautifulSoup(converted, 'html.parser'))

            # Convert the modified content back to string
            html = str(soup)
            
            # Additional cleanup
            # Fix any markdown link syntax that wasn't properly converted
            html = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', html)

            # Add complete HTML structure
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="TWTW Newsletter Generator">
    <meta name="date" content="{datetime.datetime.now().strftime('%Y-%m-%d')}">
    <title>{os.path.basename(html_file_path).replace('.html', '')}</title>
    <style>
        /* Base styles */
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
            font-size: 16px;
        }}
        
        /* Apply sans-serif to all text elements */
        p, h1, h2, h3, h4, h5, h6, span, a, li, td, th, div, blockquote {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        
        /* Headers */
        h1 {{
            font-size: 2.25em;
            font-weight: 700;
            color: #1a1a1a;
            margin-top: 1em;
            margin-bottom: 0.5em;
        }}
        
        h2 {{
            font-size: 1.75em;
            font-weight: 600;
            color: #1a1a1a;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }}
        
        h3 {{
            font-size: 1.5em;
            font-weight: 600;
            color: #1a1a1a;
            margin-top: 1.2em;
            margin-bottom: 0.5em;
        }}
        
        /* Links */
        a {{
            color: #0066cc;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        /* Lists */
        ul, ol {{
            padding-left: 1.5em;
            margin: 1em 0;
        }}
        
        li {{
            margin-bottom: 0.5em;
        }}
        
        /* Paragraphs */
        p {{
            margin-bottom: 1em;
            line-height: 1.6;
        }}
        
        /* Images */
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            margin: 20px 0;
        }}
        
        /* HR */
        hr {{
            border: none;
            border-top: 1px solid #e9ecef;
            margin: 30px 0;
        }}
        
        /* Metadata */
        .metadata {{
            margin: 1rem 0;
            padding: 0.75rem 1rem;
            background-color: #f8f9fa;
            border-radius: 6px;
            font-size: 0.95em;
            line-height: 1.4;
            color: #666;
        }}
        
        .metadata strong {{
            color: #333;
            margin-right: 0.25rem;
            font-weight: 600;
        }}
        
        /* Twitter handles */
        .twitter-handle {{
            color: #1DA1F2;
            font-weight: normal;
        }}
        
        .tweet-date {{
            color: #657786;
            font-size: 0.9em;
            margin-left: 0.5em;
        }}
        
        /* Quotes */
        blockquote {{
            border-left: 4px solid #0066cc;
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #f8f9fa;
            color: #2d3748;
        }}
        
        /* Code */
        pre, code {{
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 12px;
            overflow-x: auto;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        }}

        /* Article metadata */
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
        
        /* Override any browser default serif fonts */
        * {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }}
        
        /* Exception for code elements */
        code, pre, .code {{
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
        }}
    </style>
</head>
<body>
{html}
</body>
</html>"""

            # Write the HTML content to the output file
            with open(html_file_path, 'w', encoding='utf-8') as html_file:
                html_file.write(html_content)
            
            logger.info(f"Successfully converted {markdown_file_path} to {html_file_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Error: The file '{markdown_file_path}' does not exist.")
            return False
        except Exception as e:
            logger.error(f"Error converting {markdown_file_path} to HTML: {e}")
            return False
    
    def convert_all_markdown_files(self, directory: str) -> Dict[str, bool]:
        """
        Convert all Markdown files in a directory to HTML.
        
        Args:
            directory: Directory containing Markdown files
            
        Returns:
            Dict mapping filenames to conversion success status
        """
        results = {}
        
        # Get all Markdown files in the directory
        md_files = [f for f in os.listdir(directory) if f.endswith('.md')]
        
        for md_file in md_files:
            md_path = os.path.join(directory, md_file)
            html_path = os.path.join(directory, md_file.replace('.md', '.html'))
            
            results[md_file] = self.convert_markdown_to_html(md_path, html_path)
        
        return results


def main():
    """
    Main entry point for the HTML converter.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get the output directory from environment variable or use default
    output_dir = os.getenv('TWTW_OUTPUT_DIR', f'output_{datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S")}')
    
    # Create and run the converter
    converter = HtmlConverter()
    results = converter.convert_all_markdown_files(output_dir)
    
    # Log results
    success_count = sum(1 for success in results.values() if success)
    logger.info(f"Converted {success_count}/{len(results)} Markdown files to HTML")


if __name__ == '__main__':
    main() 