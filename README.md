# TWTW (That Was The Week) - Newsletter Generator

This is a sophisticated newsletter generation system that automatically collects, processes, and organizes technology news articles into a well-structured newsletter. The name "TWTW" stands for "That Was The Week" and the official website is [https://thatwastheweek.com](https://thatwastheweek.com).

## Core Functionality

1. **Content Collection**: 
   - The system fetches RSS/Atom feeds from Feedly (a popular feed aggregator) using the URLs specified in `feeds.txt`.
   - It supports both regular Feedly feeds and enterprise Feedly feeds with authentication.

2. **Content Processing**:
   - The system downloads article content from the original sources.
   - It extracts key information like title, author, source, publication date, and full text.
   - It implements caching to avoid re-downloading articles it has already processed.
   - It uses rate limiting to avoid overwhelming source websites.

3. **Content Analysis**:
   - The system uses NLP (Natural Language Processing) techniques to:
     - Classify articles into categories (Tech, AI, Venture Capital, etc.)
     - Generate summaries of article content
     - Estimate reading time

4. **AI Integration**:
   - It uses OpenAI's API for advanced tasks like:
     - Article classification
     - Subcategory classification
     - Text summarization
     - Content extraction and formatting

5. **Newsletter Generation**:
   - The system organizes articles into different sections based on their categories.
   - It creates multiple output formats:
     - Full newsletter with article summaries
     - Reading list with just article titles and links organized by category
     - Contents page with a table of contents
     - Original format with more detailed article information
     - Snippets with brief excerpts

6. **Output Formats**:
   - The system generates both Markdown (.md) and HTML versions of all outputs.
   - It applies consistent styling using CSS.

## Project Structure

The project has been organized into a proper Python package structure:

```
twtw/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── article.py  # Article dataclass and related functions
│   ├── cache.py    # CacheManager class
│   └── processor.py  # Content processing functionality
├── fetchers/
│   ├── __init__.py
│   └── feedly.py   # Feedly-specific fetching logic
├── formatters/
│   ├── __init__.py
│   ├── markdown.py  # Markdown formatting
│   └── html.py     # HTML conversion
├── utils/
│   ├── __init__.py
│   ├── nlp.py      # NLP utilities
│   └── http.py     # HTTP utilities
├── cli.py          # Command-line interface
└── config.py       # Configuration management
```

## Setup and Usage

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/kteare/atom_to_newsletter.git
   cd twtw
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Create a `.env` file with your API keys (see `.env.example` for reference):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Configuration

Create a `config.yaml` file to customize the behavior of the system:

```yaml
cache:
  directory: "cache"
  duration_days: 7

rate_limiting:
  requests_per_second: 1
  max_concurrent: 5
  timeout_seconds: 30

openai:
  model: "gpt-4"
  temperature: 0.7

output:
  formats:
    - markdown
    - html
  sections:
    - "Essays of the Week"
    - "News of the Week"
    - "Startup of the Week"
```

The `.env` file can also control the output directory behavior:

```
# Use a fixed output directory
TWTW_OUTPUT_DIR=output

# OR use a timestamped output directory (creates directories like 'output_2025-02-26:18:40:50')
TWTW_OUTPUT_DIR=output_datetimestamp

# Required: OpenAI API key for article processing
OPENAI_API_KEY=your_api_key_here
```

### Usage

1. Create a `feeds.txt` file with your Feedly feed URLs (one per line):
   ```
   https://feedly.com/f/your-feed-id.atom
   https://feedly.com/f/another-feed-id.atom
   ```

2. Run the application:
   ```bash
   python -m twtw.cli
   ```

3. Command-line options:
   ```bash
   python -m twtw.cli --help
   python -m twtw.cli --feeds custom_feeds.txt --output-dir custom_output
   python -m twtw.cli --refresh-cache  # Force refresh the cache
   python -m twtw.cli --skip-fetch  # Skip fetching feeds
   python -m twtw.cli --limit 0  # Process all articles in the feed (no limit)
   python -m twtw.cli --use-local-model  # Use local model instead of OpenAI
   ```

   By default, the application will process all articles in the feed. You can use the `--limit` option to specify a maximum number of articles to process, or set it to `0` to process all articles (which is now the default behavior).
   
   The output directory is determined by:
   1. The `--output-dir` command-line argument, if provided
   2. The `TWTW_OUTPUT_DIR` environment variable in your `.env` file
   3. If `TWTW_OUTPUT_DIR` contains the string "datetimestamp", it will be replaced with the current date and time
   4. If neither is set, it defaults to `output_YYYY-MM-DD:HH:MM:SS` with the current timestamp

## User Guide

### Setting Up Feeds

1. **Finding Feedly Feed URLs**:
   - Log in to your [Feedly](https://feedly.com) account
   - Navigate to the feed or collection you want to use
   - Copy the URL from the address bar, which should look like: `https://feedly.com/i/collection/content/user/...`
   - For Atom format, modify the URL to: `https://feedly.com/f/YOUR-FEED-ID.atom`
   
2. **Creating Your feeds.txt File**:
   - Create a text file named `feeds.txt` in the project root
   - Add one feed URL per line
   - You can add comments by starting a line with `#`
   - Example:
     ```
     # Tech news
     https://feedly.com/f/feed-id-1.atom
     # AI news
     https://feedly.com/f/feed-id-2.atom
     ```

### Running the Generator

1. **Basic Usage**:
   ```bash
   # Activate your virtual environment first
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Run with default settings
   python -m twtw.cli
   ```

2. **Advanced Usage**:
   ```bash
   # Use a custom feeds file and output directory
   python -m twtw.cli --feeds my_feeds.txt --output-dir newsletters/weekly
   
   # Process only the first 5 articles from each feed
   python -m twtw.cli --limit 5
   
   # Skip fetching and use previously downloaded feeds
   python -m twtw.cli --skip-fetch
   
   # Force refresh the article cache
   python -m twtw.cli --refresh-cache
   
   # Skip both fetching and processing, just regenerate HTML
   python -m twtw.cli --skip-fetch --skip-process
   ```

### Output Files

After running the generator, you'll find the following files in your output directory:

1. **Raw Feed Files**:
   - `feedly1.atom`, `feedly2.atom`, etc.: The raw feed files downloaded from Feedly

2. **Markdown Files**:
   - `newsletter.md`: Full newsletter with article summaries organized by category
   - `reading_list.md`: Simple list of article titles and links, organized by category
   
3. **HTML Files**:
   - `newsletter.html`: HTML version of the newsletter, ready to view in a browser
   - `reading_list.html`: HTML version of the reading list
   
### Viewing and Sharing Your Newsletter

1. **Viewing Locally**:
   - Open the generated HTML files in any web browser:
     ```bash
     # If using a fixed output directory (TWTW_OUTPUT_DIR=output in .env)
     open output/newsletter.html  # macOS
     xdg-open output/newsletter.html  # Linux
     start output/newsletter.html  # Windows
     
     # If using timestamped directories (TWTW_OUTPUT_DIR=output_datetimestamp in .env)
     # Replace YYYY-MM-DD:HH:MM:SS with the actual timestamp
     open output_YYYY-MM-DD:HH:MM:SS/newsletter.html  # macOS
     xdg-open output_YYYY-MM-DD:HH:MM:SS/newsletter.html  # Linux
     start output_YYYY-MM-DD:HH:MM:SS/newsletter.html  # Windows
     ```
     
   - The application will log the output directory path in the console, making it easy to locate your files.

2. **Sharing Options**:
   - Copy the HTML file to a web server
   - Convert to PDF for email distribution
   - Use the Markdown version for platforms that support it
   - Host on GitHub Pages or similar service

### Troubleshooting

1. **Missing API Keys**:
   - Ensure your `.env` file contains the required API keys
   - For OpenAI functionality, you need a valid `OPENAI_API_KEY`

2. **Feed Fetching Issues**:
   - Verify your Feedly URLs are correct and accessible
   - Check your internet connection
   - Look for error messages in the console output

3. **Article Processing Errors**:
   - Some articles may fail to process due to paywalls or site restrictions
   - The generator will still include these articles but with limited content
   - Check the log file for detailed error messages

4. **Cache Management**:
   - If articles aren't updating, try running with `--refresh-cache`
   - The cache is stored in the `cache` directory and can be deleted to start fresh

### Tips for Best Results

1. **Curate Your Feeds**:
   - Use high-quality sources that provide full article content in feeds
   - Mix different types of sources for variety
   
2. **Manage Output Size**:
   - If processing too many articles, use the `--limit` option
   - Consider creating separate feed files for different topics
   
3. **Customize Styling**:
   - Modify the CSS in `twtw/formatters/html.py` to change the appearance
   - Different output formats can be used for different purposes

4. **Scheduled Running**:
   - Set up a cron job or scheduled task to run the generator automatically
   - Example cron entry for weekly execution (Sundays at 7am):
     ```
     0 7 * * 0 cd /path/to/twtw && /path/to/venv/bin/python -m twtw.cli
     ```

## Requirements

- Python 3.7+
- Required Python packages (installed via requirements.txt):
  - aiohttp
  - beautifulsoup4
  - trafilatura
  - nltk
  - textblob
  - sumy
  - openai
  - mistune
  - pyyaml
  - requests
  - async-timeout
  - backoff
  - tqdm

## License

This project is for personal or organizational use to automatically collect, analyze, and organize technology news into a well-structured newsletter format, saving significant time that would otherwise be spent manually curating content. 