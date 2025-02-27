from setuptools import setup, find_packages

setup(
    name="twtw",
    version="0.1.0",
    description="That Was The Week - Newsletter Generator",
    author="Keith Teare",
    author_email="keith@teare.com",
    url="https://thatwastheweek.com",
    packages=find_packages(),
    install_requires=[
        "aiohttp>=3.8.0",
        "beautifulsoup4>=4.10.0",
        "trafilatura>=1.2.0",
        "nltk>=3.6.0",
        "textblob>=0.17.1",
        "sumy>=0.10.0",
        "openai>=0.27.0",
        "tqdm>=4.62.0",
        "backoff>=1.11.0",
        "mistune>=2.0.0",
        "pyyaml>=6.0",
        "requests>=2.25.0",
        "async-timeout>=4.0.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "twtw=twtw.cli:main",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 