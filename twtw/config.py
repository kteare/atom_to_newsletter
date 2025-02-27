"""
Configuration management for TWTW.
"""
import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default configuration
DEFAULT_CONFIG = {
    "cache": {
        "directory": "cache",
        "duration_days": 7
    },
    "rate_limiting": {
        "requests_per_second": 1,
        "max_concurrent": 5,
        "timeout_seconds": 30
    },
    "openai": {
        "model": "gpt-4",
        "temperature": 0.7
    },
    "output": {
        "formats": ["markdown", "html"],
        "sections": [
            "Essays of the Week",
            "News of the Week",
            "Startup of the Week"
        ]
    },
    "categories": {
        "Essays of the Week": {
            "keywords": [
                "analysis", "perspective", "opinion", "research", "deep dive",
                "exploring", "understand", "think", "believe", "argument",
                "implications", "future", "impact", "trends", "philosophical",
                "review", "critique", "examine", "investigate", "study"
            ],
            "features": {
                "min_length": 1000,
                "min_quotes": 1,
                "min_reading_time": 5
            }
        },
        "News of the Week": {
            "keywords": [
                "announces", "releases", "updates", "reports", "says",
                "launches", "introduces", "unveils", "confirms", "plans",
                "new", "latest", "product", "feature", "event"
            ],
            "features": {
                "max_reading_time": 5,
                "prefers_recent": True,
                "max_sentiment_score": 0.2
            }
        },
        "Startup of the Week": {
            "keywords": [
                "funding", "valuation", "acquisition", "ipo", "investment", 
                "venture", "capital", "startup", "series", "round"
            ],
            "features": {
                "min_length": 500,
                "min_quotes": 1,
                "min_reading_time": 3
            }
        }
    }
}

class Config:
    """
    Configuration manager for TWTW.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Config.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """
        Load configuration from file or use defaults.
        
        Returns:
            Configuration dictionary
        """
        config = DEFAULT_CONFIG.copy()
        
        if self.config_path:
            try:
                path = Path(self.config_path)
                if path.exists():
                    if path.suffix.lower() in ['.yaml', '.yml']:
                        with open(path, 'r') as f:
                            user_config = yaml.safe_load(f)
                    elif path.suffix.lower() == '.json':
                        with open(path, 'r') as f:
                            user_config = json.load(f)
                    else:
                        raise ValueError(f"Unsupported config file format: {path.suffix}")
                    
                    # Update config with user settings
                    self._update_dict(config, user_config)
            except Exception as e:
                print(f"Error loading config from {self.config_path}: {e}")
                print("Using default configuration")
        
        # Override with environment variables
        self._override_from_env(config)
        
        return config
    
    def _update_dict(self, target: Dict, source: Dict) -> None:
        """
        Recursively update a dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value
    
    def _override_from_env(self, config: Dict, prefix: str = 'TWTW_') -> None:
        """
        Override configuration with environment variables.
        
        Args:
            config: Configuration dictionary to update
            prefix: Prefix for environment variables
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and split by underscore
                parts = key[len(prefix):].lower().split('_')
                
                # Navigate to the right place in the config
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the value
                try:
                    # Try to parse as JSON
                    current[parts[-1]] = json.loads(value)
                except json.JSONDecodeError:
                    # If not valid JSON, use as string
                    current[parts[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Dot-separated key path (e.g., 'cache.directory')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        parts = key.split('.')
        current = self.config
        
        for part in parts:
            if part not in current:
                return default
            current = current[part]
        
        return current
    
    def save(self, path: Optional[str] = None) -> bool:
        """
        Save the current configuration to a file.
        
        Args:
            path: Path to save the configuration to
            
        Returns:
            True if successful, False otherwise
        """
        save_path = path or self.config_path
        if not save_path:
            print("No path specified for saving configuration")
            return False
        
        try:
            path = Path(save_path)
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif path.suffix.lower() == '.json':
                with open(path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
            
            return True
        except Exception as e:
            print(f"Error saving config to {save_path}: {e}")
            return False


# Global configuration instance
config = Config(os.getenv('TWTW_CONFIG_PATH'))

def get_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value.
    
    Args:
        key: Dot-separated key path (e.g., 'cache.directory')
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    return config.get(key, default) 