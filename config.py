# -*- coding: utf-8 -*-
"""
Configuration settings for the data pipeline
Improved version with environment-based configs and better organization
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BaseConfig:
    """Base configuration class with common settings"""
    
    # Directory paths
    DATA_DIR: Path = Path("data")
    OUTPUT_DIR: Path = Path("data")
    REPORT_DIR: Path = Path("reports")
    LOG_DIR: Path = Path("logs")
    
    # File paths
    DATA_PATH: str = "data/fordgobike.csv"
    OUTPUT_PATH: str = "data/cleaned_data.csv"
    REPORT_PATH: str = "reports/quality_report.txt"
    
    # Column definitions (None = auto-detect)
    NUMERICAL_COLUMNS: Optional[List[str]] = None
    CATEGORICAL_COLUMNS: Optional[List[str]] = None
    DATE_COLUMNS: Optional[List[str]] = None
    
    # Processing parameters
    OUTLIER_FACTOR: float = 1.5
    IMPUTE_STRATEGY: str = "median"  # Options: "mean", "median", "most_frequent"
    DROP_THRESHOLD: float = 0.7  # Drop columns with >70% missing values
    
    # Display settings
    PREVIEW_ROWS: int = 10
    MAX_COLS_TO_DISPLAY: int = 20
    DECIMAL_PLACES: int = 2
    
    # UI Constants
    HEADER_WIDTH: int = 70
    
    # Performance settings
    MAX_CATEGORY_CARDINALITY: int = 1000  # Max unique values for category dtype
    CHUNK_SIZE: int = 10000  # For chunked file reading
    
    # Feature flags
    ENABLE_OUTLIER_CLIPPING: bool = True
    ENABLE_TYPE_OPTIMIZATION: bool = True  # Optimize numeric types for memory
    ENABLE_PROGRESS_BARS: bool = True
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.DATA_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        self.REPORT_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'data_path': self.DATA_PATH,
            'output_path': self.OUTPUT_PATH,
            'outlier_factor': self.OUTLIER_FACTOR,
            'drop_threshold': self.DROP_THRESHOLD,
            'impute_strategy': self.IMPUTE_STRATEGY,
        }


@dataclass
class DevelopmentConfig(BaseConfig):
    """Development environment configuration"""
    
    # Override for development
    DATA_PATH: str = "data/sample_fordgobike.csv"
    DEBUG: bool = True
    VERBOSE: bool = True
    
    # Smaller limits for faster testing
    PREVIEW_ROWS: int = 5
    MAX_ROWS_TO_LOAD: Optional[int] = 10000  # Limit for faster dev
    
    # More aggressive cleaning for testing
    DROP_THRESHOLD: float = 0.5


@dataclass
class ProductionConfig(BaseConfig):
    """Production environment configuration"""
    
    # Production settings
    DEBUG: bool = False
    VERBOSE: bool = False
    
    # No limits in production
    MAX_ROWS_TO_LOAD: Optional[int] = None
    
    # Conservative cleaning
    DROP_THRESHOLD: float = 0.8
    
    # Performance optimizations
    ENABLE_PROGRESS_BARS: bool = True


@dataclass
class TestConfig(BaseConfig):
    """Test environment configuration"""
    
    # Test-specific paths
    DATA_DIR: Path = Path("tests/fixtures")
    OUTPUT_DIR: Path = Path("tests/output")
    REPORT_DIR: Path = Path("tests/reports")
    
    DATA_PATH: str = "tests/fixtures/test_data.csv"
    OUTPUT_PATH: str = "tests/output/test_cleaned.csv"
    
    # Fast settings for testing
    PREVIEW_ROWS: int = 3
    MAX_ROWS_TO_LOAD: int = 100
    
    DEBUG: bool = True


def get_config() -> BaseConfig:
    """
    Get configuration based on environment variable.
    
    Environment can be set via PIPELINE_ENV:
    - 'development' (default)
    - 'production'
    - 'test'
    
    Returns:
        Appropriate configuration object
    
    Examples:
        >>> # In shell: export PIPELINE_ENV=production
        >>> config = get_config()
        >>> print(config.DEBUG)
        False
    """
    env = os.getenv('PIPELINE_ENV', 'development').lower()
    
    config_map = {
        'development': DevelopmentConfig,
        'dev': DevelopmentConfig,
        'production': ProductionConfig,
        'prod': ProductionConfig,
        'test': TestConfig,
    }
    
    config_class = config_map.get(env, DevelopmentConfig)
    return config_class()


# Global settings instance
settings = get_config()


# Export configuration info
def print_config_info():
    """Print current configuration information"""
    print("\n" + "="*70)
    print("CONFIGURATION INFORMATION".center(70))
    print("="*70)
    
    env = os.getenv('PIPELINE_ENV', 'development')
    print(f"\nEnvironment: {env.upper()}")
    print(f"Config Class: {settings.__class__.__name__}")
    print(f"\nKey Settings:")
    print(f"  • Data Path: {settings.DATA_PATH}")
    print(f"  • Output Path: {settings.OUTPUT_PATH}")
    print(f"  • Debug Mode: {settings.DEBUG}")
    print(f"  • Outlier Factor: {settings.OUTLIER_FACTOR}")
    print(f"  • Drop Threshold: {settings.DROP_THRESHOLD * 100:.0f}%")
    print(f"  • Impute Strategy: {settings.IMPUTE_STRATEGY}")
    print("="*70)


if __name__ == "__main__":
    # Print config when run directly
    print_config_info()
