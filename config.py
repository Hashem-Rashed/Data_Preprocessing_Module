"""
Configuration settings for the data pipeline
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class Settings:
    """Application settings"""
    # File paths
    DATA_PATH: str = "data/fordgobike.csv"
    OUTPUT_PATH: str = "data/cleaned_data.csv"
    REPORT_PATH: str = "reports/quality_report.txt"
    
    # Column definitions
    NUMERICAL_COLUMNS: List[str] = None  # Will be auto-detected if None
    CATEGORICAL_COLUMNS: List[str] = None  # Will be auto-detected if None
    DATE_COLUMNS: List[str] = None
    
    # Processing parameters
    OUTLIER_FACTOR: float = 1.5
    IMPUTE_STRATEGY: str = "median"  # Options: "mean", "median", "most_frequent"
    DROP_THRESHOLD: float = 0.5  # Drop columns with > 50% missing values
    
    # Display settings
    PREVIEW_ROWS: int = 10
    MAX_COLS_TO_DISPLAY: int = 20
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        Path("data").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)

# Global settings instance
settings = Settings()