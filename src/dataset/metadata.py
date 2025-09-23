import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def apply_metadata_conversions(df: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    """Apply type conversions based on metadata file."""
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.loads(f.read())

        # Extract continuous column names
        continuous_cols = [
            domain["name"]
            for domain in metadata.get("domains", [])
            if not domain.get("discrete", True)
        ]

        # Convert continuous columns to float64
        converted_cols = []
        for col in continuous_cols:
            if col in df.columns:
                df[col] = df[col].astype("float64")
                converted_cols.append(col)

        if converted_cols:
            logger.info(
                f"Converted {len(converted_cols)} continuous columns to float64: {', '.join(converted_cols)}"
            )
    except Exception as err:
        logger.warning(f"Failed to apply metadata conversions: {err}")

    return df
