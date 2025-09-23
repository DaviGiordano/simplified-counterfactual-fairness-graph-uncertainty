import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)
    return cfg
