"""
data_loader.py
--------------
This file is responsible for loading the raw household sensor data
and resampling it to a chosen time resolution (10 min, 30 min, or 1 hour).
"""

from pathlib import Path
import pandas as pd
import re


def load_household_files(root: Path) -> pd.DataFrame:
    """
    Load all individual household sensor files and combine them into one table.
    
    What it does:
    - Finds all CSV and Excel files in the data folder
    - Skips hidden macOS files
    - Extracts the household ID from the filename
    - Combines everything into a single DataFrame
    """
    all_files = []
    for ext in ["*.csv", "*.xlsx", "*.xls"]:
        all_files.extend(root.rglob(ext))
    all_files = [f for f in all_files if not f.name.startswith("._")]

    def extract_household_id(path: Path) -> str:
        # Example: "Qtotal001.csv" → "001"
        match = re.search(r"(\d{2,4})$", path.stem)
        return match.group(1).zfill(3) if match else None

    def load_file(path: Path) -> pd.DataFrame:
        # Try different encodings for CSV files
        if path.suffix.lower() == ".csv":
            for enc in ["utf-8", "latin1", "cp1252"]:
                try:
                    return pd.read_csv(path, encoding=enc)
                except UnicodeDecodeError:
                    continue
            return pd.read_csv(path, encoding="latin1")
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            return pd.read_excel(path, engine="xlrd")

    # Only keep files that look like individual household data
    household_files = [f for f in all_files if re.search(r"\d{2,4}$", f.stem)]

    dfs = []
    for f in household_files:
        df = load_file(f)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        df["household_id"] = extract_household_id(f)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Standardize the time column name and format
    time_col = next((c for c in ["Time", "time", "timestamp", "Timestamp", "DateTime"] if c in df.columns), None)
    if time_col is None:
        raise KeyError("No time column found")
    df = df.rename(columns={time_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["household_id"] = df["household_id"].astype(str).str.zfill(3)
    return df


def resample_data(df: pd.DataFrame, freq: str = "10min") -> pd.DataFrame:
    """
    Resample the minute-level data to a coarser time resolution.
    
    Examples of freq:
    - "10min"  → 10-minute averages
    - "30min"  → 30-minute averages
    - "1h"     → hourly averages
    """
    df = df.set_index("timestamp")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    pieces = []
    for hid, group in df.groupby("household_id"):
        res = group[numeric_cols].resample(freq).mean()
        res["household_id"] = hid
        pieces.append(res)
    return pd.concat(pieces).reset_index()

