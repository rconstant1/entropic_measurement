import csv
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Literal

class EntropicLogger:
    """
    Simple and robust logger for saving calculation results (as dictionaries).
    Supports export to CSV or JSON, and automatic timestamping.
    """

    def __init__(self):
        self.log: List[Dict[str, Any]] = []

    def record(self, result: Dict[str, Any]):
        """
        Adds a result to the log with a timestamp.
        Args:
            result: A dictionary representing the result of a calculation.
        """
        entry = result.copy()
        entry['timestamp'] = datetime.now().isoformat()
        self.log.append(entry)

    def export(self, filepath: str, format: Literal['csv', 'json'] = "csv"):
        """
        Export the log as a CSV or JSON file.

        Args:
            filepath: The target filename (can include path).
            format: Export format ('csv' or 'json', case-insensitive).
        """
        format = format.lower()
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            if self.log:
                # Collect all keys from all entries in the log
                all_keys = {k for entry in self.log for k in entry.keys()}
                sorted_keys = sorted(all_keys)
                with open(out_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted_keys)
                    writer.writeheader()
                    for row in self.log:
                        # Fill missing columns for each row
                        row_filled = {k: row.get(k, "") for k in sorted_keys}
                        writer.writerow(row_filled)
        elif format == "json":
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(self.log, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'json'.")

    def clear(self):
        """
        Clear the entire log.
        """
        self.log.clear()
