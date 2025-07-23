import csv
import json

class EntropicLogger:
    def __init__(self):
        self.log = []

    def record(self, result):
        self.log.append(result)

    def export(self, filepath: str, format: str = "csv"):
        if format == "csv":
            if self.log:
                keys = list(self.log[0].keys())
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.log)
        elif format == "json":
            with open(filepath, 'w') as f:
                json.dump(self.log, f)
        else:
            raise ValueError("Unsupported format.")
