import json
import os
from typing import List, Optional
from telemetry.schema import TelemetryFrame

class TelemetryLogger:
    def __init__(self, output_dir: str = "./data/telemetry/"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.buffer: List[TelemetryFrame] = []

    def log_frame(self, frame: TelemetryFrame) -> None:
        """Appends a single step bundle into the in-memory buffer."""
        self.buffer.append(frame)

    def log_batch(self, frames: List[TelemetryFrame]) -> None:
        """Appends multiple step bundles into the in-memory buffer."""
        self.buffer.extend(frames)

    def flush_to_json(self, session_id: str) -> str:
        """
        Saves all buffered telemetry records to disk and clears the buffer.
        Returns the absolute filepath of the generated JSON.
        """
        filepath = os.path.join(self.output_dir, f"session_{session_id}.json")
        serialized_records = [frame.to_dict() for frame in self.buffer]
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized_records, f, indent=2)
            
        print(f"[TelemetryLogger] Flushed {len(self.buffer)} records to {filepath}")
        self.buffer.clear()
        return filepath

    def get_buffered_records(self) -> List[dict]:
        """Provides direct in-memory access for vector store ingestion."""
        return [frame.to_dict() for frame in self.buffer]