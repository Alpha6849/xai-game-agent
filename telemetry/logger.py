import json
import time
import uuid
from pathlib import Path
from typing import List, Optional
from telemetry.schema import TelemetryFrame

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "telemetry"


class TelemetryLogger:
    def __init__(self, output_dir: Optional[ str | Path ] = None):
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer: List[TelemetryFrame] = []

    def log_frame(self, frame: TelemetryFrame) -> None:
        """Appends a single step bundle into the in-memory buffer."""
        self.buffer.append(frame)

    def log_batch(self, frames: List[TelemetryFrame]) -> None:
        """Appends multiple step bundles into the in-memory buffer."""
        self.buffer.extend(frames)

    def flush_to_json(self, session_id: Optional[str] = None) -> str:
        """
        Saves all buffered telemetry records to disk using a collision-safe ID
        and clears the buffer. Returns the absolute filepath of the generated JSON.
        """
        if not session_id:
            ms_timestamp = int(time.time() * 1000)
            short_uid = uuid.uuid4().hex[:6]
            session_id = f"{ms_timestamp}_{short_uid}"

        filepath = self.output_dir / f"session_{session_id}.json"
        serialized_records = [frame.to_dict() for frame in self.buffer]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized_records, f, indent=2)

        print(f"[TelemetryLogger] Flushed {len(self.buffer)} records to {filepath}")
        self.buffer.clear()
        return str(filepath)

    def get_buffered_records(self) -> List[dict]:
        """Provides direct in-memory access for vector store ingestion."""
        return [frame.to_dict() for frame in self.buffer]