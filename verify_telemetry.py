import json
import math
import os
import sys
from pathlib import Path


def check_telemetry() -> int:
    repo_root = Path(__file__).resolve().parent
    telemetry_dir = repo_root / "data" / "telemetry"

    files = list(telemetry_dir.glob("*.json"))
    if not files:
        print(f"❌ Error: No JSON files found in {telemetry_dir}")
        return 1

    latest_file = max(files, key=os.path.getctime)
    print(f"Inspecting file: {latest_file}\n")

    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_frames = len(data)
    print(f"Total frames recorded: {total_frames}")

    if total_frames == 0:
        print("❌ Error: The telemetry file is empty!")
        return 1

    errors = []

    for idx, frame in enumerate(data):
        # 1. Step monotonic check
        if frame.get("step") != idx:
            errors.append(f"Frame {idx}: expected step {idx}, got {frame.get('step')}")

        # 2. Observation check (16-dim finite floats)
        obs = frame.get("observation", [])
        if len(obs) != 16 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in obs):
            errors.append(f"Frame {idx}: invalid observation vector (len={len(obs)})")

        # 3. Action probabilities check (7 actions, finite, sum ~= 1.0)
        probs = frame.get("action_probabilities", {})
        if len(probs) != 7:
            errors.append(f"Frame {idx}: expected 7 action probabilities, got {len(probs)}")
        else:
            prob_values = list(probs.values())
            if not all(isinstance(p, (int, float)) and math.isfinite(p) and 0.0 <= p <= 1.0 for p in prob_values):
                errors.append(f"Frame {idx}: non-finite or out-of-range action probabilities")
            elif not math.isclose(sum(prob_values), 1.0, abs_tol=1e-2):
                errors.append(f"Frame {idx}: action probabilities sum to {sum(prob_values):.4f} (expected ~1.0)")

        # 4. Critic value estimate V(s) check
        val = frame.get("value_estimate")
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            errors.append(f"Frame {idx}: invalid value_estimate {val}")

        # 5. Latent feature vector check (128-dim finite floats)
        latent = frame.get("latent_vector", [])
        if len(latent) != 128 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in latent):
            errors.append(f"Frame {idx}: invalid latent_vector (len={len(latent)})")

        # 6. Flags check
        flags = frame.get("flags", {})
        required_flags = {"is_low_health", "has_los", "in_cover"}
        if set(flags.keys()) != required_flags or not all(isinstance(v, bool) for v in flags.values()):
            errors.append(f"Frame {idx}: missing or non-boolean flags {flags}")

    sample = data[0]
    print("--- Structural Audit Summary (Sample Frame 0) ---")
    print(f"Action Probabilities tracked : {len(sample.get('action_probabilities', {}))} (Sum: {sum(sample.get('action_probabilities', {}).values()):.4f})")
    print(f"Value Estimate V(s)          : {sample.get('value_estimate')}")
    print(f"Latent Vector length         : {len(sample.get('latent_vector', []))}")
    print(f"Frames Audited               : {total_frames}/{total_frames}")

    if errors:
        print(f"\n❌ AUDIT FAILED with {len(errors)} error(s):")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more.")
        return 1

    print("\n✅ SUCCESS: All frames passed finite math, probability sum, and schema checks!")
    return 0


if __name__ == "__main__":
    sys.exit(check_telemetry())