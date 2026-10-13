import json
import glob
import os

def check_telemetry():
    # Find the newest JSON file in the telemetry folder
    files = glob.glob("./data/telemetry/*.json")
    if not files:
        print("No JSON files found in ./data/telemetry/")
        return
        
    latest_file = max(files, key=os.path.getctime)
    print(f"Inspecting file: {latest_file}\n")
    
    with open(latest_file, "r") as f:
        data = json.load(f)
        
    print(f"Total frames recorded: {len(data)}")
    
    if len(data) == 0:
        print("Error: The file is empty!")
        return
        
    # Check the first frame for the required structure
    sample = data[0]
    
    print("\n--- Structural Audit (First Frame) ---")
    
    # 1. Action Probabilities
    probs = sample.get("action_probabilities", {})
    print(f"Action Probabilities tracked: {len(probs)} (Expected: 7)")
    
    # 2. Value Estimate
    val = sample.get("value_estimate")
    print(f"Value Estimate V(s): {val} (Expected: scalar float)")
    
    # 3. Latent Vector
    latent = sample.get("latent_vector", [])
    print(f"Latent Vector length: {len(latent)} (Expected: 128)")
    
    # 4. Step Tracking
    print(f"Step Index: {sample.get('step')}")
    
    # Verify core requirements from the paper
    if len(probs) == 7 and isinstance(val, float) and len(latent) == 128:
        print("\n✅ SUCCESS: All critical Actor-Critic signals match the required architecture!")
        print("You are officially ready to start Module 3.")
    else:
        print("\n❌ WARNING: Some fields are missing or incorrectly sized.")

if __name__ == "__main__":
    check_telemetry()