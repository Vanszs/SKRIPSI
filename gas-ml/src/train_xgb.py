"""
Wrapper script untuk training standalone XGBoost model.
Shortcut untuk: python src/train.py --model-type xgboost
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("Training Standalone XGBoost Model...")
    
    # Get the directory of this script
    src_dir = Path(__file__).parent
    train_script = src_dir / "train.py"
    
    # Construct command
    # Pass all arguments received by this script to train.py, plus --model-type xgboost
    cmd = [sys.executable, str(train_script), "--model-type", "xgboost"] + sys.argv[1:]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print("\n✓ XGBoost training successful!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
