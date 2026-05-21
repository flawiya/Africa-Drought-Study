import os
import subprocess
from pathlib import Path

def validate_all_scripts():
    # 1. SETUP PATHS
    # This is where your .py files are
    base_path = Path(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk")
    
    # This is where the output folders will be created
    output_root = base_path / "Output"
    output_root.mkdir(exist_ok=True)

    # 2. FIND ALL PYTHON SCRIPTS
    # We find all .py files but ignore our utility scripts
    all_scripts = [
        f for f in base_path.glob("*.py") 
        if f.name not in ["run_validation.py", "convert_my_notebooks.py"]
    ]

    failed_scripts = []
    print(f"Found {len(all_scripts)} scripts. Starting validation...\n")

    for script_path in all_scripts:
        script_name = script_path.stem
        
        # Create a dedicated subfolder for this script's outputs
        script_output_dir = output_root / script_name
        script_output_dir.mkdir(exist_ok=True)
        
        print(f"Checking: {script_name}...", end=" ", flush=True)

        try:
            # We run the script. 
            # 'cwd' means any files the script saves will go into its Output folder automatically.
            result = subprocess.run(
                ['python', str(script_path)],
                cwd=str(script_output_dir), 
                capture_output=True,
                text=True,
                timeout=300 # 5-minute limit per script
            )

            if result.returncode == 0:
                print("✅ PASSED")
                # Save what the script printed to a file
                with open(script_output_dir / "logs_success.txt", "w") as f:
                    f.write(result.stdout)
            else:
                print("❌ FAILED")
                failed_scripts.append((script_name, result.stderr))
                # Save the error so you can read it later
                with open(script_output_dir / "logs_error.txt", "w") as f:
                    f.write(result.stderr)

        except Exception as e:
            print("⚠️ ERROR")
            failed_scripts.append((script_name, str(e)))

    # 3. FINAL SUMMARY REPORT
    print("\n" + "="*40)
    print("VALIDATION SUMMARY")
    print("="*40)
    if not failed_scripts:
        print("All scripts ran perfectly! You are ready for GitHub.")
    else:
        print(f"{len(failed_scripts)} scripts failed. Fix these before uploading:")
        for name, error in failed_scripts:
            # Show just the last line of the error (usually the most helpful)
            clean_error = error.strip().split('\n')[-1]
            print(f"- {name}: {clean_error}")
    print("="*40)

if __name__ == "__main__":
    validate_all_scripts()