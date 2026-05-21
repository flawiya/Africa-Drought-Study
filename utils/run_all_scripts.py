import os
import subprocess
import shutil
from pathlib import Path

def run_scripts():
    # --- PATH SETTINGS ---
    # Where your .py files are located
    project_folder = Path(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk")
    
    # Where you want the outputs saved
    output_root = Path(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Output")
    
    # Create Output root if it doesn't exist
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Get all .py files (excluding this script itself and the converter script)
    scripts_to_run = [
        f for f in project_folder.glob("*.py") 
        if f.name not in ["run_all_scripts.py", "convert_my_notebooks.py"]
    ]

    failed_scripts = []
    successful_scripts = []

    print(f"Starting execution of {len(scripts_to_run)} scripts...\n")

    for script_path in scripts_to_run:
        script_name = script_path.stem
        # Create a specific folder for this script's output
        script_output_dir = output_root / script_name
        script_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running: {script_name}...")

        try:
            # We run the script and set the 'cwd' (current working directory) 
            # to the script's specific output folder.
            # This means any files the script saves (CSV, PNG, etc.) will 
            # automatically land in that folder.
            result = subprocess.run(
                ['python', str(script_path)],
                cwd=str(script_output_dir), # Files saved by the script go here
                capture_output=True,
                text=True,
                timeout=600 # 10-minute timeout per script, adjust if needed
            )

            if result.returncode == 0:
                print(f"✅ Success: {script_name}")
                successful_scripts.append(script_name)
                # Save the console output (print statements) to a log file
                with open(script_output_dir / "console_log.txt", "w") as log:
                    log.write(result.stdout)
            else:
                print(f"❌ Failed: {script_name}")
                failed_scripts.append((script_name, result.stderr))
                # Save the error log
                with open(script_output_dir / "error_log.txt", "w") as log:
                    log.write(result.stderr)

        except Exception as e:
            print(f"⚠️ Error running {script_name}: {str(e)}")
            failed_scripts.append((script_name, str(e)))

    # --- FINAL REPORT ---
    print("\n" + "="*30)
    print("FINAL EXECUTION REPORT")
    print("="*30)
    print(f"Total Processed: {len(scripts_to_run)}")
    print(f"Successful: {len(successful_scripts)}")
    print(f"Failed: {len(failed_scripts)}")
    
    if failed_scripts:
        print("\nLIST OF FAILED SCRIPTS AND ERRORS:")
        for name, error in failed_scripts:
            print(f"\n--- {name} ---")
            # Print just the last line of the error for brevity
            print(error.strip().split('\n')[-1]) 
            print(f"Full error log saved in: Output/{name}/error_log.txt")
    else:
        print("\nAll scripts ran successfully!")

if __name__ == "__main__":
    run_scripts()