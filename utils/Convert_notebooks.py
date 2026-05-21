import nbformat
from nbconvert import PythonExporter
from pathlib import Path

def bulk_convert():
    # --- YOUR SPECIFIC PATHS ---
    # Source: The folder containing multiple subfolders with notebooks
    source_root = Path(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought")
    
    # Destination: Your main project folder
    dest_root = Path(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk")

    # Initialize the exporter
    exporter = PythonExporter()

    # Find all .ipynb files recursively in the source folder
    notebooks = list(source_root.rglob("*.ipynb"))
    
    if not notebooks:
        print("No notebooks found in the specified source folder.")
        return

    print(f"Found {len(notebooks)} notebooks. Starting conversion...")

    for nb_path in notebooks:
        # Skip hidden checkpoint folders
        if ".ipynb_checkpoints" in str(nb_path):
            continue

        try:
            print(f"Converting: {nb_path.name}")
            
            # 1. Read the notebook
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb_node = nbformat.read(f, as_version=4)
            
            # 2. Convert to Python
            (body, _) = exporter.from_notebook_node(nb_node)
            
            # 3. Create a unique name to prevent overwriting 
            # (e.g., if multiple folders have a 'Model.ipynb')
            # It will name it 'FolderName_NotebookName.py'
            subfolder_name = nb_path.parent.name
            output_name = f"{subfolder_name}_{nb_path.stem}.py" if subfolder_name != "Drought" else f"{nb_path.stem}.py"
            
            output_file_path = dest_root / output_name
            
            # 4. Save the file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(body)
                
        except Exception as e:
            print(f"Error converting {nb_path.name}: {e}")

    print(f"\nSuccess! All .py files are now in: {dest_root}")

if __name__ == "__main__":
    bulk_convert()