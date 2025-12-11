import nbformat
from pathlib import Path

nb_path = Path("notebooks/04_Model_Comparison.ipynb")
nb = nbformat.read(nb_path, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "markdown":
        if "4. **SVR**" in cell.source:
            cell.source = cell.source.replace("4. **SVR** (Support Vector Regression)", "4. **KNN** (K-Nearest Neighbors)")
            print("Updated SVR to KNN in markdown.")

nbformat.write(nb, nb_path)
print("Notebook saved.")
