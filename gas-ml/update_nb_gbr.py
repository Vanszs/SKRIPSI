import nbformat
from pathlib import Path

nb_path = Path("notebooks/04_Model_Comparison.ipynb")
nb = nbformat.read(nb_path, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "markdown":
        if "4. **KNN**" in cell.source:
            cell.source = cell.source.replace("4. **KNN** (K-Nearest Neighbors)", "4. **Gradient Boosting** (sklearn GBR)")
            print("Updated KNN to Gradient Boosting in markdown.")

nbformat.write(nb, nb_path)
print("Notebook saved.")
