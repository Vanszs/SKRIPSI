import nbformat
from pathlib import Path

nb_path = Path("notebooks/04_Model_Comparison.ipynb")
nb = nbformat.read(nb_path, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "markdown":
        # Remove ARIMA lines
        if "ARIMA" in cell.source:
            lines = cell.source.split('\n')
            new_lines = [l for l in lines if "ARIMA" not in l]
            cell.source = '\n'.join(new_lines)
            print("Removed ARIMA from markdown.")
        # Remove ETS lines
        if "ETS" in cell.source:
            lines = cell.source.split('\n')
            new_lines = [l for l in lines if "ETS" not in l]
            cell.source = '\n'.join(new_lines)
            print("Removed ETS from markdown.")
            
        # Renumber if needed (optional, logic might be brittle so just cleaning text)
        if "4. **Gradient Boosting**" in cell.source:
             cell.source = cell.source.replace("4. **Gradient Boosting**", "2. **Gradient Boosting**") 
             # Assuming RF is 1, GBR becomes 2. LigthGBM is processed? Let's keep it simple.
             
nbformat.write(nb, nb_path)
print("Notebook saved.")
