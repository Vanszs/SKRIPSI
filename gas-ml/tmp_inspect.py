import json, pathlib
path = pathlib.Path("gas-ml/notebooks/02_model_evaluation.ipynb")
data = json.loads(path.read_text(encoding="utf-8"))
print(f"cells: {len(data.get('cells', []))}")
for i, cell in enumerate(data.get('cells', [])):
    cell_type = cell.get('cell_type')
    src = ''.join(cell.get('source', []))
    snippet = src[:500].encode('ascii', 'replace').decode('ascii')
    print(f"\nCell {i} [{cell_type}]\n" + '-'*60)
    print(snippet)
    print()
