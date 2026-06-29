import json

nb_path = r"c:\Proyetos\Repos\MultiREx-public\examples\multirex-chemical_equilibrium.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "markdown":
        new_source = []
        for line in cell.get("source", []):
            line = line.replace("`ggchem` and `cia` dependencies", "`ggchem` dependency")
            line = line.replace("\"multirex[ggchem,cia]\"", "\"multirex[ggchem]\"")
            new_source.append(line)
        cell["source"] = new_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
