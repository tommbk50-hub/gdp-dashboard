import json

file_path = "XGBoost_GPU_(3)_(4) (2).ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        if any("MIN_POCKET_VOLUME = 50.0" in line for line in source) or any("alphashape.alphashape" in line for line in source):
            new_source = []
            for i, line in enumerate(source):
                if "MIN_POCKET_VOLUME = 50.0" in line:
                    new_source.append(line.replace("50.0", "15.0"))
                    new_source.append("MIN_POINT_DENSITY = 0.01\n")
                elif "alpha_mesh = alphashape.alphashape(pocket_coords_cpu, 0.5)" in line:
                    new_source.append(line.replace("0.5", "0.8"))
                elif "    pockets.append({" in line:
                    # we want to insert the density check right before appending
                    new_source.append("    points_count = int(cp.sum(pocket_mask))\n")
                    new_source.append("    point_density = points_count / pocket_volume if pocket_volume > 0.0 else 0.0\n")
                    new_source.append("    \n")
                    new_source.append("    if pocket_volume < MIN_POCKET_VOLUME or point_density < MIN_POINT_DENSITY:\n")
                    new_source.append("        continue\n")
                    new_source.append("    \n")
                    new_source.append(line)
                elif "'points_count': int(cp.sum(pocket_mask))," in line:
                    new_source.append("        'points_count': points_count,\n")
                elif "shallow = [p for p in pockets if p['volume'] < MIN_POCKET_VOLUME]" in line:
                    pass # We will remove these lines
                elif "pockets = [p for p in pockets if p['volume'] >= MIN_POCKET_VOLUME]" in line:
                    pass
                elif "if shallow:" in line:
                    pass
                elif "    print(f\"  -> Filtered out {len(shallow)} shallow cluster(s) below {MIN_POCKET_VOLUME:.0f} A^3 (flat surface patches).\")" in line:
                    pass
                elif "# Reject sprawling, shallow clusters (flat 'pans')" in line:
                    pass
                elif "# so a drug molecule could never actually anchor into them." in line:
                    pass
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

