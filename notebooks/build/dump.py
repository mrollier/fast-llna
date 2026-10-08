"""Print an executed notebook's text outputs and save its images, for checking a notebook's text against its outputs.

    python notebooks/build/dump.py notebooks/consensus_sweep.ipynb /tmp/sweep_   # images -> /tmp/sweep_<cell>.png
"""

import base64
import sys

import nbformat

nb = nbformat.read(sys.argv[1], 4)
print("=====", sys.argv[1])
for i, cell in enumerate(nb.cells):
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        if out.output_type == "stream":
            print(i, out.text.strip()[:900])
        elif out.output_type == "error":
            print(i, "ERROR", out.ename, out.evalue, "".join(out.traceback)[-1500:])
        elif "text/markdown" in data:
            print(i, data["text/markdown"])
        elif "image/png" in data:
            with open(f"{sys.argv[2]}{i}.png", "wb") as f:
                f.write(base64.b64decode(data["image/png"]))
            print(i, "IMAGE")
