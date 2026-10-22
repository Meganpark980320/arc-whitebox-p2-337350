"""T4. Compare the "Active Weight" column of the shipped weights.csv with the weights flopscope bills (default_weights.json)."""
import csv, json
from importlib import resources
import flopscope

data = resources.files(flopscope) / "data"
billed = json.loads((data / "default_weights.json").read_text())
billed = billed.get("weights", billed)
rows = list(csv.DictReader((data / "weights.csv").read_text().splitlines()))
diff = []
for r in rows:
    op = r["Operation"]
    if op in billed and r.get("Active Weight", "") != "":
        a, b = float(r["Active Weight"]), float(billed[op])
        if a != b:
            diff.append((op, a, b))
print(f"flopscope {flopscope.__version__}: weights.csv rows {len(rows)}, in billing {sum(r['Operation'] in billed for r in rows)}, "
      f"'Active Weight' differs from billing for {len(diff)}")
for op, a, b in sorted(diff, key=lambda t: (-abs(t[2] - t[1]), t[0]))[:25]:
    print(f"  {op:28s} csv {a:>5g}   billed {b:>5g}")
