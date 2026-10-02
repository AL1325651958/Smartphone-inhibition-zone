"""Read the full inhibition-zone spreadsheet: it doubles as the label map."""
import csv
from pathlib import Path

import openpyxl

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
XL = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\链球菌检测\Sheet1.xlsx")
CSV = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\数据集\Sheet1.csv")

wb = openpyxl.load_workbook(XL, read_only=True, data_only=True)
ws = wb.worksheets[0]
grid = [list(r) for r in ws.iter_rows(values_only=True)]
print(f"sheet {ws.title}: {len(grid)} rows x {max(len(r) for r in grid)} cols")

for i, row in enumerate(grid):
    print(f"row{i:2}: {[('' if c is None else str(c))[:10] for c in row]}")

# structure: row0 = '人工测量'/'机器测量' pairs, row1 = 'mm',
#            row2 = plate id (merged over each pair), rows3+ = drug + diameters
header_plate = grid[2]
drug_rows = {}
for r in grid[3:]:
    name = r[0]
    if name is None or str(name).strip() == "":
        continue
    drug_rows[str(name).strip()] = r

# replicate the merged plate id across its measurement columns
plate_of_col = {}
cur = None
for c, v in enumerate(header_plate):
    if v is not None and str(v).strip() != "":
        cur = str(v).strip()
    if c >= 2:
        plate_of_col[c] = cur

plates = sorted({v for v in plate_of_col.values() if v})
print(f"\nplates in spreadsheet: {len(plates)} -> {plates}")
print(f"drug rows: {list(drug_rows)}")

print("\nper plate, which drugs have a measurement (that is the label set):")
print(f"{'plate':8} {'drugs':40} {'n_meas'}")
labelled = {}
for p in plates:
    cols = [c for c, v in plate_of_col.items() if v == p]
    drugs = []
    for name, row in drug_rows.items():
        vals = [row[c] for c in cols if c < len(row)]
        if any(v is not None and str(v).strip() != "" for v in vals):
            drugs.append(name)
    labelled[p] = drugs
    print(f"{p:8} {','.join(drugs):40} {len(drugs)}")

all_drugs = sorted({d for ds in labelled.values() for d in ds})
print(f"\nall drug tokens used: {all_drugs}")
print(f"plates with >=1 drug: {sum(1 for v in labelled.values() if v)}")

# does the spreadsheet agree with the 7 classes the class dataset uses?
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
print(f"\nclass-dataset classes: {CLASSES}")
print(f"missing from spreadsheet: {[c for c in CLASSES if c not in all_drugs]}")
print(f"extra in spreadsheet   : {[d for d in all_drugs if d not in CLASSES]}")
