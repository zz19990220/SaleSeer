import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from inv_parser.inventory_parser import load_sample_inventory, load_csv

def test_load_sample_inventory_shape():
    df = load_sample_inventory()
    assert df.shape[1] >= 3          # 至少 3 列

def test_load_csv_accepts_path(tmp_path):
    csv = tmp_path / "mini.csv"
    csv.write_text("name,price\nFoo,9.9\n")
    df = load_csv(csv)
    assert df.iloc[0]["name"] == "Foo"