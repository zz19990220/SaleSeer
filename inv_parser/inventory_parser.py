import pandas as pd

def load_sample_inventory():
    """Load the default sample inventory CSV bundled in assets/."""
    return pd.read_csv("assets/sample_inventory.csv")

def load_csv(file):
    """Load a user-uploaded CSV file (Streamlit UploadedFile)."""
    return pd.read_csv(file)
