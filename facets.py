import pandas as pd
import re

def load_facets(csv_path="Facets Assignment.csv"):
    df = pd.read_csv(csv_path)
    facets = df["Facets"].dropna().str.strip().tolist()
    facets = [f for f in facets if f]
    # Remove leading numbers like "793. " or "45. "
    facets = [re.sub(r"^\d+\.\s*", "", f).strip() for f in facets]
    return facets

if __name__ == "__main__":
    facets = load_facets()
    print(f"Total facets loaded: {len(facets)}")
    print("First 5:", facets[:5])
    print("Last 5:", facets[-5:])