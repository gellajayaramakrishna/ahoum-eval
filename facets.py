import pandas as pd

def load_facets(csv_path="Facets Assignment.csv"):
    df = pd.read_csv(csv_path)
    # Clean facets — remove empty, strip whitespace
    facets = df["Facets"].dropna().str.strip().tolist()
    facets = [f for f in facets if f]
    return facets

if __name__ == "__main__":
    facets = load_facets()
    print(f"Total facets loaded: {len(facets)}")
    print("First 5:", facets[:5])
    print("Last 5:", facets[-5:])