import pandas as pd

def preprocess_facets(csv_path="Facets Assignment.csv"):
    df = pd.read_csv(csv_path)

    # Rename column cleanly
    df.columns = [col.strip() for col in df.columns]

    # Drop empty rows
    df = df.dropna(subset=["Facets"])

    # Clean facet names
    df["Facets"] = df["Facets"].str.strip()

    # Remove numbering if present (e.g. "793. Sufi practice" → "Sufi practice")
    df["Facets_Clean"] = df["Facets"].str.replace(
        r"^\d+\.\s*", "", regex=True
    )

    # Add category column based on keywords
    def categorize(facet):
        facet_lower = facet.lower()
        if any(w in facet_lower for w in ["spiritual", "sufi", "prayer", "meditation", "religious", "kabbalah", "islamic", "jewish", "mantra", "dhikr"]):
            return "Spiritual"
        elif any(w in facet_lower for w in ["emotion", "anxiety", "fear", "joy", "anger", "grief", "love", "stress"]):
            return "Emotional"
        elif any(w in facet_lower for w in ["leader", "assertive", "dominan", "democratic", "authorit"]):
            return "Leadership"
        elif any(w in facet_lower for w in ["cognitive", "logic", "reason", "intellect", "analytic", "common-sense"]):
            return "Cognitive"
        elif any(w in facet_lower for w in ["social", "empath", "compassion", "relation", "trust", "friend"]):
            return "Social"
        elif any(w in facet_lower for w in ["risk", "adventure", "impulsiv", "reckless"]):
            return "Risk"
        elif any(w in facet_lower for w in ["moral", "ethic", "honest", "decen", "integrity"]):
            return "Moral"
        elif any(w in facet_lower for w in ["food", "diet", "sleep", "exercise", "health"]):
            return "Lifestyle"
        else:
            return "General"

    df["Category"] = df["Facets_Clean"].apply(categorize)

    # Add score range column
    df["Score_Min"] = 1
    df["Score_Max"] = 5

    # Add description column
    def describe(facet):
        return f"Measures the degree of '{facet}' expressed in the conversation"

    df["Description"] = df["Facets_Clean"].apply(describe)

    # Save cleaned version
    df.to_csv("Facets_Cleaned.csv", index=False)
    print(f"Saved Facets_Cleaned.csv with {len(df)} facets")
    print(f"\nCategory distribution:")
    print(df["Category"].value_counts())
    print(f"\nSample:")
    print(df[["Facets_Clean", "Category", "Description"]].head(10))

    return df

if __name__ == "__main__":
    preprocess_facets()