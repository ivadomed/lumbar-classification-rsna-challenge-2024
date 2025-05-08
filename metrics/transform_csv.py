import pandas as pd

# Load the CSV
df = pd.read_csv("train_label_coordinates.csv")

# Remove Subarticular rows
df = df[~df["condition"].str.contains("Subarticular", case=False, na=False)]

# Create the coordinates column as a tuple
def make_coords(row):
    if "Foraminal" in row["condition"]:
        return (row["instance_number"], row["x"], row["y"])
    elif "Spinal Canal Stenosis" in row["condition"]:
        return (row["x"], row["y"], row["instance_number"])
    else:
        return None  # optional: handle unexpected conditions

df["coordinates"] = df.apply(make_coords, axis=1)

# Drop rows with None coordinates (if any)
df = df[df["coordinates"].notna()]

# Expand coordinates into separate columns
df[["L", "P", "I"]] = pd.DataFrame(df["coordinates"].tolist(), index=df.index)

# Keep only required columns
df = df[["study_id", "condition", "level", "L", "P", "I"]]

# Save to new CSV
df.to_csv("processed_annotations.csv", index=False)
