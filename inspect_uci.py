from ucimlrepo import fetch_ucirepo

# Fetch the UCI Thyroid Disease dataset
thyroid = fetch_ucirepo(id=102)

# Display basic information
print("=" * 60)
print("THYROID DATASET INFORMATION")
print("=" * 60)

print("\nDataset name:")
print(thyroid.metadata.name)

print("\nDataset description:")
print(thyroid.metadata.description)

print("\nRepository:")
print(thyroid.metadata.repository_url)

print("\nFeatures:")
print(thyroid.variables)