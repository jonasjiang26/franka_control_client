from datasets import load_dataset
import numpy as np

# Load the pusht dataset
print("Loading dataset...")
ds = load_dataset("lerobot/pusht")

print("\n" + "=" * 80)
print("Dataset Structure")
print("=" * 80)

# Get the training split
train_ds = ds['train']

print(f"\nNumber of rows: {len(train_ds):,}")
print(f"Number of columns: {len(train_ds.column_names)}")
print(f"\nColumn names: {train_ds.column_names}")

print("\n" + "=" * 80)
print("Features (Schema):")
print("=" * 80)
for feature_name, feature_type in train_ds.features.items():
    print(f"  {feature_name}: {feature_type}")

print("\n" + "=" * 80)
print("Detailed Column Information:")
print("=" * 80)
for col_name in train_ds.column_names:
    print(f"\n  Column: {col_name}")
    print(f"    Type: {train_ds.features[col_name]}")
    
    # Get first value
    try:
        first_val = train_ds[0][col_name]
        if isinstance(first_val, (int, float, str, bool, type(None))):
            print(f"    First value: {first_val}")
        elif isinstance(first_val, dict):
            print(f"    First value (dict):")
            for k, v in first_val.items():
                if isinstance(v, np.ndarray):
                    print(f"      {k}: array shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, list):
                    print(f"      {k}: list length={len(v)}")
                else:
                    print(f"      {k}: {type(v).__name__}")
        elif isinstance(first_val, (list, np.ndarray)):
            if isinstance(first_val, np.ndarray):
                print(f"    First value: array shape={first_val.shape}, dtype={first_val.dtype}")
            else:
                print(f"    First value: list length={len(first_val)}")
        else:
            print(f"    First value type: {type(first_val).__name__}")
    except Exception as e:
        print(f"    Could not display first value: {e}")

print("\n" + "=" * 80)
print("Sample rows (first 3):")
print("=" * 80)
for i in range(min(3, len(train_ds))):
    print(f"\nRow {i}:")
    row = train_ds[i]
    for key, value in row.items():
        if isinstance(value, dict):
            print(f"  {key}: (dict with keys: {list(value.keys())})")
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}, sample={v.flatten()[:3]}...")
                else:
                    print(f"    {k}: {v}")
        elif isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, sample={value.flatten()[:3]}...")
        elif isinstance(value, list):
            print(f"  {key}: list length={len(value)}, sample={value[:3] if len(value) > 0 else '[]'}...")
        else:
            print(f"  {key}: {value}")

print("\n" + "=" * 80)
print(f"Dataset info:")
print("=" * 80)
print(train_ds.info)
