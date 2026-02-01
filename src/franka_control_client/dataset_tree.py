from datasets import load_dataset
import numpy as np

def print_tree(name, value, indent=0, is_last=True, prefix=""):
    """Print a tree structure of the dataset"""
    # Tree characters
    if indent == 0:
        connector = ""
        new_prefix = ""
    else:
        connector = "└── " if is_last else "├── "
        new_prefix = prefix + ("    " if is_last else "│   ")
    
    # Print current node
    print(f"{prefix}{connector}{name}")
    
    return new_prefix

def print_feature_tree(feature_name, feature_type, indent=0, is_last=True, prefix=""):
    """Recursively print feature types in tree format"""
    if indent == 0:
        connector = ""
        new_prefix = ""
    else:
        connector = "└── " if is_last else "├── "
        new_prefix = prefix + ("    " if is_last else "│   ")
    
    # Convert feature type to string
    type_str = str(feature_type)
    
    # Print the feature
    print(f"{prefix}{connector}{feature_name}: {type_str}")
    
    # If it's a dict-like structure, recurse
    if hasattr(feature_type, 'feature'):
        # Sequence type
        sub_feature = feature_type.feature
        if hasattr(sub_feature, 'keys'):
            # Dict inside sequence
            keys = list(sub_feature.keys())
            for i, key in enumerate(keys):
                is_last_item = (i == len(keys) - 1)
                print_feature_tree(key, sub_feature[key], indent + 1, is_last_item, new_prefix)
    elif hasattr(feature_type, 'keys'):
        # Dict-like feature
        keys = list(feature_type.keys())
        for i, key in enumerate(keys):
            is_last_item = (i == len(keys) - 1)
            print_feature_tree(key, feature_type[key], indent + 1, is_last_item, new_prefix)

def print_value_tree(name, value, indent=0, is_last=True, prefix="", max_depth=3):
    """Print actual data values in tree format"""
    if indent >= max_depth:
        return
    
    if indent == 0:
        connector = ""
        new_prefix = ""
    else:
        connector = "└── " if is_last else "├── "
        new_prefix = prefix + ("    " if is_last else "│   ")
    
    # Determine value representation
    if isinstance(value, dict):
        print(f"{prefix}{connector}{name}: (dict)")
        keys = list(value.keys())
        for i, key in enumerate(keys):
            is_last_item = (i == len(keys) - 1)
            print_value_tree(key, value[key], indent + 1, is_last_item, new_prefix, max_depth)
    elif isinstance(value, np.ndarray):
        shape_str = f"shape={value.shape}, dtype={value.dtype}"
        sample = value.flatten()[:3]
        print(f"{prefix}{connector}{name}: array({shape_str}) = [{sample[0]:.4f}, {sample[1]:.4f}, ...]" if len(sample) >= 2 else f"{prefix}{connector}{name}: array({shape_str})")
    elif isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], dict):
            print(f"{prefix}{connector}{name}: list[{len(value)}] of dicts")
            if len(value) > 0:
                print(f"{new_prefix}├── [0]:")
                keys = list(value[0].keys())
                for i, key in enumerate(keys):
                    is_last_item = (i == len(keys) - 1)
                    print_value_tree(key, value[0][key], indent + 2, is_last_item, new_prefix + "│   ", max_depth)
        else:
            sample = value[:3] if len(value) > 3 else value
            print(f"{prefix}{connector}{name}: list[{len(value)}] = {sample}...")
    else:
        print(f"{prefix}{connector}{name}: {value}")

# Load dataset
print("Loading dataset...\n")
ds = load_dataset("lerobot/pusht")
train_ds = ds['train']

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Dataset: lerobot/pusht")
print(f"Split: train")
print(f"Rows: {len(train_ds):,}")
print(f"Columns: {len(train_ds.column_names)}")

print("\n" + "=" * 80)
print("SCHEMA TREE (Feature Types)")
print("=" * 80)
print("pusht-train")
columns = train_ds.column_names
for i, col_name in enumerate(columns):
    is_last = (i == len(columns) - 1)
    print_feature_tree(col_name, train_ds.features[col_name], 0, is_last, "")

print("\n" + "=" * 80)
print("DATA TREE (Sample Row 0)")
print("=" * 80)
print("Row[0]")
sample_row = train_ds[0]
keys = list(sample_row.keys())
for i, key in enumerate(keys):
    is_last = (i == len(keys) - 1)
    print_value_tree(key, sample_row[key], 0, is_last, "", max_depth=4)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total episodes: {len(train_ds)}")
print(f"Features: {', '.join(train_ds.column_names)}")
