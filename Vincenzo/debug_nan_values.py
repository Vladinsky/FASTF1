"""
Debug script to identify and fix NaN values in training data
"""

import numpy as np
import torch
from pathlib import Path

def check_for_nans(data_dir="./dataset/preprocessed"):
    """Check all preprocessed files for NaN values"""
    data_path = Path(data_dir)
    
    files_to_check = [
        'X_train.npy', 'X_val.npy', 'X_test.npy',
        'y_change_train.npy', 'y_change_val.npy', 'y_change_test.npy',
        'y_type_train.npy', 'y_type_val.npy', 'y_type_test.npy'
    ]
    
    nan_found = False
    
    for file_name in files_to_check:
        file_path = data_path / file_name
        if file_path.exists():
            data = np.load(file_path)
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            
            print(f"📊 {file_name}:")
            print(f"   Shape: {data.shape}")
            print(f"   NaN values: {nan_count}")
            print(f"   Inf values: {inf_count}")
            print(f"   Min value: {np.nanmin(data)}")
            print(f"   Max value: {np.nanmax(data)}")
            print(f"   Mean: {np.nanmean(data):.4f}")
            print(f"   Std: {np.nanstd(data):.4f}")
            print()
            
            if nan_count > 0 or inf_count > 0:
                nan_found = True
                print(f"⚠️  Found {nan_count} NaN and {inf_count} Inf values in {file_name}")
                
                # Show where NaNs are located
                if nan_count > 0:
                    nan_indices = np.where(np.isnan(data))
                    print(f"   NaN locations (first 10): {list(zip(*nan_indices))[:10]}")
                
                if inf_count > 0:
                    inf_indices = np.where(np.isinf(data))
                    print(f"   Inf locations (first 10): {list(zip(*inf_indices))[:10]}")
                print()
    
    return nan_found

def clean_nan_values(data_dir="./dataset/preprocessed"):
    """Clean NaN and Inf values from data files"""
    data_path = Path(data_dir)
    
    # Files that can have NaN/Inf values
    sequence_files = ['X_train.npy', 'X_val.npy', 'X_test.npy']
    
    for file_name in sequence_files:
        file_path = data_path / file_name
        if file_path.exists():
            print(f"🧹 Cleaning {file_name}...")
            data = np.load(file_path)
            
            # Replace NaN with 0
            nan_mask = np.isnan(data)
            if nan_mask.sum() > 0:
                print(f"   Replacing {nan_mask.sum()} NaN values with 0")
                data[nan_mask] = 0.0
            
            # Replace Inf with large finite values
            inf_mask = np.isinf(data)
            if inf_mask.sum() > 0:
                print(f"   Replacing {inf_mask.sum()} Inf values")
                data[np.isposinf(data)] = np.finfo(np.float32).max / 10
                data[np.isneginf(data)] = np.finfo(np.float32).min / 10
            
            # Clip extreme values
            q99 = np.percentile(data, 99)
            q1 = np.percentile(data, 1)
            
            extreme_mask = (data > q99 * 10) | (data < q1 * 10)
            if extreme_mask.sum() > 0:
                print(f"   Clipping {extreme_mask.sum()} extreme values")
                data = np.clip(data, q1 * 10, q99 * 10)
            
            # Save cleaned data
            backup_path = file_path.with_suffix('.npy.backup')
            if not backup_path.exists():
                np.save(backup_path, np.load(file_path))
                print(f"   Created backup: {backup_path}")
            
            np.save(file_path, data)
            print(f"   ✅ Cleaned and saved {file_name}")

if __name__ == "__main__":
    print("🔍 Checking for NaN/Inf values in preprocessed data...")
    
    nan_found = check_for_nans()
    
    if nan_found:
        print("\n🧹 Cleaning NaN/Inf values...")
        clean_nan_values()
        
        print("\n🔍 Re-checking after cleaning...")
        check_for_nans()
    else:
        print("✅ No NaN/Inf values found in preprocessed data")
