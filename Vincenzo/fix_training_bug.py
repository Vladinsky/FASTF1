"""
Quick patch to fix the validation metrics bug in training_utils.py
"""

import re

def fix_training_utils():
    """Fix the missing 'loss' key in validation metrics"""
    
    file_path = "./dataset/models/training_utils.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the validate_epoch method ending
    pattern = r"(optimal_threshold, threshold_metrics = find_optimal_threshold\(\s*predictions, targets, self\.target_recall, metric='f1'\s*\)\s*)(return threshold_metrics, optimal_threshold)"
    
    replacement = r"\1# Add validation loss to threshold metrics\n        val_loss = np.mean(val_metrics.losses)\n        threshold_metrics['loss'] = val_loss\n        \n        \2"
    
    # Apply the fix
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    if new_content != content:
        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Fixed validation metrics bug in training_utils.py")
        return True
    else:
        print("⚠️  Pattern not found, trying alternative fix...")
        
        # Alternative approach - find the return statement and add loss before it
        pattern2 = r"(\s+return threshold_metrics, optimal_threshold)"
        replacement2 = r"        \n        # Add validation loss to threshold metrics\n        val_loss = np.mean(val_metrics.losses)\n        threshold_metrics['loss'] = val_loss\n\1"
        
        new_content2 = re.sub(pattern2, replacement2, content)
        
        if new_content2 != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content2)
            print("✅ Applied alternative fix for validation metrics bug")
            return True
        else:
            print("❌ Could not apply fix automatically")
            return False

if __name__ == "__main__":
    print("🔧 Applying fix for training validation metrics bug...")
    fix_training_utils()
