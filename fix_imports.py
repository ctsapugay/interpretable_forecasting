#!/usr/bin/env python3
"""
Script to fix import statements after reorganization
"""

import os
import re

def fix_imports_in_file(filepath, replacements):
    """Fix imports in a single file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        for old_import, new_import in replacements.items():
            content = re.sub(old_import, new_import, content)
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Fixed imports in {filepath}")
        else:
            print(f"ℹ️  No changes needed in {filepath}")
            
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")

def main():
    print("🔧 Fixing import statements...")
    
    # Fix imports in main model files
    main_model_files = [
        "main model/validate_extended_model.py",
        "main model/test_spline_visualization.py"
    ]
    
    main_model_replacements = {
        r"from \.spline_visualization import": "import sys; sys.path.append('../visualizations'); from spline_visualization import",
        r"from spline_visualization import": "import sys; sys.path.append('../visualizations'); from spline_visualization import"
    }
    
    for filepath in main_model_files:
        if os.path.exists(filepath):
            fix_imports_in_file(filepath, main_model_replacements)
    
    # Fix imports in visualization files
    viz_files = [
        "visualizations/integrated_model_visualizations.py",
        "visualizations/visualize_attention.py", 
        "visualizations/visualize_simple.py",
        "visualizations/visualize_cross_attention.py"
    ]
    
    viz_replacements = {
        r"from model import": "import sys; sys.path.append('../main model'); from model import",
        r"from \.model import": "import sys; sys.path.append('../main model'); from model import",
        r"from extended_model import": "import sys; sys.path.append('../main model'); from extended_model import",
        r"from \.extended_model import": "import sys; sys.path.append('../main model'); from extended_model import"
    }
    
    for filepath in viz_files:
        if os.path.exists(filepath):
            fix_imports_in_file(filepath, viz_replacements)
    
    print("✅ Import fixing complete!")

if __name__ == "__main__":
    main()