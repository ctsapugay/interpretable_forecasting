#!/bin/bash

echo "🔄 Starting project reorganization..."

# Create directories
mkdir -p "main model"
mkdir -p "visualizations" 
mkdir -p "readmes"

echo "📁 Created directories"

# Move main model files
mv extended_model.py "main model/" 2>/dev/null
mv model.py "main model/" 2>/dev/null
mv train_extended_model.py "main model/" 2>/dev/null
mv validate_extended_model.py "main model/" 2>/dev/null
mv validate_model.py "main model/" 2>/dev/null
mv test_*.py "main model/" 2>/dev/null
mv data_splitting.py "main model/" 2>/dev/null
mv data_utils.py "main model/" 2>/dev/null
mv evaluation_utils.py "main model/" 2>/dev/null

echo "🤖 Moved main model files"

# Move visualization files
mv visualize_*.py "visualizations/" 2>/dev/null
mv integrated_model_visualizations.py "visualizations/" 2>/dev/null
mv spline_visualization.py "visualizations/" 2>/dev/null

echo "📊 Moved visualization files"

# Move README files
mv README_*.md "readmes/" 2>/dev/null
mv ETT_README.md "readmes/" 2>/dev/null
mv FORECASTING_*.md "readmes/" 2>/dev/null
mv modules1-2.md "readmes/" 2>/dev/null

echo "📚 Moved documentation files"

# Move image folders to graphs (except img folder)
mv spline_validation_outputs graphs/ 2>/dev/null
mv training_outputs graphs/ 2>/dev/null

echo "🖼️ Moved image folders to graphs"

# Clean up any leftover files
rm -f *.txt *.png 2>/dev/null

echo "🧹 Cleaned up temporary files"

echo "✅ Reorganization complete!"
echo ""
echo "📂 New structure:"
echo "   main model/ - Core model code and utilities"
echo "   visualizations/ - All plotting and visualization scripts"
echo "   readmes/ - Documentation files"
echo "   graphs/ - All generated images and outputs"
echo "   ETT-small/ - Dataset (unchanged)"
echo "   GAM/ - Teammate folder (unchanged)"
echo "   img/ - Images folder (unchanged)"