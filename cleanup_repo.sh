#!/bin/bash
# cleanup_repo.sh - Clean up DFText repository

echo "🧹 DFText Repository Cleanup Script"
echo "===================================="
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: Not in DFText directory (app.py not found)"
    echo "Please run this script from the DFText root directory"
    exit 1
fi

echo "📍 Current directory: $(pwd)"
echo ""

# Confirm before proceeding
read -p "⚠️  This will delete files. Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🗑️  Step 1: Removing virtual environment..."
if [ -d "venv310" ]; then
    rm -rf venv310/
    echo "   ✓ Deleted venv310/"
else
    echo "   ℹ️  venv310/ not found (already clean)"
fi

if [ -d "venv" ]; then
    rm -rf venv/
    echo "   ✓ Deleted venv/"
fi

if [ -d ".venv" ]; then
    rm -rf .venv/
    echo "   ✓ Deleted .venv/"
fi

echo ""
echo "🧹 Step 2: Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
echo "   ✓ Cleaned Python cache"

echo ""
echo "🧹 Step 3: Removing IDE files..."
rm -rf .vscode/ .idea/ .DS_Store 2>/dev/null
find . -name ".DS_Store" -delete 2>/dev/null
find . -name "*.swp" -delete 2>/dev/null
echo "   ✓ Cleaned IDE files"

echo ""
echo "🧹 Step 4: Cleaning uploads directory..."
if [ -d "uploads" ]; then
    rm -f uploads/* 2>/dev/null
    touch uploads/.gitkeep
    echo "   ✓ Cleaned uploads/ (kept .gitkeep)"
else
    mkdir -p uploads
    touch uploads/.gitkeep
    echo "   ✓ Created uploads/ directory"
fi

echo ""
echo "📁 Step 5: Creating necessary directories..."
mkdir -p user_feedback plots logs
touch user_feedback/.gitkeep plots/.gitkeep logs/.gitkeep
echo "   ✓ Created user_feedback/, plots/, logs/"

echo ""
echo "📝 Step 6: Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
*.egg
*.egg-info/
dist/
build/
.venv/
venv/
venv310/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Project specific
uploads/*
!uploads/.gitkeep
logs/*
!logs/.gitkeep
user_feedback/*.json
plots/*.png

# Secrets
.env
*.key
*.pem
config_secret.py

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
Thumbs.db
EOF
echo "   ✓ Created .gitignore"

echo ""
echo "📋 Step 7: Summary of repository..."
echo ""
echo "Directory structure:"
tree -L 2 -I 'venv*|__pycache__|*.pyc' 2>/dev/null || find . -maxdepth 2 -type d | grep -v '^\./\.' | head -20

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 Repository statistics:"
echo "   Files: $(find . -type f | grep -v '.git' | wc -l)"
echo "   Size: $(du -sh . | cut -f1)"
echo ""
echo "🚀 Next steps:"
echo "   1. Review changes: git status"
echo "   2. Stage all: git add ."
echo "   3. Commit: git commit -m 'chore: Clean up repository'"
echo "   4. Push: git push origin main"
echo ""
echo "💡 Tip: Run 'python app.py' to test everything works!"