# VSCode Setup Guide

Guide to set up VSCode for Python development with Pandas.

## 🚀 Installation

### Download VSCode
1. Go to [code.visualstudio.com](https://code.visualstudio.com/)
2. Download for your operating system
3. Install following the setup wizard

### Essential Extensions
Install these extensions for Python development:

1. **Python** (Microsoft)
   - Provides Python language support
   - Install from Extensions tab (Ctrl+Shift+X)

2. **Pylance** (Microsoft)
   - Fast Python language server
   - Auto-installed with Python extension

3. **Python Indent** (Kevin Rose)
   - Better Python indentation
   - Helps with code formatting

4. **Jupyter** (Microsoft)
   - Support for Jupyter notebooks
   - Useful for data analysis

## 🔧 Python Interpreter Setup

### 1. Open VSCode
```bash
# Navigate to your project
cd session-2
code .
```

### 2. Select Python Interpreter
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Python: Select Interpreter"
3. Choose your conda environment: `cse572`

### 3. Verify Setup
Create a test file `test.py`:
```python
import pandas as pd
print("Pandas version:", pd.__version__)
print("Setup successful!")
```

Run with `Ctrl+F5` or right-click → "Run Python File in Terminal"

## 📁 Project Structure
```
session-2/
├── setup/
│   └── vscode_setup.md
├── demo/
│   ├── pandas_basics.py
│   └── pandas_exercises.py
└── data/
    ├── students.csv
    └── sales.csv
```

## 🎯 Tips for Pandas Development

### 1. Use Integrated Terminal
- `Ctrl+`` (backtick) to open terminal
- Run Python scripts directly
- Test pandas operations

### 2. Debugging
- Set breakpoints by clicking line numbers
- Use `F5` to start debugging
- Inspect variables in debug panel

### 3. Code Completion
- VSCode provides autocomplete for pandas
- Use `Ctrl+Space` for suggestions
- Hover over functions for documentation

## 🚨 Common Issues

### Python Not Found
- Make sure conda environment is activated
- Check Python interpreter selection
- Restart VSCode after installing Python

### Pandas Import Error
- Verify pandas is installed: `pip install pandas`
- Check you're using correct environment
- Try running in terminal first

### Extension Issues
- Reload VSCode: `Ctrl+Shift+P` → "Developer: Reload Window"
- Check extension is enabled
- Update extensions regularly

## 📚 Next Steps
1. Open `demo/pandas_basics.py`
2. Run the script to see examples
3. Try `demo/pandas_exercises.py` for practice
4. Experiment with the sample data

---

*Need help? Check VSCode documentation or ask TA.*
