# Intraday Prediction Project

## Project Overview
This is an intraday prediction project focused on financial market analysis and prediction.

## Project Structure
- **Working Directory**: `C:\Users\skysn\workspace\intraday_predication`
- **Platform**: Windows (win32)

## Development Environment
- Python-based project (inferred from typical ML/prediction projects)
- Virtual environment likely located at `./venv/` or `../venv/`

## Common Commands

### Environment Setup
```bash
# Activate virtual environment (Windows)
. venv/Scripts/activate
# or
. ./venv/Scripts/activate
# or
. ../venv/Scripts/activate
```

### Dependencies
```bash
# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run tests (check project for specific test framework)
python -m pytest
# or
python -m unittest
```

### Linting & Type Checking
```bash
# Python linting
python -m pylint *.py
# or
python -m flake8
# or
ruff check .

# Type checking
python -m mypy .
```

## Key Areas
- Data processing and analysis
- Model training and evaluation
- Prediction algorithms
- Market data handling

## Notes for AI Assistants
1. Always check for existing code conventions before making changes
2. Verify testing framework before running tests
3. Use appropriate Python virtual environment
4. Follow existing code style and patterns
5. Check for configuration files (config.py, settings.py, .env) for project-specific settings

## Important Reminders
- This is a financial prediction project - ensure data handling follows best practices
- Be cautious with API keys and credentials - never commit them to code
- Validate data inputs and outputs thoroughly
- Consider market hours and timezone handling for intraday predictions