# Contributing to SentinelAI

Thank you for considering contributing to SentinelAI! This document describes how to set up a development environment, run tests, and submit changes.

## Development Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Radhikapatel-code/SentinelAI.git
cd SentinelAI

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov httpx
```

### Environment Variables

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

## Running the Project

```bash
# Start the FastAPI backend
uvicorn api.main:app --reload

# In a separate terminal, start the dashboard
streamlit run dashboard/app.py

# Run the decision engine demo
python -m decision_engine.demo
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/test_decision_engine.py

# Run a specific test
pytest tests/test_decision_engine.py::TestAStarDecisionEngine::test_high_risk_transaction_blocked
```

## Code Style

- **Type hints**: Every function must have input and return type annotations
- **Docstrings**: Google-style docstrings on all public classes and functions
- **Imports**: Use absolute imports from the project root
- **Config**: All thresholds and parameters go in `config.yaml`, not hardcoded

## Commit Message Convention

Use clear, descriptive commit messages with conventional prefixes:

```
feat: add configurable cost weights to A* engine
fix: handle edge case in SHAP explainer for zero values
test: add unit tests for decision engine
docs: update README with performance metrics
refactor: extract feature columns to module-level constant
ci: add GitHub Actions workflow
chore: pin dependency versions
```

## Pull Request Workflow

1. Create a feature branch from `main`: `git checkout -b feat/your-feature`
2. Make your changes with clear, atomic commits
3. Run tests: `pytest --cov=.`
4. Push and open a PR against `main`
5. Link the relevant GitHub Issue in the PR description

## Project Structure

```
SentinelAI/
├── api/                  # FastAPI backend
├── models/               # ML models (Isolation Forest, Classifier)
├── decision_engine/      # A* search decision engine
├── explainability/       # SHAP + LLM explanation layer
├── dashboard/            # Streamlit frontend
├── tests/                # pytest test suite
├── notebooks/            # Evaluation scripts
├── data/                 # Transaction datasets
├── config.py             # Centralized configuration
├── config.yaml           # Configuration values
└── requirements.txt      # Dependencies
```

## Questions?

Open a GitHub Issue or reach out at rp773061@gmail.com.
