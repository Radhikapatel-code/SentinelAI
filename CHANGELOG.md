# Changelog

All notable changes to SentinelAI are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-08

### Added
- Unit test suite with pytest (44 tests across 4 test modules)
- GitHub Actions CI pipeline with multi-version Python testing
- Centralized configuration system (`config.py` + `config.yaml`)
- LLM-powered explanation layer with OpenAI integration and fallback
- Model evaluation script for Kaggle Credit Card Fraud dataset
- Decision engine demo script with verbose A* search visualization
- SHAP explanations with descriptive natural language generation
- Health check endpoint (`GET /health`)
- Pydantic response model for API type safety
- Deployment configs for Render (`render.yaml`, `Procfile`)
- Risk level progress bar in Streamlit dashboard
- `__init__.py` for all packages with clean imports
- `LICENSE` (MIT), `CONTRIBUTING.md`, `CHANGELOG.md`
- `.env.example` with documented environment variables

### Changed
- Pinned all dependency versions for reproducibility
- Added type hints to every function signature
- Added Google-style docstrings to all modules and classes
- Improved SHAP text explanations with feature descriptions and values
- Dashboard reads API URL from environment variable
- API uses centralized config for all parameters
- Cost function includes `describe()` method and documented decision boundaries
- Fixed Pydantic V2 deprecation (`dict()` → `model_dump()`)

### Fixed
- Pydantic deprecation warning in API endpoint

## [0.1.0] - Initial Release

### Added
- Isolation Forest anomaly detection
- Logistic Regression fraud classifier
- A* search decision engine with cost function
- SHAP-based explainability
- FastAPI REST API backend
- Streamlit interactive dashboard
- Basic project structure and README
