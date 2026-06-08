"""
SentinelAI Centralized Configuration.

Provides typed configuration for all system components using dataclasses.
Supports loading from config.yaml with sensible defaults as fallback.

Usage:
    from config import get_config
    config = get_config()
    print(config.cost.fraud_loss)  # 1000.0
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Optional YAML support — falls back to defaults if PyYAML not installed
try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for ML models.

    Attributes:
        contamination: Expected proportion of anomalies in the dataset.
            Higher values make the Isolation Forest flag more transactions.
        n_estimators: Number of trees in the Isolation Forest ensemble.
        random_state: Random seed for reproducibility.
    """

    contamination: float = 0.2
    n_estimators: int = 100
    random_state: int = 42


@dataclass(frozen=True)
class CostConfig:
    """Configuration for the decision engine cost function.

    Real-world fraud cost model:
        - fraud_loss: Cost of a false negative (allowing fraud through).
          In production, this equals the transaction amount.
        - false_positive_cost: Cost of blocking a legitimate transaction.
          Includes customer service calls ($5-10) + churn risk ($5-15).
        - review_cost: Fixed cost per manual review case.
          Includes analyst time (~$20 per case).

    Attributes:
        fraud_loss: Expected loss when a fraudulent transaction is approved.
        false_positive_cost: Cost of blocking a legitimate transaction.
        review_cost: Fixed cost of escalating to manual review.
    """

    fraud_loss: float = 1000.0
    false_positive_cost: float = 50.0
    review_cost: float = 20.0


@dataclass(frozen=True)
class RiskThresholds:
    """Thresholds for risk categorization.

    Attributes:
        high_risk_threshold: Fraud probability above this → high risk.
        medium_risk_threshold: Fraud probability above this → medium risk.
            Below this → low risk.
    """

    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.3


@dataclass(frozen=True)
class APIConfig:
    """Configuration for the FastAPI backend.

    Attributes:
        host: Host address to bind the API server.
        port: Port number for the API server.
        data_path: Path to the transaction data CSV file.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    data_path: str = "data/transactions.csv"


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the optional LLM explanation layer.

    Attributes:
        enabled: Whether to attempt LLM-based explanations.
        api_key: OpenAI API key (read from OPENAI_API_KEY env var).
        model: The LLM model identifier to use.
        max_tokens: Maximum tokens in the LLM response.
    """

    enabled: bool = True
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    max_tokens: int = 150


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration.

    Aggregates all component configs into a single immutable object.

    Attributes:
        model: ML model configuration.
        cost: Decision engine cost weights.
        risk: Risk categorization thresholds.
        api: API server configuration.
        llm: LLM explanation layer configuration.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    risk: RiskThresholds = field(default_factory=RiskThresholds)
    api: APIConfig = field(default_factory=APIConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def _load_yaml(config_path: str) -> dict:
    """Load and parse a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        dict: Parsed YAML content, or empty dict if file not found
        or PyYAML not installed.
    """
    if not _HAS_YAML:
        return {}

    path = Path(config_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if isinstance(data, dict) else {}


def get_config(config_path: str = "config.yaml") -> AppConfig:
    """Load application configuration.

    Merges values from (highest priority first):
    1. Environment variables (for API keys, host overrides)
    2. config.yaml file (if present)
    3. Dataclass defaults

    Args:
        config_path: Path to the YAML config file. Defaults to 'config.yaml'.

    Returns:
        AppConfig: Fully resolved application configuration.
    """
    data = _load_yaml(config_path)

    model_data = data.get("model", {})
    cost_data = data.get("cost", {})
    risk_data = data.get("risk", {})
    api_data = data.get("api", {})
    llm_data = data.get("llm", {})

    # Environment variable overrides
    api_host = os.environ.get("API_HOST", api_data.get("host", "127.0.0.1"))
    api_port = int(
        os.environ.get("API_PORT", api_data.get("port", 8000))
    )
    openai_key = os.environ.get(
        "OPENAI_API_KEY", llm_data.get("api_key")
    )

    return AppConfig(
        model=ModelConfig(
            contamination=model_data.get("contamination", 0.2),
            n_estimators=model_data.get("n_estimators", 100),
            random_state=model_data.get("random_state", 42),
        ),
        cost=CostConfig(
            fraud_loss=cost_data.get("fraud_loss", 1000.0),
            false_positive_cost=cost_data.get("false_positive_cost", 50.0),
            review_cost=cost_data.get("review_cost", 20.0),
        ),
        risk=RiskThresholds(
            high_risk_threshold=risk_data.get("high_risk_threshold", 0.7),
            medium_risk_threshold=risk_data.get(
                "medium_risk_threshold", 0.3
            ),
        ),
        api=APIConfig(
            host=api_host,
            port=api_port,
            data_path=api_data.get("data_path", "data/transactions.csv"),
        ),
        llm=LLMConfig(
            enabled=llm_data.get("enabled", True),
            api_key=openai_key,
            model=llm_data.get("model", "gpt-4o-mini"),
            max_tokens=llm_data.get("max_tokens", 150),
        ),
    )
