"""Generate a 500-row synthetic transactions dataset for SentinelAI."""
import numpy as np
import pandas as pd

np.random.seed(42)
n_total = 500
n_fraud = 40  # 8% fraud rate - realistic for training
n_legit = n_total - n_fraud

# --- Legitimate transactions ---
legit_amounts = np.random.lognormal(mean=6.5, sigma=1.0, size=n_legit).clip(10, 15000).round(2)
legit_times = np.random.choice(range(7, 22), size=n_legit)  # business hours
legit_location = np.random.binomial(1, 0.05, size=n_legit)   # 5% location change
legit_device = np.random.binomial(1, 0.03, size=n_legit)     # 3% device change
legit_merchant = np.random.beta(2, 20, size=n_legit).round(4) # low risk merchants

# --- Fraudulent transactions ---
fraud_amounts = np.random.lognormal(mean=10.0, sigma=1.2, size=n_fraud).clip(5000, 100000).round(2)
fraud_times = np.random.choice([0, 1, 2, 3, 22, 23], size=n_fraud)  # late night
fraud_location = np.random.binomial(1, 0.75, size=n_fraud)   # 75% location change
fraud_device = np.random.binomial(1, 0.70, size=n_fraud)     # 70% device change
fraud_merchant = np.random.beta(8, 3, size=n_fraud).round(4) # high risk merchants

# --- Combine ---
df = pd.DataFrame({
    "transaction_id": range(1, n_total + 1),
    "amount": np.concatenate([legit_amounts, fraud_amounts]),
    "transaction_time": np.concatenate([legit_times, fraud_times]),
    "location_change": np.concatenate([legit_location, fraud_location]),
    "device_change": np.concatenate([legit_device, fraud_device]),
    "merchant_risk": np.concatenate([legit_merchant, fraud_merchant]),
    "is_fraud": np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)]),
})

# Shuffle rows so fraud isn't all at the bottom
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["transaction_id"] = range(1, n_total + 1)

# Ensure integer types
df["transaction_time"] = df["transaction_time"].astype(int)
df["location_change"] = df["location_change"].astype(int)
df["device_change"] = df["device_change"].astype(int)

df.to_csv("data/transactions.csv", index=False)
print(f"Generated {len(df)} rows ({df['is_fraud'].sum()} fraud, {(~df['is_fraud'].astype(bool)).sum()} legit)")
print(f"Fraud rate: {df['is_fraud'].mean()*100:.1f}%")
print(f"Amount range: ${df['amount'].min():.2f} - ${df['amount'].max():.2f}")
print(f"Saved to data/transactions.csv")
