import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_data(output_path: Path, n_samples: int = 10000, fraud_ratio: float = 0.0017):
    np.random.seed(42)

    # Generate Time and Amount
    time = np.sort(np.random.uniform(0, 172792, n_samples))
    amount = np.random.exponential(scale=88, size=n_samples)

    # Generate V1-V28
    v_features = np.random.normal(loc=0, scale=1, size=(n_samples, 28))

    # Generate Class (imbalanced)
    n_fraud = int(n_samples * fraud_ratio)
    if n_fraud == 0: n_fraud = 1
    classes = np.zeros(n_samples, dtype=int)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    classes[fraud_indices] = 1

    # Combine into DataFrame
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    data = np.column_stack((time, v_features, amount, classes))
    df = pd.DataFrame(data, columns=columns)
    df['Class'] = df['Class'].astype(int)

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path} with {n_samples} samples and {n_fraud} fraud cases.")

if __name__ == "__main__":
    generate_synthetic_data(Path("data/creditcard.csv"))
