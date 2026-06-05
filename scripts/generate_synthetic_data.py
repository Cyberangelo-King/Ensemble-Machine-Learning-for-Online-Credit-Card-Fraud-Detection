import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_data(output_path: Path, n_samples=10000):
    np.random.seed(42)
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    data = np.random.randn(n_samples, len(columns))
    df = pd.DataFrame(data, columns=columns)

    # Adjust Time and Amount to be positive
    df['Time'] = np.arange(n_samples)
    df['Amount'] = np.random.exponential(scale=100, size=n_samples)

    # Create imbalanced classes (0.17% fraud as per README)
    df['Class'] = 0
    fraud_indices = np.random.choice(n_samples, int(n_samples * 0.0017), replace=False)
    df.loc[fraud_indices, 'Class'] = 1

    # Make some features predictive for class 1
    # Typically V17, V14, V12, V10 are important
    for col in ['V17', 'V14', 'V12', 'V10']:
        df.loc[df['Class'] == 1, col] -= 3.0

    df.to_csv(output_path, index=False)
    print(f"Synthetic data generated at {output_path}")

if __name__ == "__main__":
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    generate_synthetic_data(data_dir / "creditcard.csv")
