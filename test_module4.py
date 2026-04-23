import pandas as pd

from module4.train import run_training


df = pd.read_csv("titanic.csv")

target_column = "Survived"
X = df.drop(columns=[target_column])
y = df[target_column]

data_contract = {
    "X": X,
    "Y": y,
    "metadata": {
        "column_types": {},
    },
}

result = run_training(data_contract)

print("Metrics:", result["metrics"])
print("Sample Predictions:", result["predictions"][:10])
print("Sample Probabilities:", result["probabilities"][:5])
