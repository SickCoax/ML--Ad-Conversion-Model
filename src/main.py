import pandas as pd
from train import train_model
from evaluate import eval_model

df = pd.read_csv(r"dataset\digital_marketing_campaign_dataset.csv")

model , X_test , y_test = train_model(df)

y_pred , f1 = eval_model(model , X_test , y_test)

print(f"Model Prediction : {y_pred}")
print()
print(f"F1 Score : {f1}")
