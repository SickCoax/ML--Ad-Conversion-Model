import pandas as pd

df = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\Ad Conversion Model\dataset\digital_marketing_campaign_dataset.csv")

print(len(df))

df = df.groupby("ConversionRate")

print(len(df))