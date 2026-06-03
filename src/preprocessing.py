import pandas as pd

# This dataset does not need any preprocessing steps to be done 
# Also Encoding and Scaling is done in model pipline (train.py)

def get_X_and_y(df) :
    X = df.drop(["Conversion" , "ConversionRate"] , axis=1)
    y = df["Conversion"]
    return X , y