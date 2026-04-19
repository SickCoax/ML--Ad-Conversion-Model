from preprocessing import get_X_and_y
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler

df = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\Ad Conversion Model\dataset\digital_marketing_campaign_dataset.csv")
X , y = get_X_and_y(df)

cat_cols = X.select_dtypes(include=["object" , "string"]).columns
num_cols = X.select_dtypes(include=["number"]).columns

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42 , stratify=y)

preprocessing = ColumnTransformer([
    ("cat" , OneHotEncoder(handle_unknown="ignore") , cat_cols) ,
    ("num" , StandardScaler() , num_cols)
])

