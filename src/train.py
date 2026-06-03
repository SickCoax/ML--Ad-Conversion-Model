from preprocessing import get_X_and_y
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score , classification_report


df = pd.read_csv(r"C:\Users\sailj\OneDrive\文档\GitHub\Ad Conversion Model\dataset\digital_marketing_campaign_dataset.csv")
X , y = get_X_and_y(df)

cat_cols = X.select_dtypes(include=["object" , "string"]).columns
num_cols = X.select_dtypes(include=["number"]).columns

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42 , stratify=y)

weights = compute_sample_weight(
    class_weight="balanced" ,
    y = y_train
)

preprocessing = ColumnTransformer([
    ("cat" , OneHotEncoder(handle_unknown="ignore") , cat_cols) ,
    ("num" , StandardScaler() , num_cols)
])

pipeline = Pipeline([
    ("preprocess" , preprocessing) ,
    ("xgbc" , XGBClassifier(
        n_jobs = -1,
        random_state = 42 ,
        subsample = 0.8 ,
        colsample_bytree = 0.8 ,
    ))
])

pipeline.fit(
    X_train ,
    y_train,
    xgbc__sample_weight = weights
)

y_pred = pipeline.predict(X_test)

print(f1_score(y_test , y_pred))
print(classification_report(y_test, y_pred))