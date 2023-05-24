from src.loadsplit import load_data, split_data
from src.eda import eda
from src.cleandata import cleandata
from src.build_features import make_features
from src.preprocess import preprocess
from src.train_model import train
df = load_data('/Users/rianrachmanto/pypro/project/smoker-detection/data/raw/train_smokin.csv')
print(df.head())
X_train,X_val,y_train,y_val = split_data(df)
print(X_train)
print(X_val)
eda(X_train,y_train)
cleandata(X_train,X_val)
print(X_train,X_val)
make_features(X_train,X_val)
print (X_train,X_val)
X_train_scaled, y_train_sm, X_val_scaled = preprocess(X_train, y_train, X_val)
print(X_train_scaled, y_train_sm, X_val_scaled)
train (X_train_scaled, y_train_sm, X_val_scaled,y_val)