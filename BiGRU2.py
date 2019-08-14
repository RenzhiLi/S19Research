import numpy as np
from sklearn.externals import joblib
doc = joblib.load("text_emb_doc.pkl")
query = joblib.load("query_emb.pkl")