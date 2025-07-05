import pandas as pd
import joblib

# Load models and encoders
category_model = joblib.load("Model_v6/xgboost_category_classifier.pkl")
sub_category_model = joblib.load("Model_v6/xgboost_sub_category_classifier.pkl")
vectorizer = joblib.load("Model_v6/tfidf_vectorizer.pkl")
category_encoder = joblib.load("Model_v6/category_encoder.pkl")
sub_category_encoder = joblib.load("Model_v6/sub_category_encoder.pkl")

# Read CSV with 'Narration' column
input_file = "pat1.csv"
df = pd.read_csv(input_file)
df['Narration'] = df['Narration'].fillna("").str.lower()

# Transform narration and predict
X_input = vectorizer.transform(df['Narration'])
y_pred_cat = category_model.predict(X_input)
y_pred_sub = sub_category_model.predict(X_input)

# Decode labels
df['Predicted_Category'] = category_encoder.inverse_transform(y_pred_cat)
df['Predicted_Sub_Category'] = sub_category_encoder.inverse_transform(y_pred_sub)

output_file = input_file.replace(".csv", "_output.csv")
df.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")