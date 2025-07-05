import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import os

# === Load the dataset ===
file_path = "/Users/shubhkalaria/Documents/METIS/Categorization/Categorization_Training_v1_only_Sub_Category.xlsx"
data = pd.read_excel(file_path)

# === Clean column names and handle missing Category column ===
data.columns = data.columns.str.strip()
data.rename(columns={"SubCategory": "Sub Category"}, inplace=True)
if "Category" not in data.columns:
    data["Category"] = "TRANSFER"

# === Ensure required columns are present ===
print("Columns found in Excel:", list(data.columns))
assert all(col in data.columns for col in ['Narration', 'Category', 'Sub Category']), "Missing required columns!"

# === Preprocess narration text ===
data['Narration'] = data['Narration'].fillna("").str.lower()

# === Load Sub Category keyword mapping CSV ===
subcat_keywords_path = "/Users/shubhkalaria/Documents/METIS/Categorization/category_latest.csv"  # 🔁 Update this path
subcat_keywords_df = pd.read_csv(subcat_keywords_path)

# Clean and standardize keyword and sub-category text
subcat_keywords_df['Key_words'] = subcat_keywords_df['Key_words'].str.strip().str.lower()
subcat_keywords_df['Sub Category'] = subcat_keywords_df['Sub Category'].str.strip().str.upper()

# Function to match keywords in narration and assign Sub Category
def assign_sub_category_from_keywords(narration, keyword_df):
    narration = str(narration).lower()
    for _, row in keyword_df.iterrows():
        if row['Key_words'] in narration:
            return row['Sub Category']
    return None

# Apply keyword matching
data['Assigned_Sub_Category'] = data['Narration'].apply(
    lambda x: assign_sub_category_from_keywords(x, subcat_keywords_df)
)

# Override 'Sub Category' where keyword match was found
data['Sub Category'] = data['Assigned_Sub_Category'].combine_first(data['Sub Category'])

# Drop helper column
data.drop(columns=['Assigned_Sub_Category'], inplace=True)

# Fill missing values (if any)
data['Category'] = data['Category'].fillna("TRANSFER")
data['Sub Category'] = data['Sub Category'].fillna("TRANSFER")

# === Encode target labels ===
category_encoder = LabelEncoder()
data['Category_encoded'] = category_encoder.fit_transform(data['Category'])

sub_category_encoder = LabelEncoder()
data['Sub_Category_encoded'] = sub_category_encoder.fit_transform(data['Sub Category'])

# === Prepare features and targets ===
X = data['Narration']
y_category = data['Category_encoded']
y_sub_category = data['Sub_Category_encoded']

# === Train-test split ===
X_train, X_test, y_train_cat, y_test_cat, y_train_sub, y_test_sub = train_test_split(
    X, y_category, y_sub_category, test_size=0.2, random_state=42
)

# === Ensure no NaNs ===
X_train = X_train.fillna("")
X_test = X_test.fillna("")

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === Train XGBoost model for Category ===
category_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    objective='multi:softmax',
    num_class=len(data['Category_encoded'].unique())
)
category_model.fit(X_train_tfidf, y_train_cat)

# === Train XGBoost model for Sub Category ===
sub_category_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    objective='multi:softmax',
    num_class=len(data['Sub_Category_encoded'].unique())
)
sub_category_model.fit(X_train_tfidf, y_train_sub)

# === Make predictions ===
y_pred_cat = category_model.predict(X_test_tfidf)
y_pred_sub = sub_category_model.predict(X_test_tfidf)

# === Decode predictions ===
y_pred_categories = category_encoder.inverse_transform(y_pred_cat)
y_pred_sub_categories = sub_category_encoder.inverse_transform(y_pred_sub)

y_test_categories = category_encoder.inverse_transform(y_test_cat)
y_test_sub_categories = sub_category_encoder.inverse_transform(y_test_sub)

# === Ensure alignment between Category and Sub Category ===
misaligned = []
aligned_y_pred_categories = []
aligned_y_pred_sub_categories = []

for pred_cat, pred_sub, true_cat, true_sub in zip(
    y_pred_categories, y_pred_sub_categories, y_test_categories, y_test_sub_categories
):
    if (
        (true_cat == pred_cat and true_sub == pred_sub) or
        (data[(data['Category'] == pred_cat) & (data['Sub Category'] == pred_sub)].shape[0] > 0)
    ):
        aligned_y_pred_categories.append(pred_cat)
        aligned_y_pred_sub_categories.append(pred_sub)
    else:
        aligned_y_pred_categories.append(true_cat)
        aligned_y_pred_sub_categories.append(true_sub)
        misaligned.append((pred_cat, pred_sub, true_cat, true_sub))

# === Evaluation Reports ===
print("Classification Report for Categories:")
print(classification_report(y_test_categories, aligned_y_pred_categories))

os.makedirs("Model_v6", exist_ok=True)
with open("Model_v6/category_classification_report.txt", "w") as f:
    f.write(classification_report(y_test_categories, aligned_y_pred_categories))

print("Classification Report for Sub-Categories:")
print(classification_report(y_test_sub_categories, aligned_y_pred_sub_categories))

with open("Model_v6/sub_category_classification_report.txt", "w") as f:
    f.write(classification_report(y_test_sub_categories, aligned_y_pred_sub_categories))

if misaligned:
    print("Warning: Some predictions were misaligned and corrected.")
    print(f"Total misalignments: {len(misaligned)}")
else:
    print("All predictions are properly aligned.")

# === Save models and encoders ===
joblib.dump(category_model, "Model_v6/xgboost_category_classifier.pkl")
joblib.dump(sub_category_model, "Model_v6/xgboost_sub_category_classifier.pkl")
joblib.dump(vectorizer, "Model_v6/tfidf_vectorizer.pkl")
joblib.dump(category_encoder, "Model_v6/category_encoder.pkl")
joblib.dump(sub_category_encoder, "Model_v6/sub_category_encoder.pkl")

print("✅ Models, vectorizer, and encoders have been saved.")