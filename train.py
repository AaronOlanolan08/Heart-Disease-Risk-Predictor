import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# Load dataset
df = pd.read_csv('dataset/heart_2020_uncleaned.csv')

# Handle missing values
df['BMI'].fillna(df['BMI'].median(), inplace=True)
df['BMI'] = df['BMI'].astype(float)
df['SleepTime'].fillna(df['SleepTime'].median(), inplace=True)
df['SleepTime'] = df['SleepTime'].astype(float)
df['PhysicalHealth'].fillna(df['PhysicalHealth'].mode()[0], inplace=True)
df['SkinCancer'].fillna(df['SkinCancer'].mode()[0], inplace=True)

# Binary encoding
binaryCols = [
    'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 
    'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer'
]

# Lowercase the binary texts for uniformity
for col in binaryCols:
    df[col] = df[col].astype(str).str.lower()

# Map the binary values to make it numerical
binary_map = {
    'yes': 1,
    'no': 0,
    'male': 1,
    'female': 0
}
df[binaryCols] = df[binaryCols].applymap(lambda x: binary_map.get(x, x))

# Features and target
featureCols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
categoricalCols = ['AgeCategory', 'Race', 'Diabetic', 'GenHealth']
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease'].apply(lambda x: 1 if x == 'Yes' else 0)

numeric_transformer = RobustScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, featureCols),      # Apply scaling to numeric features
        ('cat', categorical_transformer, categoricalCols),  # Apply one-hot encoding to categorical features
    ],
    remainder='passthrough'
)

# Wrap the preprocessor in a pipeline for easier integration and future extension
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)  # Only preprocessing for now; model is trained separately
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Train model
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train_processed, y_train)

# Save model and pipeline
joblib.dump(model, 'models/model.joblib')
joblib.dump(pipeline, 'models/preprocessing.joblib')

print("Training complete. Model and preprocessor saved.")
