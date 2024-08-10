import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from feature_engineering import feature_engineering

# Load dataset
data = pd.read_csv('customer_purchase_history.csv')

# Apply Feature Engineering
data = feature_engineering(data)

# Create Target Variable
data['repeat_purchase'] = data['purchase_count'].apply(lambda x: 1 if x > 1 else 0)

# Drop unnecessary columns
X = data.drop(columns=['repeat_purchase', 'customer_id', 'last_purchase_date', 'total_spent', 'purchase_count'])
y = data['repeat_purchase']

# Preprocessing: Handle Missing Values and Encoding
numeric_features = ['age', 'previous_purchases', 'discount_received', 'days_since_last_purchase', 'avg_purchase_value', 'quantity']
categorical_features = ['gender', 'location', 'price_sensitivity', 'product_line_popularity']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Define Model Pipeline with Hyperparameter Tuning using GridSearchCV
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier(random_state=42))])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameter tuning for RandomForestClassifier
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Use the best estimator found by GridSearchCV
best_model = grid_search.best_estimator_

# Train the best model on the full training data
best_model.fit(X_train, y_train)

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model to a file
joblib.dump(best_model, 'rebuy_model.pkl')
