import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load the dataset
print("Loading data...")
df = pd.read_csv('archive/exoplanets_data.csv')

def calculate_habitability_index(row):
    """
    Calculate a habitability index (0-100) based on multiple factors:
    1. Temperature (20-30 points)
    2. Size (15-25 points)
    3. Orbital Period (15-25 points)
    4. Insolation Flux (15-25 points)
    """
    # Temperature score (20-30 points)
    # Ideal temperature range: 0-50°C (273-323K)
    temp = row['koi_teq']
    if 273 <= temp <= 323:
        temp_score = 30
    elif 250 <= temp < 273 or 323 < temp <= 350:
        temp_score = 20
    else:
        temp_score = 10

    # Size score (15-25 points)
    # Ideal size: 0.8-2.5 Earth radii
    size = row['koi_prad']
    if 0.8 <= size <= 2.5:
        size_score = 25
    elif 0.5 <= size < 0.8 or 2.5 < size <= 4:
        size_score = 15
    else:
        size_score = 10

    # Orbital Period score (15-25 points)
    # Ideal period: 200-400 days
    period = row['koi_period']
    if 200 <= period <= 400:
        period_score = 25
    elif 100 <= period < 200 or 400 < period <= 600:
        period_score = 15
    else:
        period_score = 10

    # Insolation Flux score (15-25 points)
    # Ideal flux: 0.8-1.2 times Earth's flux
    flux = row['koi_insol']
    if 0.8 <= flux <= 1.2:
        flux_score = 25
    elif 0.5 <= flux < 0.8 or 1.2 < flux <= 1.5:
        flux_score = 15
    else:
        flux_score = 10

    # Calculate total score
    total_score = temp_score + size_score + period_score + flux_score
    
    # Normalize to 0-100 scale
    normalized_score = (total_score / 105) * 100
    
    return normalized_score

# Feature engineering
def engineer_features(df):
    # Create new features
    df['koi_ratio'] = df['koi_prad'] / df['koi_srad']  # Planet to star radius ratio
    df['koi_density'] = df['koi_prad'] ** 3 / df['koi_period'] ** 2  # Rough density estimate
    df['koi_habitable'] = (df['koi_teq'] > 200) & (df['koi_teq'] < 400)  # Rough habitable zone estimate
    df['koi_habitable'] = df['koi_habitable'].astype(int)
    
    # Calculate habitability index
    df['habitability_index'] = df.apply(calculate_habitability_index, axis=1)
    
    return df

# Prepare features and target
df = engineer_features(df)

# Select most important features
features = [
    'koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 
    'koi_steff', 'koi_srad', 'koi_ratio', 'koi_density', 'koi_habitable'
]

X = df[features]
y = df['koi_disposition']

# Handle missing values
print("Preprocessing data...")
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define simplified parameter grid for GridSearchCV
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}

# Create and train model with GridSearchCV
print("Training model...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=3,
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_grid.predict(X_test_scaled)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_grid.best_estimator_.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Exoplanet Classification')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Function to predict if a new observation is an exoplanet
def predict_exoplanet(period, prad, teq, insol, steff, srad):
    # Create a single observation with only the features used in training
    observation = pd.DataFrame({
        'koi_period': [period],
        'koi_prad': [prad],
        'koi_teq': [teq],
        'koi_insol': [insol],
        'koi_steff': [steff],
        'koi_srad': [srad]
    })
    
    # Add engineered features
    observation['koi_ratio'] = observation['koi_prad'] / observation['koi_srad']
    observation['koi_density'] = observation['koi_prad'] ** 3 / observation['koi_period'] ** 2
    observation['koi_habitable'] = ((observation['koi_teq'] > 200) & (observation['koi_teq'] < 400)).astype(int)
    
    # Calculate habitability index separately
    habitability = calculate_habitability_index(observation.iloc[0])
    
    # Handle any missing values
    observation = pd.DataFrame(imputer.transform(observation), columns=observation.columns)
    
    # Scale the observation
    observation_scaled = scaler.transform(observation)
    
    # Make prediction
    prediction = rf_grid.predict(observation_scaled)
    probability = rf_grid.predict_proba(observation_scaled)
    return prediction[0], probability[0], habitability

# Example prediction
print("\nExample Prediction:")
example = {
    'period': 10.0,
    'prad': 2.5,
    'teq': 800.0,
    'insol': 100.0,
    'steff': 5500.0,
    'srad': 1.0
}

prediction, probability, habitability = predict_exoplanet(
    example['period'],
    example['prad'],
    example['teq'],
    example['insol'],
    example['steff'],
    example['srad']
)

print(f"Prediction: {'Confirmed Exoplanet' if prediction == 2 else 'Candidate' if prediction == 1 else 'False Positive'}")
print("Probabilities:")
print(f"False Positive: {probability[0]:.2%}")
print(f"Candidate: {probability[1]:.2%}")
print(f"Confirmed: {probability[2]:.2%}")
print(f"Habitability Index: {habitability:.2f}/100")

# Print best parameters
print("\nBest Parameters:")
print(rf_grid.best_params_)

# Print top 10 most habitable planets
print("\nTop 10 Most Habitable Planets:")
top_habitable = df[df['koi_disposition'] == 2].nlargest(10, 'habitability_index')
print(top_habitable[['kepoi_name', 'habitability_index', 'koi_prad', 'koi_teq', 'koi_period', 'koi_insol']])

# Plot habitability distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df[df['koi_disposition'] == 2], x='habitability_index', bins=30)
plt.title('Distribution of Habitability Index for Confirmed Exoplanets')
plt.xlabel('Habitability Index')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('habitability_distribution.png')
plt.close() 