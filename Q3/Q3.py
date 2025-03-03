import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
file_path = "/mnt/data/Copy of for_all2 - Memphis.csv"
df = pd.read_csv(file_path)

# Selecting relevant columns for the vulnerability model
features = [
    "Median household income (in US dollars)", "Households with no vehicles",
    "Population with Bachelor's degree or higher", "Households with one or more people 65 years and over",
    "Households with one or more people under 18 years", "Number of households",
    "Apartments", "Mobile Homes/Other", "Homes built 1950 to 1969", "Homes built 1950 or earlier"]

df_selected = df[features]

# Handling missing values
df_selected = df_selected.fillna(df_selected.median())

# Normalize features
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=features)

# Creating the target variable: Categorizing vulnerability into 3 levels (Low, Medium, High)
num_bins = 3  # Categorize into 3 groups
labels = ["Low", "Medium", "High"]
df_scaled["Vulnerability_Score"] = df_scaled.mean(axis=1)
df_scaled["Vulnerability_Level"] = pd.qcut(df_scaled["Vulnerability_Score"], q=num_bins, labels=labels)

# Splitting data into training and testing sets
X = df_scaled[features]
y = df_scaled["Vulnerability_Level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Assigning predictions back to neighborhoods
df["Predicted_Vulnerability_Level"] = model.predict(X)
print(df[["Neighborhood", "Predicted_Vulnerability_Level"]])
