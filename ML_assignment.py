import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Set the title of the app
st.title("Heart Disease Prediction App")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataframe Preview:")
    st.dataframe(df.head())

    # Step 2: Split data into features and target
    if 'target' in df.columns:
        X = df.drop(columns=['target'])
        y = df['target']

        # Step 3: Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Step 5: Define RandomForestClassifier and parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        # Step 6: Fine-Tuning using GridSearchCV
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)

        # Get the best model after tuning
        best_rf = grid_search.best_estimator_

        # Step 7: Model Evaluation on Test Set
        y_pred = best_rf.predict(X_test_scaled)
        y_prob = best_rf.predict_proba(X_test_scaled)[:, 1]

        # Step 8: Display Results
        st.subheader("Model Evaluation Results:")
        
        # Classification Report
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        st.subheader("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)

        # AUC-ROC Score
        roc_score = roc_auc_score(y_test, y_prob)
        st.text(f"AUC-ROC Score: {roc_score:.2f}")

    else:
        st.error("Target column 'target' not found in the uploaded dataset.")
