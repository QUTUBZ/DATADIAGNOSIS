
# import streamlit as st
# import pandas as pd
# from sklearn.impute import SimpleImputer
# import numpy as np
# import joblib
# import plotly.express as px
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     mean_absolute_error, mean_squared_error, r2_score
# )
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.svm import SVC, SVR
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from xgboost import XGBClassifier

# # Page config
# st.set_page_config(page_title="Smart ML Trainer", page_icon="🤖", layout="wide")

# # Session state
# if 'data' not in st.session_state:
#     st.session_state.update({
#         'data': None,
#         'models': {},
#         'best_model': None,
#         'problem_type': None,
#         'feature_ranges': {},
#         'preprocessor': None,
#         'target_encoder': None
#     })

# def load_data(uploaded_file):
#     if uploaded_file.name.endswith('.csv'):
#         return pd.read_csv(uploaded_file)
#     return pd.read_excel(uploaded_file)

# def detect_problem_type(y):
#     unique_values = y.nunique()
#     if unique_values < 10 or y.dtype == 'object':
#         return "Classification"
#     return "Regression"

# def train_model(model, X_train, y_train):
#     model.fit(X_train, y_train)
#     return model

# def evaluate_model(model, X_test, y_test, problem_type):
#     y_pred = model.predict(X_test)
#     metrics = {}
#     if problem_type == "Classification":
#         metrics.update({
#             'accuracy': accuracy_score(y_test, y_pred),
#             'precision': precision_score(y_test, y_pred, average='weighted'),
#             'recall': recall_score(y_test, y_pred, average='weighted'),
#             'f1': f1_score(y_test, y_pred, average='weighted')
#         })
#     else:
#         metrics.update({
#             'mae': mean_absolute_error(y_test, y_pred),
#             'mse': mean_squared_error(y_test, y_pred),
#             'r2': r2_score(y_test, y_pred)
#         })
#     return metrics

# def main():
#     st.title("🧠 Smart ML Training Platform")
#     st.markdown("Automated machine learning with intelligent problem detection")

#     with st.expander("📤 Data Upload", expanded=True):
#         uploaded_file = st.file_uploader("Upload dataset (CSV/Excel)", type=["csv", "xlsx"])
#         if uploaded_file:
#             st.session_state.data = load_data(uploaded_file)
#             st.success("Dataset loaded successfully!")
#             st.write("Data Preview:")
#             st.dataframe(st.session_state.data.head(3))

#     if st.session_state.data is not None:
#         df = st.session_state.data
        
#         with st.expander("🔧 Data Preparation", expanded=True):
#             st.subheader("Configure Prediction Task")
#             target_col = st.selectbox("Select Target Variable", df.columns)
#             feature_cols = st.multiselect("Select Feature Variables", df.columns.drop(target_col))

#             if target_col and feature_cols:
#                 X = df[feature_cols]
#                 y = df[target_col]

#                 st.session_state.problem_type = detect_problem_type(y)
#                 st.info(f"Detected Problem Type: {st.session_state.problem_type}")

#                 if st.session_state.problem_type == "Classification":
#                     le = LabelEncoder()
#                     y = le.fit_transform(y)
#                     st.session_state.target_encoder = le

#                 numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
#                 categorical_features = X.select_dtypes(include=['object', 'category']).columns

#                 st.session_state.feature_ranges = {
#                     col: (X[col].min(), X[col].max()) 
#                     for col in numeric_features
#                 }

#                 numeric_transformer = Pipeline(steps=[
#                     ('imputer', SimpleImputer(strategy='mean')),
#                     ('scaler', StandardScaler())
#                 ])

#                 categorical_transformer = Pipeline(steps=[
#                     ('imputer', SimpleImputer(strategy='most_frequent')),
#                     ('onehot', OneHotEncoder(handle_unknown='ignore'))
#                 ])

#                 preprocessor = ColumnTransformer(
#                     transformers=[
#                         ('num', numeric_transformer, numeric_features),
#                         ('cat', categorical_transformer, categorical_features)
#                     ]
#                 )

#                 st.session_state.preprocessor = preprocessor

#                 X_train, X_test, y_train, y_test = train_test_split(
#                     X, y, test_size=0.2, random_state=42
#                 )

#                 X_train_scaled = preprocessor.fit_transform(X_train)
#                 X_test_scaled = preprocessor.transform(X_test)

#         # Model Training
#         if target_col and feature_cols:
#             with st.expander("🤖 Model Training", expanded=True):
#                 st.subheader("Training Models...")

#                 models = {}
#                 if st.session_state.problem_type == "Classification":
#                     models = {
#                         'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#                         'SVM': SVC(kernel='rbf', probability=True),
#                         'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
#                     }
#                 else:
#                     models = {
#                         'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#                         'Linear Regression': LinearRegression()
#                     }

#                 st.info(f"The following models are being trained: {', '.join(models.keys())}")

#                 progress_bar = st.progress(0)
#                 metrics = []

#                 for i, (name, model) in enumerate(models.items()):
#                     with st.spinner(f"Training {name}..."):
#                         trained_model = train_model(model, X_train_scaled, y_train)
#                         model_metrics = evaluate_model(
#                             trained_model, X_test_scaled, y_test,
#                             st.session_state.problem_type
#                         )
#                         metrics.append({'Model': name, **model_metrics})
#                         st.session_state.models[name] = trained_model
#                         progress_bar.progress((i+1)/len(models))

#                 metrics_df = pd.DataFrame(metrics)
#                 st.subheader("Model Performance Comparison")
#                 if st.session_state.problem_type == "Classification":
#                     fig = px.bar(metrics_df, x='Model', y='accuracy', title='Model Accuracy Comparison')
#                 else:
#                     fig = px.bar(metrics_df, x='Model', y='r2', title='Model R² Score Comparison')
#                 st.plotly_chart(fig)
#                 st.dataframe(metrics_df.style.highlight_max(color='lightgreen', axis=0))

#                 best_metric = 'accuracy' if st.session_state.problem_type == "Classification" else 'r2'
#                 best_model_name = metrics_df.loc[metrics_df[best_metric].idxmax(), 'Model']
#                 st.session_state.best_model = st.session_state.models[best_model_name]
#                 st.success(f"Best Performing Model: {best_model_name}")

#         # Prediction Interface
#         if st.session_state.best_model and st.session_state.preprocessor:
#             with st.expander("🔮 Make Predictions", expanded=True):
#                 st.subheader("Prediction Input")
#                 input_data = {}
#                 cols = st.columns(2)

#                 for i, col in enumerate(feature_cols):
#                     with cols[i % 2]:
#                         if col in st.session_state.feature_ranges:
#                             min_val, max_val = st.session_state.feature_ranges[col]
#                             input_data[col] = st.slider(
#                                 f"{col} (Range: {min_val:.2f}-{max_val:.2f})",
#                                 min_value=float(min_val),
#                                 max_value=float(max_val),
#                                 value=float((min_val + max_val)/2)
#                             )
#                         else:
#                             unique_values = df[col].unique()
#                             if len(unique_values) < 20:
#                                 input_data[col] = st.selectbox(f"Select {col}", unique_values)
#                             else:
#                                 input_data[col] = st.text_input(f"Enter {col}")

#                 if st.button("Predict"):
#                     try:
#                         input_df = pd.DataFrame([input_data])
#                         input_scaled = st.session_state.preprocessor.transform(input_df)
#                         prediction = st.session_state.best_model.predict(input_scaled)

#                         if st.session_state.problem_type == "Classification":
#                             prob = st.session_state.best_model.predict_proba(input_scaled).max()
#                             decoded_pred = st.session_state.target_encoder.inverse_transform(prediction)
#                             st.success(f"Prediction: {decoded_pred[0]} (Confidence: {prob:.2%})")
#                         else:
#                             st.success(f"Predicted Value: {prediction[0]:.2f}")
#                     except Exception as e:
#                         st.error(f"Prediction error: {str(e)}")

# if __name__ == "__main__":
#     main()
import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Page config
st.set_page_config(page_title="Smart ML Trainer", page_icon="🤖", layout="wide")

# Session state
if 'data' not in st.session_state:
    st.session_state.update({
        'data': None,
        'models': {},
        'best_model': None,
        'problem_type': None,
        'feature_ranges': {},
        'preprocessor': None,
        'target_encoder': None
    })

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

def detect_problem_type(y):
    unique_values = y.nunique()
    if unique_values < 10 or y.dtype == 'object':
        return "Classification"
    return "Regression"

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, problem_type):
    y_pred = model.predict(X_test)
    metrics = {}
    if problem_type == "Classification":
        metrics.update({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        })
    else:
        metrics.update({
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        })
    return metrics

def main():
    st.title("🧠 Smart ML Training Platform")
    st.markdown("Automated machine learning with intelligent problem detection")

    with st.expander("📤 Data Upload", expanded=True):
        uploaded_file = st.file_uploader("Upload dataset (CSV/Excel)", type=["csv", "xlsx"])
        if uploaded_file:
            st.session_state.data = load_data(uploaded_file)
            st.success("Dataset loaded successfully!")
            st.write("Data Preview:")
            st.dataframe(st.session_state.data.head(3))

    if st.session_state.data is not None:
        df = st.session_state.data
        
        with st.expander("🔧 Data Preparation", expanded=True):
            st.subheader("Configure Prediction Task")
            target_col = st.selectbox("Select Target Variable", df.columns)
            feature_cols = st.multiselect("Select Feature Variables", df.columns.drop(target_col))

            if target_col and feature_cols:
                X = df[feature_cols]
                y = df[target_col]

                # Silent handling of missing target values
                if y.isna().any():
                    valid_mask = y.notna()
                    X = X[valid_mask]
                    y = y[valid_mask]

                st.session_state.problem_type = detect_problem_type(y)
                st.info(f"Detected Problem Type: {st.session_state.problem_type}")

                if st.session_state.problem_type == "Classification":
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    st.session_state.target_encoder = le

                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                categorical_features = X.select_dtypes(include=['object', 'category']).columns

                st.session_state.feature_ranges = {
                    col: (X[col].min(), X[col].max()) 
                    for col in numeric_features
                }

                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])

                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ]
                )

                st.session_state.preprocessor = preprocessor

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                X_train_scaled = preprocessor.fit_transform(X_train)
                X_test_scaled = preprocessor.transform(X_test)

        # Model Training
        if target_col and feature_cols:
            with st.expander("🤖 Model Training", expanded=True):
                st.subheader("Training Models...")

                models = {}
                if st.session_state.problem_type == "Classification":
                    models = {
                        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                        'SVM': SVC(kernel='rbf', probability=True),
                        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
                    }
                else:
                    models = {
                        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                        'Linear Regression': LinearRegression()
                    }

                st.info(f"The following models are being trained: {', '.join(models.keys())}")

                progress_bar = st.progress(0)
                metrics = []

                for i, (name, model) in enumerate(models.items()):
                    with st.spinner(f"Training {name}..."):
                        trained_model = train_model(model, X_train_scaled, y_train)
                        model_metrics = evaluate_model(
                            trained_model, X_test_scaled, y_test,
                            st.session_state.problem_type
                        )
                        metrics.append({'Model': name, **model_metrics})
                        st.session_state.models[name] = trained_model
                        progress_bar.progress((i+1)/len(models))

                metrics_df = pd.DataFrame(metrics)
                st.subheader("Model Performance Comparison")
                if st.session_state.problem_type == "Classification":
                    fig = px.bar(metrics_df, x='Model', y='accuracy', title='Model Accuracy Comparison')
                else:
                    fig = px.bar(metrics_df, x='Model', y='r2', title='Model R² Score Comparison')
                st.plotly_chart(fig)
                st.dataframe(metrics_df.style.highlight_max(color='lightgreen', axis=0))

                best_metric = 'accuracy' if st.session_state.problem_type == "Classification" else 'r2'
                best_model_name = metrics_df.loc[metrics_df[best_metric].idxmax(), 'Model']
                st.session_state.best_model = st.session_state.models[best_model_name]
                st.success(f"Best Performing Model: {best_model_name}")

        # Prediction Interface
        if st.session_state.best_model and st.session_state.preprocessor:
            with st.expander("🔮 Make Predictions", expanded=True):
                st.subheader("Prediction Input")
                input_data = {}
                cols = st.columns(2)

                for i, col in enumerate(feature_cols):
                    with cols[i % 2]:
                        if col in st.session_state.feature_ranges:
                            min_val, max_val = st.session_state.feature_ranges[col]
                            input_data[col] = st.slider(
                                f"{col} (Range: {min_val:.2f}-{max_val:.2f})",
                                min_value=float(min_val),
                                max_value=float(max_val),
                                value=float((min_val + max_val)/2)
                            )
                        else:
                            unique_values = df[col].unique()
                            if len(unique_values) < 20:
                                input_data[col] = st.selectbox(f"Select {col}", unique_values)
                            else:
                                input_data[col] = st.text_input(f"Enter {col}")

                if st.button("Predict"):
                    try:
                        input_df = pd.DataFrame([input_data])
                        input_scaled = st.session_state.preprocessor.transform(input_df)
                        prediction = st.session_state.best_model.predict(input_scaled)

                        if st.session_state.problem_type == "Classification":
                            prob = st.session_state.best_model.predict_proba(input_scaled).max()
                            decoded_pred = st.session_state.target_encoder.inverse_transform(prediction)
                            st.success(f"Prediction: {decoded_pred[0]} (Confidence: {prob:.2%})")
                        else:
                            st.success(f"Predicted Value: {prediction[0]:.2f}")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()