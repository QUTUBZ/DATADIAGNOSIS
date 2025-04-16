
# import streamlit as st
# import pandas as pd
# from sdv.single_table import GaussianCopulaSynthesizer
# from sdv.metadata import SingleTableMetadata

# st.title("CSV to Synthetic Data Generator")

# # File upload section
# uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# if uploaded_file is not None:
#     # Read original data
#     real_data = pd.read_csv(uploaded_file)
    
#     # Show original data preview
#     st.subheader("Original Data Preview")
#     st.write(f"Shape: {real_data.shape}")
#     st.dataframe(real_data.head())

#     # Synthetic data generation controls
#     st.subheader("Synthetic Data Settings")
#     num_rows = st.slider("Number of synthetic rows", 
#                         min_value=10, 
#                         max_value=len(real_data)*2, 
#                         value=len(real_data))

#     @st.cache_data
#     def train_synthesizer(data):
#         # Auto-detect metadata
#         metadata = SingleTableMetadata()
#         metadata.detect_from_dataframe(data)
        
#         # Create and train synthesizer
#         synthesizer = GaussianCopulaSynthesizer(metadata)
#         synthesizer.fit(data)
#         return synthesizer

#     if st.button("Generate Synthetic Data"):
#         # Train mode
#         synthesizer = train_synthesizer(real_data)
        
#         # Generate synthetic data
#         synthetic_data = synthesizer.sample(num_rows)
        
#         # Show results
#         st.subheader("Generated Synthetic Data")
#         st.write(f"Shape: {synthetic_data.shape}")
#         st.dataframe(synthetic_data.head())

#         # Download capability
#         csv = synthetic_data.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             label="Download Synthetic Data",
#             data=csv,
#             file_name='synthetic_data.csv',
#             mime='text/csv',
#         )

import streamlit as st
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

def clean_target_column(y):
    """Clean target column by stripping whitespace and standardizing case"""
    return y.astype(str).str.strip().str.upper()

def main():
    st.title("Synthetic Data Generator using SMOTE")
    st.write("Automatically generates balanced dataset using default SMOTE parameters")

    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("Data successfully loaded!")
            
            # Show original data
            st.subheader("Original Dataset")
            st.write(f"Shape: {df.shape}")
            st.write(df.head())

            # Select target variable
            target_col = st.selectbox("Select the target variable", df.columns)
            
            # Clean target column
            df[target_col] = clean_target_column(df[target_col])
            
            # Check if target is categorical
            if df[target_col].nunique() / len(df) > 0.05:
                st.error("SMOTE requires categorical target variable. Selected target appears to be continuous.")
                return

            # Show class distribution
            st.subheader("Original Class Distribution")
            class_dist = df[target_col].value_counts()
            st.write(class_dist)
            
            plt.figure(figsize=(10, 4))
            sns.countplot(x=target_col, data=df)
            st.pyplot(plt)

            # Encode target variable
            le = LabelEncoder()
            y_encoded = le.fit_transform(df[target_col])
            classes = le.classes_
            n_classes = len(classes)

            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Identify categorical columns
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()

            # Preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', numeric_features),
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
                ])

            # Process features
            X_processed = preprocessor.fit_transform(X)

            # Auto-configure SMOTE parameters
            default_sampling_strategy = 'auto'  # Balances all minority classes with majority
            default_k_neighbors = 5

            # Apply SMOTE with defaults
            sm = SMOTE(
                sampling_strategy=default_sampling_strategy,
                k_neighbors=default_k_neighbors,
                random_state=42
            )
            X_res, y_res = sm.fit_resample(X_processed, y_encoded)

            # Reconstruct dataframe
            synthetic_df = pd.DataFrame(
                data=preprocessor.named_transformers_['cat'].inverse_transform(
                    X_res[:, len(numeric_features):]
                ),
                columns=categorical_features
            )
            synthetic_df[numeric_features] = X_res[:, :len(numeric_features)]
            synthetic_df[target_col] = le.inverse_transform(y_res)
            synthetic_df = synthetic_df[df.columns]

            # Show results
            st.subheader("Balanced Dataset")
            st.write(f"New shape: {synthetic_df.shape}")
            st.write(synthetic_df.head())

            st.subheader("New Class Distribution")
            new_class_dist = synthetic_df[target_col].value_counts()
            st.write(new_class_dist)
            
            plt.figure(figsize=(10, 4))
            sns.countplot(x=target_col, data=synthetic_df)
            st.pyplot(plt)

            # Download button
            csv = synthetic_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Balanced Data",
                data=csv,
                file_name='balanced_data.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()