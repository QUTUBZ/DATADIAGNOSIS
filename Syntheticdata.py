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
#         # Train model
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
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

st.title("CSV to Synthetic Data Generator")

# File upload section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read original data
    real_data = pd.read_csv(uploaded_file)

    # Show original data preview
    st.subheader("Original Data Preview")
    st.write(f"Shape: {real_data.shape}")
    st.dataframe(real_data.head())

    # Synthetic data generation controls
    st.subheader("Synthetic Data Settings")
    num_rows = st.slider("Number of synthetic rows",
                         min_value=10,
                         max_value=len(real_data)*2,
                         value=len(real_data))

    @st.cache_data
    def train_synthesizer(data):
        # Auto-detect metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)

        # Create and train synthesizer
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(data)
        return synthesizer

    if st.button("Generate Synthetic Data"):
        # Train model
        synthesizer = train_synthesizer(real_data)

        # Generate synthetic data
        synthetic_data = synthesizer.sample(num_rows)

        # Add row_type labels
        real_data_labeled = real_data.copy()
        real_data_labeled['row_type'] = 'original'

        synthetic_data['row_type'] = 'synthetic'

        # Combine both
        combined_data = pd.concat([real_data_labeled, synthetic_data], ignore_index=True)

        # Show results
        st.subheader("Combined Dataset (Original + Synthetic)")
        st.write(f"Shape: {combined_data.shape}")
        st.dataframe(combined_data.head())

        # Download capability
        csv = combined_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Combined Data",
            data=csv,
            file_name='combined_data.csv',
            mime='text/csv',
        )
