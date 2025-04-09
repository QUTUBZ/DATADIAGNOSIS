import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Configure page settings
st.set_page_config(page_title="Data Cleaning Platform", page_icon=":bar_chart:", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'removed_outliers' not in st.session_state:
    st.session_state.removed_outliers = None

def convert_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def main():
    st.title("📊 Data Cleaning Platform")
    st.markdown("Upload your dataset and clean it using the options below")

    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.data = df
            st.session_state.cleaned_data = df.copy()
        except Exception as e:
            st.error(f"Error loading file: {e}")

    if st.session_state.data is not None:
        st.subheader("Original Data Preview")
        st.write(st.session_state.data.head())

        if st.checkbox("Show Basic Statistics"):
            st.subheader("Basic Statistics")
            st.write(st.session_state.data.describe())

        st.sidebar.header("Cleaning Options")

        # 1. Handle Missing Values
        with st.sidebar.expander("Handle Missing Values"):
            missing_options = st.selectbox("Choose action for missing values", 
                                         ["Select option", "Drop missing values", "Fill missing values"])
            
            if missing_options == "Drop missing values":
                st.session_state.cleaned_data = st.session_state.cleaned_data.dropna()
            
            elif missing_options == "Fill missing values":
                fill_method = st.radio("Select fill method",
                                     ["Fill with zero", "Fill with mean", "Fill with median", "Fill with mode"])
                
                columns_to_fill = st.multiselect("Select columns to fill", 
                                               st.session_state.cleaned_data.columns)
                
                if columns_to_fill and fill_method:
                    for col in columns_to_fill:
                        if fill_method == "Fill with zero":
                            st.session_state.cleaned_data[col] = st.session_state.cleaned_data[col].fillna(0)
                        elif fill_method == "Fill with mean":
                            st.session_state.cleaned_data[col] = st.session_state.cleaned_data[col].fillna(
                                st.session_state.cleaned_data[col].mean())
                        elif fill_method == "Fill with median":
                            st.session_state.cleaned_data[col] = st.session_state.cleaned_data[col].fillna(
                                st.session_state.cleaned_data[col].median())
                        elif fill_method == "Fill with mode":
                            st.session_state.cleaned_data[col] = st.session_state.cleaned_data[col].fillna(
                                st.session_state.cleaned_data[col].mode()[0])

        # 2. Remove Duplicates
        with st.sidebar.expander("Remove Duplicates"):
            if st.checkbox("Remove duplicate rows"):
                st.session_state.cleaned_data = st.session_state.cleaned_data.drop_duplicates()

        # 3. Column Operations
        with st.sidebar.expander("Column Operations"):
            columns_to_drop = st.multiselect("Select columns to drop",
                                            st.session_state.cleaned_data.columns)
            if columns_to_drop:
                st.session_state.cleaned_data = st.session_state.cleaned_data.drop(columns=columns_to_drop)

        # 4. Range Filtering
        with st.sidebar.expander("🔢 Range Filtering"):
            numeric_columns = st.session_state.cleaned_data.select_dtypes(include=np.number).columns.tolist()
            
            if numeric_columns:
                range_col = st.selectbox("Select column for range filtering", numeric_columns)
                
                if range_col:
                    current_min = st.session_state.cleaned_data[range_col].min()
                    current_max = st.session_state.cleaned_data[range_col].max()
                    
                    st.write(f"Current range: {current_min:.2f} to {current_max:.2f}")
                    
                    new_min, new_max = st.slider(
                        "Select new range",
                        min_value=float(current_min),
                        max_value=float(current_max),
                        value=(float(current_min), float(current_max)),
                        step=0.01
                    )
                    
                    if st.button("Apply Range Filter"):
                        before_count = len(st.session_state.cleaned_data)
                        st.session_state.cleaned_data = st.session_state.cleaned_data[
                            (st.session_state.cleaned_data[range_col] >= new_min) & 
                            (st.session_state.cleaned_data[range_col] <= new_max)
                        ]
                        after_count = len(st.session_state.cleaned_data)
                        removed_count = before_count - after_count
                        st.success(f"Removed {removed_count} rows outside the selected range")
            else:
                st.warning("No numeric columns available for range filtering")

        # 5. Outlier Removal
        with st.sidebar.expander("Outlier Removal"):
            numeric_columns = st.session_state.cleaned_data.select_dtypes(include=np.number).columns.tolist()
            
            if numeric_columns:
                outlier_col = st.selectbox("Select column for outlier removal", numeric_columns)
                method = st.selectbox("Select outlier detection method", 
                                    ["Z-score", "IQR", "Percentile"])
                
                outliers = None
                lower_bound = upper_bound = None
                
                if method == "Z-score":
                    threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.5)
                    z_scores = np.abs((st.session_state.cleaned_data[outlier_col] - 
                                    st.session_state.cleaned_data[outlier_col].mean()) / 
                                    st.session_state.cleaned_data[outlier_col].std())
                    outliers = z_scores > threshold
                    
                elif method == "IQR":
                    iqr_multiplier = st.slider("IQR multiplier", 0.5, 5.0, 1.5, 0.5)
                    Q1 = st.session_state.cleaned_data[outlier_col].quantile(0.25)
                    Q3 = st.session_state.cleaned_data[outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - (iqr_multiplier * IQR)
                    upper_bound = Q3 + (iqr_multiplier * IQR)
                    outliers = (st.session_state.cleaned_data[outlier_col] < lower_bound) | \
                            (st.session_state.cleaned_data[outlier_col] > upper_bound)
                            
                elif method == "Percentile":
                    lower_pct = st.slider("Lower percentile", 0.0, 50.0, 1.0, 0.5)
                    upper_pct = st.slider("Upper percentile", 50.0, 100.0, 99.0, 0.5)
                    lower_bound = st.session_state.cleaned_data[outlier_col].quantile(lower_pct/100)
                    upper_bound = st.session_state.cleaned_data[outlier_col].quantile(upper_pct/100)
                    outliers = (st.session_state.cleaned_data[outlier_col] < lower_bound) | \
                            (st.session_state.cleaned_data[outlier_col] > upper_bound)
                
                if st.button("Remove Outliers"):
                    original_count = len(st.session_state.cleaned_data)
                    st.session_state.removed_outliers = st.session_state.cleaned_data[outliers]
                    st.session_state.cleaned_data = st.session_state.cleaned_data[~outliers]
                    new_count = len(st.session_state.cleaned_data)
                    st.success(f"Removed {original_count - new_count} outlier rows")
                    if method != "Z-score":
                        st.info(f"Bounds used: Lower = {lower_bound:.2f}, Upper = {upper_bound:.2f}")

   

        # Display cleaned data
        st.subheader("Cleaned Data Preview")
        st.write(st.session_state.cleaned_data.head())

        # Display removed outliers
        if st.session_state.removed_outliers is not None:
            if not st.session_state.removed_outliers.empty:
                with st.expander("🔍 View Removed Outlier Rows"):
                    st.write("### Removed Outlier Rows")
                    st.dataframe(st.session_state.removed_outliers)
                    st.download_button(
                        label="Download Removed Outliers as CSV",
                        data=convert_to_csv(st.session_state.removed_outliers),
                        file_name='removed_outliers.csv',
                        mime='text/csv'
                    )

        if st.checkbox("Show Cleaned Data Statistics"):
            st.subheader("Cleaned Data Statistics")
            st.write(st.session_state.cleaned_data.describe())

        csv = convert_to_csv(st.session_state.cleaned_data)
        st.download_button(
            label="Download cleaned data as CSV",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()