# import streamlit as st
# import pandas as pd
# import plotly.express as px

# def main():
#     st.set_page_config(page_title="Dataset Insights", page_icon="📊", layout="wide")

#     # Sidebar - File Upload Section
#     st.sidebar.title("📂 Upload Your Dataset")
#     uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         st.sidebar.success("✅ Dataset loaded successfully!")

#         # Dataset Preview
#         st.title("📊 Dataset Insights Explorer")
#         st.markdown("### Preview of Uploaded Dataset")
#         st.dataframe(df.head(10))

#         # Basic Info Section
#         st.markdown("---")
#         st.subheader("📄 Basic Information")
#         col1, col2, col3 = st.columns(3)
#         col1.metric("🧾 Total Records", df.shape[0])
#         col2.metric("🧾 Total Columns", df.shape[1])
#         col3.metric("⚠️ Missing Values", df.isnull().sum().sum())

#         # Enhanced Summary Statistics for Numeric Columns
#         st.markdown("---")
#         st.subheader("📈 Enhanced Summary of Numeric Columns")
#         numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

#         if len(numeric_cols) > 0:
#             summary_data = {
#                 "Column": [],
#                 "Mean": [],
#                 "Min": [],
#                 "Max": []
#             }

#             for col in numeric_cols:
#                 summary_data["Column"].append(col)
#                 summary_data["Mean"].append(round(df[col].mean(), 2))
#                 summary_data["Min"].append(df[col].min())
#                 summary_data["Max"].append(df[col].max())

#             summary_df = pd.DataFrame(summary_data)

#             with st.expander("🔍 View Summary Table for Numeric Columns"):
#                 st.dataframe(summary_df, height=300, use_container_width=True)

#         else:
#             st.warning("⚠️ No numeric columns found!")

#         # Summary for Categorical Columns
#         st.markdown("---")
#         st.subheader("🔤 Summary of Categorical Columns")
#         categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#         if len(categorical_cols) > 0:
#             for col in categorical_cols:
#                 st.markdown(f"**{col}**")
#                 st.markdown(f"- Unique Values: {df[col].nunique()}")
#                 st.markdown(f"- Most Frequent: {df[col].mode()[0]} ({df[col].value_counts().max()} occurrences)")
#                 with st.expander(f"🔍 View All Classes in {col}"):
#                     class_counts = df[col].value_counts().reset_index().rename(columns={'index': col, col: 'Count'})
#                     st.dataframe(class_counts)
#         else:
#             st.warning("⚠️ No categorical columns found!")

#         # Potential Outliers Section
#         st.markdown("---")
#         st.subheader("🚨 Potential Outliers (3 Std Dev)")
#         outliers_detected = False
#         for col in numeric_cols:
#             upper_limit = df[col].mean() + 3 * df[col].std()
#             lower_limit = df[col].mean() - 3 * df[col].std()
#             outliers = df[(df[col] > upper_limit) | (df[col] < lower_limit)]
#             if not outliers.empty:
#                 outliers_detected = True
#                 with st.expander(f"🔍 View Potential Outliers in {col}"):
#                     st.dataframe(outliers)
        
#         if not outliers_detected:
#             st.success("✅ No potential outliers detected in any column.")

#         # Graph Generator
#         st.markdown("---")
#         st.subheader("📊 Generate Graphs")
#         graph_type = st.selectbox("Select Graph Type", ["Histogram", "Scatter Plot", "Bar Plot (Categorical)", "Pie Chart (Categorical)"])

#         if graph_type == "Histogram":
#             graph_col = st.selectbox("Select a Column for Histogram", numeric_cols)
#             if graph_col and st.button("🔍 Generate Histogram"):
#                 fig = px.histogram(df, x=graph_col, title=f"{graph_col} Distribution")
#                 st.plotly_chart(fig, use_container_width=True)

#         elif graph_type == "Scatter Plot":
#             if len(numeric_cols) >= 2:
#                 scatter_cols = st.multiselect(
#                     "Select X and Y for Scatter Plot",
#                     options=numeric_cols,
#                     default=numeric_cols[:2] if len(numeric_cols) >= 2 else []
#                 )
#                 if len(scatter_cols) == 2:
#                     if st.button("🔍 Generate Scatter Plot"):
#                         fig = px.scatter(df, x=scatter_cols[0], y=scatter_cols[1], title=f"Scatter Plot: {scatter_cols[0]} vs {scatter_cols[1]}")
#                         st.plotly_chart(fig, use_container_width=True)
#                 else:
#                     st.warning("⚠️ Please select exactly 2 numeric columns for a scatter plot.")
#             else:
#                 st.warning("⚠️ At least 2 numeric columns are required to generate a scatter plot!")

#         elif graph_type == "Bar Plot (Categorical)":
#             cat_col = st.selectbox("Select a Categorical Column for Bar Plot", categorical_cols)
#             if cat_col and st.button("🔍 Generate Bar Plot"):
#                 cat_data = df[cat_col].value_counts().reset_index()
#                 cat_data.columns = [cat_col, 'Count']
#                 fig = px.bar(cat_data, x=cat_col, y="Count", title=f"{cat_col} Distribution")
#                 st.plotly_chart(fig, use_container_width=True)

#         elif graph_type == "Pie Chart (Categorical)":
#             cat_col = st.selectbox("Select a Categorical Column for Pie Chart", categorical_cols)
#             if cat_col and st.button("🔍 Generate Pie Chart"):
#                 fig = px.pie(df, names=cat_col, title=f"{cat_col} Distribution")
#                 st.plotly_chart(fig, use_container_width=True)

#         # Download Button
#         st.markdown("---")
#         st.subheader("⬇️ Download Insights Report")
#         if st.button("📥 Generate Report"):
#             summary_report = df.describe(include="all").transpose().to_csv(index=True)
#             st.download_button("📄 Download CSV Report", data=summary_report, file_name="insights_report.csv", mime="text/csv")

#     else:
#         st.info("📂 Please upload a CSV file to generate insights.")

# if __name__ == "__main__":
#     main()
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import base64
from datetime import datetime

def main():
    st.set_page_config(page_title="Dataset Insights", page_icon="📊", layout="wide")

    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #2E4B8B;
        color: white;
        margin: 5px;
    }
    .hover-effect:hover {
        transform: scale(1.02);
        transition: transform 0.2s;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar - Enhanced File Upload Section
    with st.sidebar:
        st.title("📂 Data Explorer")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="Maximum file size: 200MB")
        
        if uploaded_file:
            st.success("✅ Dataset loaded successfully!")
            st.caption(f"File: {uploaded_file.name}")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return

        # Main Container
        with st.container():
            st.title("🔍 Data Insights Explorer")
            
            # Data Overview Section
            st.header("🌐 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card hover-effect">🧾 Total Records<br><h2>{df.shape[0]:,}</h2></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card hover-effect">📑 Total Columns<br><h2>{df.shape[1]}</h2></div>', unsafe_allow_html=True)
            with col3:
                missing = df.isnull().sum().sum()
                st.markdown(f'<div class="metric-card hover-effect" style="background-color: {"#B03A2E" if missing > 0 else "#2E4B8B"}">⚠️ Missing Values<br><h2>{missing:,}</h2></div>', 
                            unsafe_allow_html=True)
            with col4:
                dupes = df.duplicated().sum()
                st.markdown(f'<div class="metric-card hover-effect" style="background-color: {"#B03A2E" if dupes > 0 else "#2E4B8B"}">♻️ Duplicates<br><h2>{dupes:,}</h2></div>', 
                            unsafe_allow_html=True)

            # Reordered Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Data Properties", "📈 Numerical Analysis", "🔤 Categorical Insights", "📊 Advanced Visualizations"])

            # Initialize report data
            report_data = {
                "properties": "",
                "numerical": "",
                "categorical": "",
                "visualizations": ""
            }

            with tab1:  # Data Properties
                st.subheader("⚙️ Dataset Properties")
                prop_data = []
                for col in df.columns:
                    col_info = {
                        "Column": col,
                        "Type": str(df[col].dtype),
                        "Unique Values": df[col].nunique(),
                        "Missing Values": df[col].isnull().sum()
                    }
                    if pd.api.types.is_numeric_dtype(df[col]):
                        col_info.update({
                            "Mean": round(df[col].mean(), 2),
                            "Min": df[col].min(),
                            "Max": df[col].max()
                        })
                    prop_data.append(col_info)
                
                report_data["properties"] = "Data Properties:\n" + pd.DataFrame(prop_data).to_string(index=False)
                st.dataframe(pd.DataFrame(prop_data), use_container_width=True)

            with tab2:  # Numerical Analysis (Original Implementation)
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    st.subheader("📈 Numerical Analysis")
                    
                    # Distribution Analysis
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        selected_num_col = st.selectbox("Select numerical column", numeric_cols)
                        show_boxplot = st.checkbox("Show Box Plot", True)
                        show_histogram = st.checkbox("Show Histogram", True)
                    
                    with col2:
                        if show_boxplot:
                            fig = px.box(df, y=selected_num_col, title=f"📦 {selected_num_col} Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        if show_histogram:
                            fig = px.histogram(df, x=selected_num_col, title=f"📊 {selected_num_col} Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation Matrix
                    st.subheader("📶 Correlation Matrix")
                    corr_matrix = df[numeric_cols].corr()
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='Blues'))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Numerical Summary for Report
                    num_summary = df[numeric_cols].agg(['mean', 'min', 'max']).round(2)
                    report_data["numerical"] = f"Numerical Analysis:\n{num_summary.to_string()}"
                else:
                    st.warning("No numerical columns found in the dataset")

            with tab3:  # Categorical Insights
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if cat_cols:
                    st.subheader("🔤 Categorical Insights")
                    
                    selected_cat_col = st.selectbox("Select categorical column", cat_cols)
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.pie(df, names=selected_cat_col, title=f"🍰 {selected_cat_col} Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.bar(df[selected_cat_col].value_counts(), 
                                   title=f"📊 {selected_cat_col} Value Counts")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Categorical Summary for Report
                    cat_summary = []
                    for col in cat_cols:
                        cat_summary.append({
                            "Column": col,
                            "Unique Values": df[col].nunique(),
                            "Most Common": df[col].mode()[0],
                            "Count": df[col].count()
                        })
                    report_data["categorical"] = "Categorical Insights:\n" + pd.DataFrame(cat_summary).to_string(index=False)
                else:
                    st.warning("No categorical columns found in the dataset")

            with tab4:  # Advanced Visualizations
                st.subheader("📊 Advanced Visualizations")
                viz_type = st.selectbox("Choose visualization type", ["Scatter Plot", "Bar Chart", "3D Scatter Plot"])
                
                try:
                    if viz_type == "Scatter Plot":
                        x_axis = st.selectbox("X-axis", df.columns)
                        y_axis = st.selectbox("Y-axis", df.columns)
                        if x_axis and y_axis:
                            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
                            st.plotly_chart(fig, use_container_width=True)
                            report_data["visualizations"] += f"Scatter Plot: {x_axis} vs {y_axis}\n"

                    elif viz_type == "3D Scatter Plot":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            x_3d = st.selectbox("X-axis", df.columns)
                        with col2:
                            y_3d = st.selectbox("Y-axis", df.columns)
                        with col3:
                            z_3d = st.selectbox("Z-axis", df.columns)
                        fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d)
                        st.plotly_chart(fig, use_container_width=True)
                        report_data["visualizations"] += f"3D Scatter Plot: {x_3d}, {y_3d}, {z_3d}\n"

                    elif viz_type == "Bar Chart":
                        x_axis = st.selectbox("X-axis", df.columns)
                        y_axis = st.selectbox("Y-axis", df.columns)
                        if x_axis and y_axis:
                            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
                            st.plotly_chart(fig, use_container_width=True)
                            report_data["visualizations"] += f"Bar Chart: {x_axis} vs {y_axis}\n"

                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")

            # Enhanced Report Generation
    #         st.markdown("---")
    #         st.subheader("📑 Comprehensive Report")
            
    #         if st.button("📥 Generate Full Report"):
    #             report_content = f"""
    #             =====================
    #             COMPREHENSIVE DATA REPORT
    #             =====================
    #             Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
    #             {report_data['properties']}
                
    #             {report_data['numerical']}
                
    #             {report_data['categorical']}
                
    #             VISUALIZATIONS CREATED:
    #             {report_data['visualizations']}
                
    #             MISSING VALUES SUMMARY:
    #             {df.isnull().sum().to_string()}
    #             """
                
    #             b64 = base64.b64encode(report_content.encode()).decode()
    #             st.markdown(f'<a href="data:file/txt;base64,{b64}" download="full_data_report.txt">📥 Download Full Report</a>', 
    #                       unsafe_allow_html=True)

    # else:
    #     st.info("📤 Please upload a CSV file using the sidebar to begin analysis")

if __name__ == "__main__":
    main()