import streamlit as st

# Set page config
st.set_page_config(
    page_title="App Hub",
    layout="centered"
)

# Page navigation
def navigate_to(page):
    st.session_state.current_page = page

# Main page
def main_page():
    st.title("My App Collection")
    st.write("Choose an app to run:")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Get Data Insights", on_click=navigate_to, args=("Datainsights",))
        st.button("Generate Synthetic data", on_click=navigate_to, args=("Syntheticdata",))
        
    with col2:
        st.button("Clean your Data", on_click=navigate_to, args=("Datacleaning",))
        st.button("Make Predictions", on_click=navigate_to, args=("Dataprediction",))

# Page management
if 'current_page' not in st.session_state:
    st.session_state.current_page = None

if st.session_state.current_page is None:
    main_page()
else:
    if st.session_state.current_page == "Datainsights":
        from pages import Datainsights
        Datainsights.main()
    elif st.session_state.current_page == "Datacleaning":
        from pages import Datacleaning
        Datacleaning.main()
    elif st.session_state.current_page == "Syntheticdata":
        from pages import Syntheticdata
        Syntheticdata.main()
    elif st.session_state.current_page == "Dataprediction":
        from pages import Dataprediction
        Dataprediction.main()

    # Add back button
    if st.button("Back to Main Menu"):
        st.session_state.current_page = None
        st.rerun()  # Correct usage to rerun the app with reset page
