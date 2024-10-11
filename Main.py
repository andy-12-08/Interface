import streamlit as st

st.set_page_config(layout="wide",
               page_title="GeauxTune",
               page_icon="images/lsu_tiger.png",
               )
html_title = """
<span style="color:purple; font-size:24px; font-weight:bold;">GEAUX</span>
<span style="color:gold; font-size:24px; font-weight:bold;">TUNE</span>
<span style="font-size:18px; margin-left: 20px;"> FOPDT model fitting </span>
"""
st.markdown(html_title, unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = \
    st.tabs(["Data Source","Input/Output","Model fitting","Manual fit","D-S","Z-N","C-C","Compare","Export results"])

# Initialize session state for sliders if not already set
if 'k_value' not in st.session_state:
    st.session_state.k_value = -0.2  # Default value for K Gain
if 't_value' not in st.session_state:
    st.session_state.t_value = 41.8  # Default value for Time constant
if 'o_value' not in st.session_state:
    st.session_state.o_value = 27.5  # Default value for Dead time

with tab1:
    st.write("Codes for Data Source")
with tab2:
    st.write("Codes for I/O implementation")
with tab3:
    st.write("Codes for Model fitting")
with tab4:
    # Create two columns with a 3:1 ratio and insert a vertical divider
    col1, col2, col3 = st.columns([3, 0.05, 1])  # The middle column for the vertical line is given a small width
    # Column 1: Add content (like a plot)
    with col1:
        st.image("images/photo_eraser.png", use_column_width=True)
    # Column 2: Add the vertical divider using custom HTML and CSS
    with col2:
        st.markdown("""
        <style>
            .vertical-line {
                border-left: 2px solid rgba(49, 51, 63, 0.2); /* Light gray vertical line */
                height: 100vh; /* Full height of the viewport */
                margin: 0 auto; /* Center the line vertically */
            }
        </style>
        <div class="vertical-line"></div>
        """, unsafe_allow_html=True) 
    # Column 3: Add the configuration content

    with col3:
        st.write("Configuration")
       # K (Gain) slider with reset checkbox
        st.session_state.k_value = st.slider("K (Gain)", min_value=-1.0, max_value=1.0, step=0.001, 
                                             value=st.session_state.k_value, key='k_gain')
        if st.checkbox("Reset K (Gain)"):
            st.session_state.k_value = 0.0  # Reset to zero
            
        # T (Time constant) slider with reset checkbox
        st.session_state.t_value = st.slider("T (Time constant)", min_value=0.0, max_value=100.0, step=0.1, 
                                             value=st.session_state.t_value, key='t_constant')
        if st.checkbox("Reset T (Time constant)"):
            st.session_state.t_value = 0.0  # Reset to zero

        # O (Dead time) slider with reset checkbox
        st.session_state.o_value = st.slider("O (Dead time)", min_value=0.0, max_value=50.0, step=0.1, 
                                             value=st.session_state.o_value, key='o_dead_time')
        if st.checkbox("Reset O (Dead time)"):
            st.session_state.o_value = 0.0  # Reset to zero

with tab5:
    st.write("Codes for D-S")
with tab6:
    st.write("Codes for Z-N")
with tab7:
    st.write("Codes for C-C")
with tab8:
    st.write("Codes for Compare")
with tab9:
    st.write("Codes for Exporting results")
