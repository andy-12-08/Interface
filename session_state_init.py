import streamlit as st

def initialize_session_state():
    ## Initialize the necessary session state variables
    if 'fit_model_status' not in st.session_state:
        st.session_state.fit_model_status = "Model fitting in progress!"
    # Initialize session state for file upload status if not already set
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    # Initialize session state for columns if not already set
    if 'time_column' not in st.session_state:
        st.session_state.time_column = None
    if 'input_column' not in st.session_state:
        st.session_state.input_column = None
    if 'output_column' not in st.session_state:
        st.session_state.output_column = None
    # Initialize session state for the input, output and time variables if not already set
    if 'output_variable' not in st.session_state:
        st.session_state.output_variable = None
    if 'input_variable' not in st.session_state:
        st.session_state.input_variable = None
    if 'time_variable' not in st.session_state:
        st.session_state.time_variable = None
    # Initialize session state for the scaling variables if not already set
    if 'min_input' not in st.session_state:
        st.session_state.min_input = None
    if 'max_input' not in st.session_state:
        st.session_state.max_input = None
    if 'min_output' not in st.session_state:
        st.session_state.min_output = None
    if 'max_output' not in st.session_state:
        st.session_state.max_output = None
    # initialize the checkbox state for scaling the input and output variables
    if 'scale_input' not in st.session_state:
        st.session_state.scale_input = False
    if 'scale_output' not in st.session_state:
        st.session_state.scale_output = False
    # Initialize session state for the optimized parameters if not already set
    if 'K' not in st.session_state:
        st.session_state.K = None
    if 'tau' not in st.session_state:
        st.session_state.tau = None
    if 'theta' not in st.session_state:
        st.session_state.theta = None
    # Initialize session state for the optimized parameters if not already set
    if 'K_optimal' not in st.session_state:
        st.session_state.K_optimal = None
    if 'tau_optimal' not in st.session_state:
        st.session_state.tau_optimal = None
    if 'theta_optimal' not in st.session_state:
        st.session_state.theta_optimal = None
    if 'save_optimal' not in st.session_state:
        st.session_state.save_optimal = False
    # Initialize session state for the PI controller parameters if not already set
    if 'Kc_DS' not in st.session_state:
        st.session_state.Kc_DS = None
    if 'Ti_DS' not in st.session_state:
        st.session_state.Ti_DS = None
    # Initialize session state for the lambda value if not already set
    if 'lambda_value' not in st.session_state:
        st.session_state.lambda_value = 1.0
    # Initialize session state for Ziegler-Nichols PI tuning parameters if not already set
    if 'Kc_zn' not in st.session_state:
        st.session_state.Kc_zn = None
    if 'Ti_zn' not in st.session_state:
        st.session_state.Ti_zn = None
    # Initialize session state for Cohen-Coon PI tuning parameters if not already set
    if 'Kc_cc' not in st.session_state:
        st.session_state.Kc_cc = None
    if 'Ti_cc' not in st.session_state:
        st.session_state.Ti_cc = None
    # Initialize session state for sliders if not already set
    if 'k_value' not in st.session_state:
        st.session_state.k_value = 1  # Default value for K Gain before optimization
    if 't_value' not in st.session_state:
        st.session_state.t_value = 10  # Default value for Time constant before optimization
    if 'o_value' not in st.session_state:
        st.session_state.o_value = 1  # Default value for Dead time before optimization
    # Initialize session state for the model fitting status
    if 'fit_model' not in st.session_state:
        st.session_state.fit_model = False
