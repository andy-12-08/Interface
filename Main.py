# import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np

# import external functions from utils.py
from utils import add_medium_vertical_space, add_large_vertical_space, convert_time_to_timestamp
from utils import add_vertical_divider, prepare_time_column, rescale_input, rescale_output
from utils import I_O_Plot, foptd_optimization, sim_foptd_model, generate_model_plot
from utils import convert_timestamp_to_minutes, response_plot, plot_closed_loop_response_all
from utils import calculate_pi_controller_with_delay, simulate_closed_loop, simulate_p_control
from utils import calculate_cohen_coon, calculate_zn_pi_parameters, simulate_closed_loop_all
from session_state_init import initialize_session_state

# Set the page configuration
st.set_page_config(layout="wide",
               page_title="GeauxTune",
               page_icon="images/lsu_tiger.png",
               )
html_title = """
<span style="color:purple; font-size:40px; font-weight:bold;">GEAUX</span>
<span style="color:gold; font-size:40px; font-weight:bold;">TUNE</span>
<span style="font-size:30px; margin-left: 20px;"> FOPDT model fitting </span>
"""
st.markdown(html_title, unsafe_allow_html=True)

# initialize the session state variables
initialize_session_state()

# Create the tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = \
    st.tabs(["Data Source","Input/Output","Model fitting","Manual fit","D-S","Z-N","C-C","Compare","Export results"])

with tab1:
    # Create two columns and insert a vertical divider
    col1, col2, col3 = st.columns([0.8, 0.05, 2])  # The middle column for the vertical line is given a small width
    with col1:
        st.image("images/icon4.png",use_column_width=True)
        label_html = """
        <span style="color:gold; font-size:20px; font-weight:bold;">Upload your data file</span>
        """
        # Display the custom-styled label
        st.markdown(f"""
        <div style="margin-bottom: -100px;">{label_html}</div>
        """, unsafe_allow_html=True)
        # Add a file uploader widget
        uploaded_file = st.file_uploader(type=["csv","xlsx"], label=' ')
        # Update the session state with the uploaded file
        st.session_state.uploaded_file = uploaded_file
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
    with col2:
        add_vertical_divider()
    with col3:
        st.markdown("##### Configuration #####")
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.session_state.uploaded_file is not None:
                time_cols = [col for col in df.columns if 'time' in col.lower()] # Get columns with 'time' in the name
                # The rest of the columns are considered as input and output columns
                input_output_cols = [col for col in df.columns if col not in time_cols]
                time_column = st.selectbox("Time column", time_cols, key='time_column')
                # the key parameter ensures that the value is stored in the session state automatically and 
                # can be accessed directly from st.session_state['time_column']
                input_column = st.selectbox("Input column", input_output_cols, key='input_column')
                output_column = st.selectbox("Output column", input_output_cols, key='output_column')
                # Update the session state with the selected columns              
                if st.session_state.input_column is not None:
                    st.session_state.input_variable = df[input_column]
                    st.session_state.min_input = st.session_state.input_variable.min() # default input min value
                    st.session_state.max_input = st.session_state.input_variable.max() # default input max value
                if st.session_state.output_column is not None:
                    st.session_state.output_variable = df[output_column]
                    st.session_state.min_output = st.session_state.output_variable.min() # default output min value
                    st.session_state.max_output = st.session_state.output_variable.max() # default output max value
            else:
                st.selectbox("Time column", [], key='time_column')
                st.selectbox("Input column", [], key='input_column')
                st.selectbox("Output column", [], key='output_column')
            add_medium_vertical_space()
            if st.session_state.uploaded_file is not None:
                if st.session_state['input_column'] or st.session_state['output_column'] is not None:
                    st.markdown("##### Scaling settings #####")
                if st.session_state['input_column'] is not None:
                    if st.session_state.scale_input: # Check if the checkbox is checked (with the key 'scale_input', the widget state is stored in the session state)
                        rescale_input() # Scale the input variable with the user input values for min and max input values 
                    else:
                        st.session_state.input_variable = df[input_column]
                    st.number_input("Input Min", value=st.session_state['min_input'], key='input_min', on_change=rescale_input)
                if st.session_state['output_column'] is not None:
                    if st.session_state.scale_output:
                        rescale_output() # Scale the output variable with the user input values for min and max output values
                    else:
                        st.session_state.output_variable = df[output_column]
                    st.number_input("Output Min", value=st.session_state['min_output'], key='output_min', on_change=rescale_output)  # the input variable is also scaled when the min value is changed  
        with col2:
            if st.session_state.uploaded_file is not None:
                if st.session_state.time_column is not None:
                    st.checkbox("Timestamp", value=False, key='timestamp')
                else:
                    st.checkbox("Timestamp", value=False, key='timestamp', disabled=True)
                st.empty() # Add vertical space
                add_medium_vertical_space()
                if st.session_state.input_column is not None:
                    st.checkbox("Scale input (%)", value=False, key='scale_input')
                    st.empty() # Add vertical space
                    add_medium_vertical_space()
                    if st.session_state.output_column is None:
                        st.checkbox("Scale output (%)", value=False, key='scale_output', disabled=True)
                    else:
                        st.checkbox("Scale output (%)", value=False, key='scale_output')
                    if st.session_state.output_column is None:
                        add_large_vertical_space()
                        add_large_vertical_space()
                        add_large_vertical_space()
                if st.session_state.output_column is not None:
                    add_large_vertical_space()
                    add_large_vertical_space()
                    add_large_vertical_space()
                if st.session_state.input_column is not None:
                    st.number_input("Input Max", value=st.session_state['max_input'], key='input_max', on_change=rescale_input)
                if st.session_state.output_column is not None:
                    st.number_input("Output Max", value=st.session_state['max_output'], key='output_max', on_change=rescale_output)  
            else:
                # Uncheck the checkboxes when the file is not available (# using session state forces it to be unchecked)
                st.session_state.timestamp = False
                st.session_state.data_sec_min = False
                st.session_state.scale_input = False
                st.session_state.scale_output = False
                # Disable the checkboxes when the file is not available
                st.checkbox("Timestamp", value=False, key='timestamp', disabled=True)
                st.empty() # Add vertical space
                add_medium_vertical_space()
                st.checkbox("Scale input (%)", value=False, key='scale_input', disabled=True)
                st.empty() # Add vertical space
                add_medium_vertical_space()
                st.checkbox("Scale output (%)", value=False, key='scale_output', disabled=True)
                add_large_vertical_space()
                add_large_vertical_space()
                add_large_vertical_space()
        with col3:
            if st.session_state.uploaded_file is not None:
                if st.session_state.time_column is not None:
                    st.checkbox("Data in sec (or min)", value=False, key='data_sec_min')
                else:
                    st.checkbox("Data in sec (or min)", value=False, key='data_sec_min', disabled=True)
            else:
               st.checkbox("Data in sec (or min)", value=False, key='data_sec_min', disabled=True)
        if st.session_state.uploaded_file is not None:
            if st.session_state.time_column is not None and st.session_state.input_column is not None and st.session_state.output_column is not None:
                if st.button("Fit model"):
                    st.session_state.fit_model = True
                                           
with tab2:
    if st.session_state.uploaded_file is not None:
        if st.session_state.timestamp: # Check if the timestamp checkbox is checked
            st.session_state.time_variable = convert_time_to_timestamp(time_column_name=time_column, time_col_values=df[time_column])
        if st.session_state.data_sec_min: # Check if the data in sec (or min) checkbox is checked
            st.session_state.time_variable = convert_timestamp_to_minutes(time_col_values=df[time_column])
        if st.session_state.time_column is not None: # Check if the time column is selected
            if st.session_state.timestamp and st.session_state.data_sec_min:
                st.session_state.time_variable = convert_timestamp_to_minutes(time_col_values=df[time_column])
            if not st.session_state.timestamp and not st.session_state.data_sec_min:
                st.session_state.time_variable = convert_timestamp_to_minutes(time_col_values=df[time_column])       
        if st.session_state.time_column is not None and st.session_state.input_column is not None and st.session_state.output_column is not None:
            df_T = st.session_state.time_variable.to_frame(name='Time')
            df_I_O = pd.concat([st.session_state.input_variable, 
                                st.session_state.output_variable], axis=1)
            df = pd.concat([df_T, df_I_O], axis=1)
            df = prepare_time_column(df)
            # Display the input and output variables
            I_O_Plot(df, st.session_state.input_column, 'Input')
            I_O_Plot(df, st.session_state.output_column, 'Output')

with tab3:
    if st.session_state.uploaded_file is not None and st.session_state.time_column is not None and st.session_state.input_column is not None and st.session_state.output_column is not None:
        if st.session_state.fit_model == True: # and st.session_state.K is None and st.session_state.tau is None and st.session_state.theta is None:
            if not st.session_state.timestamp:
                # st.success("Model fitting in progress!")
                st.warning(st.session_state.fit_model_status)
                # Assuming t, yp, and u are numpy arrays from session state
                t = np.array(st.session_state.time_variable)
                yp = np.array(st.session_state.output_variable)
                u = np.array(st.session_state.input_variable)
                # Fit optimized FOTPD model to the data
                x = foptd_optimization(t, yp, u)
                if x[2] < 0:
                    x[2] = 0
                # Display the optimized parameters
                K = x[0]
                tau = x[1] 
                theta = x[2]
                # Store the optimized parameters in the session state
                st.session_state.K = x[0] 
                st.session_state.tau = x[1]
                st.session_state.theta = x[2]
                # Simulate the model using optimized parameters
                ym = sim_foptd_model(t, yp, u, K, tau, theta)
                # calculate the R2 value and RMSE value
                r2 = 1 - np.sum((yp - ym) ** 2) / np.sum((yp - np.mean(yp)) ** 2)
                rmse = np.sqrt(np.mean((yp - ym) ** 2))
                # After fitting is done, update the status
                st.session_state.fit_model_status = "Model fitting completed!"

                # Change the status to success
                st.success(st.session_state.fit_model_status)

                # Retrun the status to the initial state
                st.session_state.fit_model_status = "Model fitting in progress!"

                col1, col2, col3 = st.columns([3, 0.05, 1])
                with col1:
                    generate_model_plot(t,yp,ym,r2,rmse, key='model_plot')
                with col2:
                    add_vertical_divider()
                with col3:
                    # Add some padding at the top to align with the plot
                    st.markdown("<div style='padding-top:30px;'></div>", unsafe_allow_html=True)
                    # Create HTML for the parameters and transfer function
                    K_html = f"<span style='color:blue; font-size:20px; font-weight:bold;'>K = </span> <span style='font-size:20px;'>{K:.4f}</span>"
                    tau_html = f"<span style='color:red; font-size:20px; font-weight:bold;'>τ = </span> <span style='font-size:20px;'>{tau:.4f}</span>"
                    theta_html = f"<span style='color:green; font-size:20px; font-weight:bold;'>θ = </span> <span style='font-size:20px;'>{theta:.4f}</span>"
                    # Transfer function
                    transfer_function_html = f"""
                    <div style='margin-top:10px; font-size:20px; font-weight:bold;'>
                        <span>Transfer Function:</span><br>
                        G(s) = <span style='color:blue;'>K</span> e<sup>-<span style='color:green;'>θ</span>s</sup> / (<span style='color:red;'>τ</span> s + 1)
                    </div>
                    """
                    # Combine with a header for "Fitted Parameters:"
                    st.markdown(f"""
                    <div style='background-color:lightgray; padding:15px; border-radius:10px; max-width:400px; margin:auto;'>
                        <h4 style='font-weight:bold; color:black;'>Fitted Parameters:</h4>
                        <div>{K_html}</div>
                        <div>{tau_html}</div>
                        <div>{theta_html}</div>
                        {transfer_function_html}
                    </div>
                    """, unsafe_allow_html=True)
                st.session_state.fit_model = False
            else:
                st.warning("You need to convert the timestamp to minutes before fitting the model!")
                
with tab4:
    if st.session_state.uploaded_file is not None and st.session_state.time_column is not None and st.session_state.input_column is not None and st.session_state.output_column is not None:
        if not st.session_state.timestamp:

            if st.session_state.K is not None and st.session_state.tau is not None and st.session_state.theta is not None:
                manual_K_value_upper_limit_default = st.session_state.K*1.5
                manual_tau_value_upper_limit_default = st.session_state.tau*1.5
                manual_theta_value_upper_limit_default = st.session_state.theta*1.5 if st.session_state.theta > 0 else 10

                manual_K_value_lower_limit_default = st.session_state.K*0.5
                manual_tau_value_lower_limit_default = st.session_state.tau*0.5
                manual_theta_value_lower_limit_default = st.session_state.theta*0.5
            else:
                manual_K_value_upper_limit_default = 100
                manual_tau_value_upper_limit_default = 100
                manual_theta_value_upper_limit_default = 100

                manual_K_value_lower_limit_default = 0
                manual_tau_value_lower_limit_default = 0
                manual_theta_value_lower_limit_default = 0

            # Assuming t, yp, and u are numpy arrays from session state
            t = np.array(st.session_state.time_variable)
            yp = np.array(st.session_state.output_variable)
            u = np.array(st.session_state.input_variable)

            col1, col2, col3 = st.columns([1, 0.05, 3])  # The middle column for the vertical line is given a small width
            with col1:
                st.write("Configuration")
                if st.session_state.K is None and st.session_state.tau is None and st.session_state.theta is None:
                    st.session_state.k_value = st.slider("K (Gain)", min_value=float(manual_K_value_lower_limit_default), max_value=float(manual_K_value_upper_limit_default), step=0.001, 
                                                    value=float(st.session_state.k_value), key='k_gain')
                    if st.checkbox("Reset K (Gain)"):
                        st.session_state.k_value = 1.0  # Reset to default

                    st.session_state.t_value = st.slider("T (Time constant)", min_value=float(manual_tau_value_lower_limit_default), max_value=float(manual_tau_value_upper_limit_default), step=0.001, 
                                                    value=float(st.session_state.t_value), key='t_constant')
                    if st.checkbox("Reset T (Time constant)"):
                        st.session_state.t_value = 10.0

                    st.session_state.o_value =  st.slider("O (Dead time)", min_value=float(manual_theta_value_lower_limit_default), max_value=float(manual_theta_value_upper_limit_default), step=0.001, 
                                                    value=float(st.session_state.o_value), key='o_dead_time')
                    if st.checkbox("Reset O (Dead time)"):
                        st.session_state.o_value = 1.0                    

                    ym_manual = sim_foptd_model(t, yp, u, st.session_state.k_value, st.session_state.t_value, st.session_state.o_value)
                
                else:
                    st.session_state.k_value = st.slider("K (Gain)", 
                                     min_value=float(manual_K_value_lower_limit_default), 
                                     max_value=float(manual_K_value_upper_limit_default), 
                                     step=0.001, 
                                     value=float(st.session_state.K),
                                     key='k_gain',
                                     )    
                    if st.checkbox("Reset K (Gain)", key='reset_k_gain'):
                        st.session_state.k_value = st.session_state.K  # Reset to default
                        # Use HTML to color the text red
                        st.markdown(
                            f'<p style="color:red;">K value reset = {st.session_state.k_value:.3f}</p>',
                            unsafe_allow_html=True
                        )
                    
                    st.session_state.t_value = st.slider("T (Time constant)",
                                                         min_value=float(manual_tau_value_lower_limit_default),
                                                         max_value=float(manual_tau_value_upper_limit_default),
                                                         step=0.001,
                                                         value=float(st.session_state.tau),
                                                         key='t_constant',
                                                         )
                    if st.checkbox("Reset T (Time constant)", key='reset_t_constant'):
                        st.session_state.t_value = st.session_state.tau
                        st.markdown(
                            f'<p style="color:red;">T value reset = {st.session_state.t_value:.3f}</p>',
                            unsafe_allow_html=True
                        )
                    
                    st.session_state.o_value = st.slider("O (Dead time)",
                                                         min_value=float(manual_theta_value_lower_limit_default),
                                                         max_value=float(manual_theta_value_upper_limit_default),
                                                         step=0.001,
                                                         value=float(st.session_state.theta),
                                                         key='o_dead_time',
                                                         )
                    if st.checkbox("Reset O (Dead time)", key='reset_o_dead_time'):
                        st.session_state.o_value = st.session_state.theta
                        st.markdown(
                            f'<p style="color:red;">O value reset = {st.session_state.o_value:.3f}</p>',
                            unsafe_allow_html=True
                        )
                    ym_manual = sim_foptd_model(t, yp, u, st.session_state.k_value, st.session_state.t_value, st.session_state.o_value)       
                # calculate the R2 value and RMSE value                  
                r2_manual = 1 - np.sum((yp - ym_manual) ** 2) / np.sum((yp - np.mean(yp)) ** 2)
                rmse_manual = np.sqrt(np.mean((yp - ym_manual) ** 2))
                # CSS to change the button background to blue
                button_style = """
                    <style>
                    div.stButton > button {
                        background-color: blue;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 5px;
                        font-size: 16px;
                    }
                    </style>
                    """
                # Inject the custom CSS into the Streamlit app
                st.markdown(button_style, unsafe_allow_html=True)
                if st.button("Save as optimal"):
                    st.session_state.K_optimal = st.session_state.k_value
                    st.session_state.tau_optimal = st.session_state.t_value
                    st.session_state.theta_optimal = st.session_state.o_value
                    st.session_state.save_optimal = True
                    st.success("Configuration saved successfully!")
                # Set default values only if optimal values are not already saved
                if not st.session_state.save_optimal:
                    st.session_state.K_optimal = st.session_state.K
                    st.session_state.tau_optimal = st.session_state.tau
                    st.session_state.theta_optimal = st.session_state.theta

            # Column 2: Add the vertical divider using custom HTML and CSS
            with col2:
                add_vertical_divider()
            with col3:
                generate_model_plot(t,yp,ym_manual,r2_manual,rmse_manual, key='manual_model_plot')   
        else:
            st.warning("You need to convert the timestamp to minutes before fitting the model!")
with tab5:
    if st.session_state.uploaded_file is not None and st.session_state.time_column is not None and st.session_state.input_column is not None and st.session_state.output_column is not None:
        if not st.session_state.timestamp and st.session_state.K is not None and st.session_state.tau is not None and st.session_state.theta is not None:
            col1, col2, col3 = st.columns([1.1, 0.05, 2])
            t = np.array(st.session_state.time_variable)
            with col1:
                # Create HTML for the parameters
                K_optimal_html = f"<span style='color:blue; font-size:15px; font-weight:bold;'>Kp (Process Gain): </span> <span style='font-size:15px;'>{st.session_state.K_optimal:.2f}</span>"
                tau_optimal_html = f"<span style='color:red; font-size:15px; font-weight:bold;'>τ (Time Constant): </span> <span style='font-size:15px;'>{st.session_state.tau_optimal:.2f}</span>"
                theta_optimal_html = f"<span style='color:green; font-size:15px; font-weight:bold;'>θ (Dead Time): </span> <span style='font-size:15px;'>{st.session_state.theta_optimal:.2f}</span>"
                # Combine with a header
                st.markdown(f"""
                <div style='background-color:lightgray; padding:10px; border-radius:10px; max-width:400px; margin:auto;'>
                    <h5 style='font-weight:bold; color:black;margin-bottom:-10px'>Process Model Parameters</h5>
                    <div>{K_optimal_html}</div>
                    <div>{tau_optimal_html}</div>
                    <div>{theta_optimal_html}</div>
                </div>
                """, unsafe_allow_html=True)
                add_medium_vertical_space()
                lambda_value_default = max(st.session_state.theta_optimal, 1.0)
                # Lambda slider for closed-loop time constant
                st.session_state.lambda_value = st.slider(
                    "λ (Desired Closed-Loop Time Constant):", 
                    min_value=0.1, max_value=10.0, step=0.1,
                    value=lambda_value_default,
                )
                # Calculate PI controller parameters using Direct Synthesis method
                st.session_state.Kc_DS, st.session_state.Ti_DS = calculate_pi_controller_with_delay(st.session_state.K_optimal, st.session_state.tau_optimal, st.session_state.theta_optimal, st.session_state.lambda_value)
                Kc_DS_html = f"<span style='color:blue; font-size:15px; font-weight:bold;'>Kc (Controller Gain): </span> <span style='font-size:15px;'>{st.session_state.Kc_DS:.2f}</span>"
                Ti_DS_html = f"<span style='color:red; font-size:15px; font-weight:bold;'>Ti (Integral Time): = </span> <span style='font-size:15px;'>{st.session_state.Ti_DS:.2f}</span>"
                st.markdown(f"""
                <div style='background-color:lightgray; padding:10px; border-radius:10px; max-width:400px; margin:auto;'>
                    <h5 style='font-weight:bold; color:black;margin-top:0px;margin-bottom:-30px'>PI Controller Parameters (Direct Synthesis)</h5>
                    <div>{Kc_DS_html}</div>
                    <div>{Ti_DS_html}</div>
                </div>
                """, unsafe_allow_html=True)
                add_medium_vertical_space()
                sim_time_ds = st.number_input("Simulation Time (min)", value=float(max(t)), key='sim_time_ds')
            with col2:
                add_vertical_divider()
            with col3:
                time, response = simulate_closed_loop(st.session_state.K_optimal, st.session_state.tau_optimal, st.session_state.theta_optimal, st.session_state.Kc_DS, st.session_state.Ti_DS,t,sim_time_ds)
                annotation_text = f"λ = {st.session_state.lambda_value:.2f}"
                plot_title = 'Closed-Loop Response'
                response_plot(time, response, annotation_text, plot_title)
                       
with tab6:
    if st.session_state.uploaded_file is not None and st.session_state.time_column is not None and st.session_state.input_column is not None and st.session_state.output_column is not None:
        if not st.session_state.timestamp and st.session_state.K is not None and st.session_state.tau is not None and st.session_state.theta is not None:
            col1, col2, col3 = st.columns([1.5, 0.05, 3])
            t = np.array(st.session_state.time_variable)
            with col1:
                col1a, col1b = st.columns([0.5, 1])
                with col1a:
                    with st.expander('Bounds'):
                        lower_ku_bound = st.number_input("Low", value=0.01, step=0.01, key='lower_ku_bound')
                        upper_ku_bound = st.number_input("High", value=100.0, step=0.01, key='upper_ku_bound')
                with col1b:
                    if st.session_state.K_optimal >= 0:
                        Ku_slider_zn = st.slider("Ku (Ultimate Gain)", min_value=lower_ku_bound, max_value=upper_ku_bound, step=0.1, key='Ku')
                    else:
                        Ku_slider_zn = st.slider("Ku (Ultimate Gain)", min_value=-100.0, max_value=-0.01, step=0.1, key='Ku')
                    Ku_ZN_html = f"<span style='color:blue; font-size:15px; font-weight:bold;'>Ku (Ultimate Gain): </span> <span style='font-size:15px;'>{Ku_slider_zn:.2f}</span>"       
                # Number input for Pu
                Pu_value = st.number_input("Pu (Ultimate Period)", min_value=0.01, step=0.01, key="Pu")
                Pu_ZN_html = f"<span style='color:red; font-size:15px; font-weight:bold;'>Pu (Ultimate Period): </span> <span style='font-size:15px;'>{Pu_value:.2f}</span>"
                st.markdown(f"""
                <div style='background-color:lightgray; padding:10px; border-radius:10px; max-width:400px; margin:auto;'>
                    <div>{Ku_ZN_html}</div>
                    <div>{Pu_ZN_html}</div>
                </div>
                """, unsafe_allow_html=True)
                sim_time_zn = st.number_input("Simulation Time (min)", value=float(max(t)), key='sim_time_zn')
                if st.button("Calculate PI parameters", key='calculate_pi_parameters_zn'):
                    # Calculate PI parameters using Ziegler-Nichols method
                    st.session_state.Kc_zn, st.session_state.Ti_zn = calculate_zn_pi_parameters(Ku_slider_zn, Pu_value)
                    Kc_zn_html = f"<span style='color:blue; font-size:15px; font-weight:bold;'>Kc (Calculated): </span> <span style='font-size:15px;'>{st.session_state.Kc_zn:.2f}</span>"
                    Ti_zn_html = f"<span style='color:red; font-size:15px; font-weight:bold;'>Ti (Calculated): </span> <span style='font-size:15px;'>{st.session_state.Ti_zn:.2f}</span>"
                    st.markdown(f"""
                    <div style='background-color:lightgray; padding:10px; border-radius:10px; max-width:400px; margin:auto;'>
                        <div>{Kc_zn_html}</div>
                        <div>{Ti_zn_html}</div>
                    </div>
                    """, unsafe_allow_html=True)
            with col2:
                add_vertical_divider()
            with col3:
                time, response =  simulate_p_control(st.session_state.K_optimal, st.session_state.tau_optimal, st.session_state.theta_optimal, Ku_slider_zn,t,sim_time_zn)
                annotation_text = f"Ku = {Ku_slider_zn:.2f}"
                plot_title = 'Closed-Loop Response in P-Contol'
                response_plot(time, response, annotation_text, plot_title)

with tab7:
    if st.session_state.uploaded_file is not None and st.session_state.time_column is not None and st.session_state.input_column is not None and st.session_state.output_column is not None:
        if not st.session_state.timestamp and st.session_state.K is not None and st.session_state.tau is not None and st.session_state.theta is not None:
            # Create HTML for the parameters
            K_optimal_html = f"<span style='color:blue; font-size:15px; font-weight:bold;'>Kp (Process Gain): </span> <span style='font-size:15px;'>{st.session_state.K_optimal:.2f}</span>"
            tau_optimal_html = f"<span style='color:red; font-size:15px; font-weight:bold;'>τ (Time Constant): </span> <span style='font-size:15px;'>{st.session_state.tau_optimal:.2f}</span>"
            theta_optimal_html = f"<span style='color:green; font-size:15px; font-weight:bold;'>θ (Dead Time): </span> <span style='font-size:15px;'>{st.session_state.theta_optimal:.2f}</span>"
            # Combine with a header
            st.markdown(f"""
            <div style='background-color:lightgray; padding:10px; border-radius:10px; max-width:400px; margin:auto;'>
                <h5 style='font-weight:bold; color:black;margin-bottom:-10px'>Process Model Parameters</h5>
                <div>{K_optimal_html}</div>
                <div>{tau_optimal_html}</div>
                <div>{theta_optimal_html}</div>
            </div>
            """, unsafe_allow_html=True)
            add_medium_vertical_space()
            # Centered Streamlit button
            col1, col2, col3 = st.columns([1.5, 1.5, 1.5])  # Create three columns to center the button
            with col2:   
                if st.button("Calculate Cohen-Coon PI parameters"):
                    # Calculate PI controller parameters using Cohen-Coon method
                    st.session_state.Kc_cc, st.session_state.Ti_cc =  calculate_cohen_coon(st.session_state.K_optimal,st.session_state.tau_optimal, st.session_state.theta_optimal)
                    st.success("Cohen-Coon PI parameters calculated successfully!") 
            if st.session_state.Kc_cc is not None and st.session_state.Ti_cc is not None:
                Kc_cc_html = f"<span style='color:blue; font-size:15px; font-weight:bold;'>Kc (Proportional Gain): </span> <span style='font-size:15px;'>{st.session_state.Kc_cc:.2f}</span>"
                Ti_cc_html = f"<span style='color:red; font-size:15px; font-weight:bold;'>Ti (Integral Gain): </span> <span style='font-size:15px;'>{st.session_state.Ti_cc:.2f}</span>"
                st.markdown(f"""
                <div style='background-color:lightgray; padding:10px; border-radius:10px; max-width:400px; margin:auto;'>
                    <h5 style='font-weight:bold; color:black;margin-bottom:-10px'> Cohen-Coon PI Parameters</h5>
                    <div>{Kc_cc_html}</div>
                    <div>{Ti_cc_html}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                Kc_cc_html = f"<span style='color:blue; font-size:15px; font-weight:bold;'>Kc (Proportional Gain): </span> <span style='font-size:15px;'>{st.session_state.Kc_cc}</span>"
                Ti_cc_html = f"<span style='color:red; font-size:15px; font-weight:bold;'>Ti (Integral Gain): </span> <span style='font-size:15px;'>{st.session_state.Ti_cc}</span>"
                st.markdown(f"""
                <div style='background-color:lightgray; padding:10px; border-radius:10px; max-width:400px; margin:auto;'>
                    <h5 style='font-weight:bold; color:black;margin-bottom:-10px'> Cohen-Coon PI Parameters</h5>
                    <div>{Kc_cc_html}</div>
                    <div>{Ti_cc_html}</div>
                </div>
                """, unsafe_allow_html=True)
with tab8:
    if st.session_state.uploaded_file is not None and st.session_state.time_column is not None and st.session_state.input_column is not None and st.session_state.output_column is not None:
        if not st.session_state.timestamp and st.session_state.K is not None and st.session_state.tau is not None and st.session_state.theta is not None:
            if st.session_state.Kc_cc is not None and st.session_state.Ti_cc is not None and st.session_state.Kc_zn is not None and st.session_state.Ti_zn is not None and st.session_state.Kc_DS is not None and st.session_state.Ti_DS is not None:
                t = np.array(st.session_state.time_variable)
                st.markdown("#### Closed Loop Response for All Controllers ####")
                col1, col2, col3 = st.columns([1.5, 0.05, 3])
                with col1:
                    sim_time_all = st.number_input("Simulation Time (min)", value=float(max(t)), key='sim_time_all')
                    if st.button("Compare Controllers", key='compare_controllers'):
                        # simulate the closed loop response for all controllers using simulate_close_loop_all function
                        time_DS, response_DS, time_ZN, response_ZN, time_CC, response_CC = simulate_closed_loop_all(st.session_state.K_optimal, st.session_state.tau_optimal, st.session_state.theta_optimal, 
                                                                                                                    st.session_state.lambda_value, st.session_state.Kc_DS, st.session_state.Ti_DS, 
                                                                                                                    st.session_state.Kc_zn, st.session_state.Ti_zn, 
                                                                                                                    st.session_state.Kc_cc, st.session_state.Ti_cc, 
                                                                                                                    t, sim_time_all)
                with col2:
                    add_vertical_divider()
                with col3:
                        if st.session_state.compare_controllers:
                            # Plot the closed loop response for all controllers
                            plot_closed_loop_response_all(time_DS, response_DS, time_ZN, response_ZN, time_CC, response_CC)
            else: 
                st.error("Please calculate controller parameters first in their respective tabs.")

with tab9:
    st.write("Codes for Exporting results")
