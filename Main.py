import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

st.set_page_config(layout="wide",
               page_title="GeauxTune",
               page_icon="images/lsu_tiger.png",
               )
html_title = """
<span style="color:purple; font-size:40px; font-weight:bold;">GEAUX</span>
<span style="color:gold; font-size:40px; font-weight:bold;">TUNE</span>
<span style="font-size:30px; margin-left: 20px;"> FOPDT model fitting </span>
"""
def add_medium_vertical_space():
    # Add vertical space between widgets
    st.markdown("""
        <style>
        .medium_spacer {
            height: 10px;  /* Adjust this value to control the space */
        }
        </style>
    """, unsafe_allow_html=True)
    return st.markdown("<div class='medium_spacer'></div>", unsafe_allow_html=True)

def add_large_vertical_space():
    # Add vertical space between widgets
    st.markdown("""
        <style>
        .small_spacer {
            height: 22px;  /* Adjust this value to control the space */
        }
        </style>
    """, unsafe_allow_html=True)
    return st.markdown("<div class='small_spacer'></div>", unsafe_allow_html=True)

def convert_time_to_timestamp(time_column_name, time_col_values):
    if time_col_values.dtypes == 'object' or time_col_values.dtypes == 'datetime64[ns]' or time_col_values.dtypes == 'timedelta64[ns]':
        return time_col_values
    else:
        converted_time = []  # List to store the converted time values
        # Check if the time is in hours
        if 'hr' in time_column_name.lower() or 'hour' in time_column_name.lower():
            for t in time_col_values:
                h = int(t)  # Calculate hours
                m = int((t - h) * 60)  # Calculate minutes from the remainder
                converted_time.append(f'{h:02d}:{m:02d}:00')  # Format time as HH:MM:SS
        # Check if the time is in minutes
        elif 'min' in time_column_name.lower():
            for t in time_col_values:
                h = int(t // 60)  # Calculate hours
                m = int(t % 60)   # Calculate minutes
                s = int((t - h * 60 - m) * 60)  # Calculate seconds from the remainder
                converted_time.append(f'{h:02d}:{m:02d}:{s:02d}')  # Format time as HH:MM:SS
        # If neither hours nor minutes are detected, assume the time is in seconds
        else:
            for t in time_col_values:  
                h = int(t // 3600)  # Convert seconds to hours
                m = int((t % 3600) // 60)  # Convert the remainder into minutes
                s = int(t % 60)  # Remaining seconds
                converted_time.append(f'{h:02d}:{m:02d}:{s:02d}')  # Format time as HH:MM:SS
        # Convert the list to a new pandas Series
        converted_time = pd.Series(converted_time)
        # Convert to dataframe with the same column name
        # converted_time = pd.DataFrame({time_column_name: converted_time})
        return converted_time

def convert_timestamp_to_minutes(time_col_values):
    if time_col_values.dtypes == 'object' or time_col_values.dtypes == 'datetime64[ns]' or time_col_values.dtypes == 'timedelta64[ns]':
        converted_time = []
        for time_val in time_col_values:
            time_val = time_val.split(':') 
            # Convert the split entities to float for the minutes calculation
            hours = float(time_val[0])
            minutes = float(time_val[1])
            seconds = float(time_val[2])
            # Convert the timestamp to total minutes
            total_minutes = hours * 60 + minutes + seconds / 60
            converted_time.append(total_minutes)
            # convert to series
            converted_time_series = pd.Series(converted_time)
            # Make the series relative to the first time value
            converted_time_series = converted_time_series - converted_time_series.iloc[0]
        return converted_time_series
    else:
        return time_col_values

def prepare_time_column(df):
    if df['Time'].dtypes == 'object':
        def convert_time(value):
            try:
                # Try to convert with microseconds (fractional seconds)
                return pd.to_datetime(value, format='%H:%M:%S.%f')
            except ValueError:
                # Fallback to convert without microseconds
                return pd.to_datetime(value, format='%H:%M:%S')
        # Apply conversion and rounding
        df['Time'] = df['Time'].apply(convert_time)
        # Round the seconds to two decimal places (100 milliseconds)
        df['Time'] = df['Time'].dt.round('100ms')
        # Format the result as 'HH:MM:SS.SS' (up to 2 decimal places for fractional seconds)
        df['Time'] = df['Time'].dt.strftime('%H:%M:%S.%f').str.rstrip('0').str.rstrip('.')
        return df
    else:
        return df

def I_O_Plot(df, column_name, I_O):
    fig = go.Figure()
    # If 'Time' is an object, convert it to datetime
    if df['Time'].dtypes == 'object':
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    # Add the trace for the input/output column with custom color and labels
    fig.add_trace(go.Scatter(
        x=df['Time'], 
        y=df[column_name], 
        mode='lines+markers',  # Add markers to improve visibility
        name=I_O,  # This adds "Input" or "Output" as the legend label
        showlegend=True,  # Ensure the legend is shown
        line=dict(width=3, color='purple' if I_O == 'Input' else 'gold'),  # Custom colors for Input and Output
        marker=dict(size=5)  # Adjust marker size for better visibility
    ))
    # Check if 'Time' is datetime and update the layout accordingly
    if df['Time'].dtype == 'datetime64[ns]':
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title=f'{I_O} Variable ({column_name})',
            margin=dict(t=0, b=0, l=0, r=0),  # Adjust margins to control space around the plot
            xaxis=dict(
                type='date',
                tickformat="%H:%M:%S",  # Format the ticks to show time as HH:MM:SS
                title=dict(
                    text='Time',
                    font=dict(size=16, family='Arial Black', color='black')  # Larger, bold x-axis title
                ),
                tickfont=dict(size=14, family='Arial', color='black')  # Larger x-axis tick labels
            ),
            yaxis=dict(
                title=dict(
                    text=f'{I_O} Variable ({column_name})',
                    font=dict(size=16, family='Arial Black', color='black')  # Larger, bold y-axis title
                ),
                tickfont=dict(size=14, family='Arial', color='black')  # Larger y-axis tick labels
            ),
            plot_bgcolor='lightgray',  # Set plot background color
            font=dict(size=12),  # General font size
            height=400,  # Set height of the plot      
            # Legend settings
            legend=dict(
                font=dict(size=14),  # Increase legend font size
                x=0.9,  # Move it to the right
                y=0.95,  # Place it towards the top
                xanchor='right',  # Anchor it to the right
                yanchor='top',  # Anchor it to the top
                bgcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent background
                bordercolor='black',  # Add border for clarity
                borderwidth=1  # Border width for the legend
            )
        )
    else:
        # If 'Time' is not datetime, fall back to default layout
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title=f'{I_O} Variable ({column_name})',
            margin=dict(t=0, b=0, l=0, r=0),  # Adjust margins to control space around the plot
            xaxis=dict(
                title=dict(
                    text='Time',
                    font=dict(size=16, family='Arial Black', color='black')  # Larger, bold x-axis title
                ),
                tickfont=dict(size=14, family='Arial', color='black')  # Larger x-axis tick labels
            ),
            yaxis=dict(
                title=dict(
                    text=f'{I_O} Variable ({column_name})',
                    font=dict(size=16, family='Arial Black', color='black')  # Larger, bold y-axis title
                ),
                tickfont=dict(size=14, family='Arial', color='black')  # Larger y-axis tick labels
            ),
            plot_bgcolor='lightgray',  # Set plot background color
            font=dict(size=12),  # General font size
            height=400,  # Set height of the plot            
            # Legend settings
            legend=dict(
                font=dict(size=14),  # Increase legend font size
                x=0.9,  # Move it to the right
                y=0.95,  # Place it towards the top
                xanchor='right',  # Anchor it to the right
                yanchor='top',  # Anchor it to the top
                bgcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent background
                bordercolor='black',  # Add border for clarity
                borderwidth=1  # Border width for the legend
            )
        )
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def rescale_input():
    if st.session_state.scale_input:
        st.session_state.input_variable = 100*((st.session_state.input_variable - st.session_state.input_min)/(st.session_state.input_max - st.session_state.input_min))

def rescale_output():
    if st.session_state.scale_output:
        st.session_state.output_variable = 100*((st.session_state.output_variable - st.session_state.output_min)/(st.session_state.output_max - st.session_state.output_min))
    

st.markdown(html_title, unsafe_allow_html=True)

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

if 'K' not in st.session_state:
    st.session_state.K = None
if 'tau' not in st.session_state:
    st.session_state.tau = None
if 'theta' not in st.session_state:
    st.session_state.theta = None

# Initialize session state for sliders if not already set
if 'k_value' not in st.session_state:
    st.session_state.k_value = 1  # Default value for K Gain
if 't_value' not in st.session_state:
    st.session_state.t_value = 10  # Default value for Time constant
if 'o_value' not in st.session_state:
    st.session_state.o_value = 1  # Default value for Dead time

if 'k_slider_reset' not in st.session_state:
    st.session_state.k_slider_reset = None
if 't_slider_reset' not in st.session_state:
    st.session_state.t_slider_reset = None
if 'o_slider_reset' not in st.session_state:
    st.session_state.o_slider_reset = None



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
        st.markdown("""
        <style>
            .vertical-line {
                border-left: 0px solid rgba(49, 51, 63, 0.2); /* Light gray vertical line */
                height: 100vh; /* Full height of the viewport */
                margin: 1 auto; /* Center the line vertically */
            }
        </style>
        <div class="vertical-line"></div>
        """, unsafe_allow_html=True) 
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
                if st.button("Fit model", key='fit_model'):
                    # st.warning("Model fitting has not been implemented yet!")
                    st.warning("Model is about to run!")
                          
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
        if st.session_state.fit_model:
            if not st.session_state.timestamp:

                # st.success("Model fitting in progress!")
                st.warning(st.session_state.fit_model_status)

                # Assuming t, yp, and u are numpy arrays from session state
                t = np.array(st.session_state.time_variable)
                yp = np.array(st.session_state.output_variable)
                u = np.array(st.session_state.input_variable)

                # scale input from 0 to 1 
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                # u = scaler.fit_transform(u.reshape(-1,1)).flatten()

                # Initialize u0, yp0, and ns
                u0 = u[0]
                yp0 = yp[0]

                # specify number of steps
                ns = len(t)
                # delta_t = t[1]-t[0]
                # create linear interpolation of the u data versus time
                uf = interp1d(t,u)

                # define first-order plus dead-time approximation    
                def fopdt(y,t,uf,Km,taum,thetam):
                    # arguments
                    #  y      = output
                    #  t      = time
                    #  uf     = input linear function (for time shift)
                    #  Km     = model gain
                    #  taum   = model time constant
                    #  thetam = model time constant
                    # time-shift u
                    try:
                        if (t-thetam) <= 0:
                            um = uf(0.0)
                        else:
                            um = uf(t-thetam)
                    except:
                        #print('Error with time extrapolation: ' + str(t))
                        um = u0
                    # calculate derivative
                    dydt = (-(y-yp0) + Km * (um-u0))/taum
                    return dydt

                # simulate FOPDT model with x=[Km,taum,thetam]
                def sim_model(x):
                    # input arguments
                    Km = x[0]
                    taum = x[1]
                    thetam = x[2]
                    # storage for model values
                    ym = np.zeros(ns)  # model
                    # initial condition
                    ym[0] = yp0
                    # loop through time steps    
                    for i in range(0,ns-1):
                        ts = [t[i],t[i+1]]
                        y1 = odeint(fopdt,ym[i],ts,args=(uf,Km,taum,thetam))
                        ym[i+1] = y1[-1]
                    return ym

                # define objective
                def objective(x):
                    # simulate model
                    ym = sim_model(x)
                    # calculate objective
                    obj = 0.0
                    for i in range(len(ym)):
                        obj = obj + (ym[i]-yp[i])**2    
                    # return result
                    return obj

                # initial guesses
                x0 = np.zeros(3)
                x0[0] = 1 # Km
                x0[1] = 10 # taum
                x0[2] = 1 # thetam

                # optimize Km, taum, thetam
                solution = minimize(objective,x0)

                # Another way to solve: with bounds on variables
                bnds = ((0, None), (0, None), (0.0, None))
                # solution = minimize(objective,x0,bounds=bnds,method='L-BFGS-B')
                x = solution.x

                # Display the optimized parameters
                K = x[0]
                tau = x[1] 
                theta = x[2]
                # Store the optimized parameters in the session state
                st.session_state.K = x[0] 
                st.session_state.tau = x[1]
                st.session_state.theta = x[2]

                # store a copy of the optimized parameters for resetting the sliders
                st.session_state.k_slider_reset = x[0]
                st.session_state.t_slider_reset = x[1]
                st.session_state.o_slider_reset = x[2]

                # Simulate the model using optimized parameters
                ym = sim_model(x)

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
                    fig = go.Figure()
                    # Add traces
                    fig.add_trace(go.Scatter(x=t, y=ym, mode='lines+markers', name='Model Prediction', 
                                            line=dict(width=3, color='black', dash='dash'), 
                                            marker=dict(size=1)))
                    fig.add_trace(go.Scatter(x=t, y=yp, mode='lines', name='Actual Output', 
                                            line=dict(width=3, color='blue')))

                    # Add annotation for R2 and RMSE
                    fig.add_annotation(
                    text=f"R² = {r2:.4f}<br>RMSE = {rmse:.4f}",
                    xref="paper", yref="paper", x=0.05, y=0.95, showarrow=False,
                    font=dict(size=14, family="Arial", color="black"),
                    bordercolor="black", borderwidth=1, borderpad=4, bgcolor="lightgray", opacity=0.8
                    )
                    # Update layout
                    fig.update_layout(
                        xaxis_title='Time', 
                        yaxis_title='Output', 
                        plot_bgcolor='lightgray', 
                        font=dict(size=16, family='Arial Black', color='black'), 
                        height=400,  # Increase the height of the plot
                        margin=dict(t=0, b=0, l=0, r=0),  # Adjust margins to control space around the plot
                        xaxis=dict(
                            tickfont=dict(size=14, family='Arial', color='black'),
                            title_font=dict(size=16, family='Arial Black', color='black')
                        ),
                        yaxis=dict(
                            tickfont=dict(size=14, family='Arial', color='black'),
                            title_font=dict(size=16, family='Arial Black', color='black')
                        ),
                        legend=dict(
                            x=0.75,  # Position the legend closer to the right edge (0 to 1 scale)
                            y=0.1,   # Position the legend near the bottom of the plot (0 to 1 scale)
                            bgcolor='rgba(255,255,255,0.5)',  # Transparent background for the legend
                            bordercolor='black',
                            borderwidth=1
                        )
                    )
                    # Render plotly chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

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


            else:
                st.warning("You need to convert the timestamp to minutes before fitting the model!")
                

with tab4:
    if st.session_state.uploaded_file is not None and st.session_state.time_column is not None and st.session_state.input_column is not None and st.session_state.output_column is not None:
        if not st.session_state.timestamp:
            # Create two columns with a 3:1 ratio and insert a vertical divider

            if st.session_state.K is not None and st.session_state.tau is not None and st.session_state.theta is not None:
                manual_K_value_default = st.session_state.K
                manual_tau_value_default = st.session_state.tau
                manual_theta_value_default = st.session_state.theta

                manual_K_value_upper_limit_default = st.session_state.k_slider_reset*100
                manual_tau_value_upper_limit_default = st.session_state.t_slider_reset*100
                manual_theta_value_upper_limit_default = (st.session_state.o_slider_reset*10 + 100)

            else:
                manual_K_value_default = st.session_state.k_value
                manual_tau_value_default = st.session_state.t_value
                manual_theta_value_default = st.session_state.o_value

                manual_K_value_upper_limit_default = 100
                manual_tau_value_upper_limit_default = 100
                manual_theta_value_upper_limit_default = 100


            # Assuming t, yp, and u are numpy arrays from session state
            t = np.array(st.session_state.time_variable)
            yp = np.array(st.session_state.output_variable)
            u = np.array(st.session_state.input_variable)

            # scale input from 0 to 1 
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            # u = scaler.fit_transform(u.reshape(-1,1)).flatten()

            # Initialize u0, yp0, and ns
            u0 = u[0]
            yp0 = yp[0]

            # specify number of steps
            ns = len(t)
            delta_t = t[1]-t[0]
            # create linear interpolation of the u data versus time
            uf = interp1d(t,u)
        

            # define first-order plus dead-time approximation    
            def fopdt(y,t,uf,Km,taum,thetam):
                # arguments
                #  y      = output
                #  t      = time
                #  uf     = input linear function (for time shift)
                #  Km     = model gain
                #  taum   = model time constant
                #  thetam = model time constant
                # time-shift u
                try:
                    if (t-thetam) <= 0:
                        um = uf(0.0)
                    else:
                        um = uf(t-thetam)
                except:
                    #print('Error with time extrapolation: ' + str(t))
                    um = u0
                # calculate derivative
                dydt = (-(y-yp0) + Km * (um-u0))/taum
                return dydt
                
            # simulate FOPDT model with x=[Km,taum,thetam]
            def sim_model_manual(Km,taum,thetam):

                ns = len(t)
                # storage for model values
                ym = np.zeros(ns)  # model
                # initial condition
                ym[0] = yp0
                # loop through time steps    
                for i in range(0,ns-1):
                    ts = [t[i],t[i+1]]
                    y1 = odeint(fopdt,ym[i],ts,args=(uf,Km,taum,thetam))
                    ym[i+1] = y1[-1]
                return ym

            col1, col2, col3 = st.columns([1, 0.05, 3])  # The middle column for the vertical line is given a small width
            # Column 1: Add content (like a plot)
            with col1:
                st.write("Configuration")

                if st.session_state.K is None and st.session_state.tau is None and st.session_state.theta is None:

                    st.session_state.k_value = st.slider("K (Gain)", min_value=0.0, max_value=float(manual_K_value_upper_limit_default), step=0.1, 
                                                    value=float(st.session_state.k_value), key='k_gain')
                    if st.checkbox("Reset K (Gain)"):
                        st.session_state.k_value = 1.0  # Reset to default

                    st.session_state.t_value = st.slider("T (Time constant)", min_value=0.0, max_value=float(manual_tau_value_upper_limit_default), step=0.1, 
                                                    value=float(st.session_state.t_value), key='t_constant')
                    if st.checkbox("Reset T (Time constant)"):
                        st.session_state.t_value = 10.0

                    st.session_state.o_value =  st.slider("O (Dead time)", min_value=0.0, max_value=float(manual_theta_value_upper_limit_default), step=0.1, 
                                                    value=float(st.session_state.o_value), key='o_dead_time')
                    if st.checkbox("Reset O (Dead time)"):
                        st.session_state.o_value = 1.0                    

                    ym_manual = sim_model_manual(st.session_state.k_value, st.session_state.t_value, st.session_state.o_value)
                
                else:
                    st.session_state.K = st.slider("K (Gain)", min_value=0.0, max_value=float(manual_K_value_upper_limit_default), step=0.1, 
                                                    value= float(st.session_state.K), key='k_gain')
                    if st.checkbox("Reset K (Gain)"):
                        st.session_state.K = st.session_state.k_slider_reset  # Reset to default

                    st.session_state.tau = st.slider("T (Time constant)", min_value=0.0, max_value=float(manual_tau_value_upper_limit_default), step=0.1,
                                                    value=float(st.session_state.tau), key='t_constant')
                    if st.checkbox("Reset T (Time constant)"):
                        st.session_state.tau = st.session_state.t_slider_reset

                    st.session_state.theta =  st.slider("O (Dead time)", min_value=0.0, max_value=float(manual_theta_value_upper_limit_default), step=0.1, 
                                                    value=float(st.session_state.theta), key='o_dead_time')
                    if st.checkbox("Reset O (Dead time)"):
                        st.session_state.theta = st.session_state.o_slider_reset

                    ym_manual = sim_model_manual(st.session_state.K, st.session_state.tau, st.session_state.theta)
                

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
                    st.success("Configuration saved successfully!")


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

            with col3:
                # Plot the actual output and fitted output using plotly
                fig = go.Figure()
                # Add traces
                fig.add_trace(go.Scatter(x=t, y=ym_manual, mode='lines+markers', name='Model Prediction', 
                                        line=dict(width=3, color='black', dash='dash'), 
                                        marker=dict(size=1)))
                fig.add_trace(go.Scatter(x=t, y=yp, mode='lines', name='Actual Output', 
                                        line=dict(width=3, color='blue')))
                # Update layout
                fig.update_layout(
                    xaxis_title='Time', 
                    yaxis_title='Output', 
                    plot_bgcolor='lightgray', 
                    font=dict(size=16, family='Arial Black', color='black'), 
                    height=400,  # Increase the height of the plot
                    margin=dict(t=0, b=0, l=0, r=0),  # Adjust margins to control space around the plot
                    xaxis=dict(
                        tickfont=dict(size=14, family='Arial', color='black'),
                        title_font=dict(size=16, family='Arial Black', color='black')
                    ),
                    yaxis=dict(
                        tickfont=dict(size=14, family='Arial', color='black'),
                        title_font=dict(size=16, family='Arial Black', color='black')
                    ),
                    legend=dict(
                        x=0.75,  # Position the legend closer to the right edge (0 to 1 scale)
                        y=0.1,   # Position the legend near the bottom of the plot (0 to 1 scale)
                        bgcolor='rgba(255,255,255,0.5)',  # Transparent background for the legend
                        bordercolor='black',
                        borderwidth=1
                    )
                )
                # Render plotly chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
             
        else:
            st.warning("You need to convert the timestamp to minutes before fitting the model!")

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
