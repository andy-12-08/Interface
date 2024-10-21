import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.integrate import odeint
from scipy.interpolate import interp1d

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
 
def generate_model_plot(t,yp,ym,r2,rmse):
    """Generate and return Plotly figure comparing model and actual output."""
    # Plot the actual output and fitted output using plotly
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

def foptd_optimization(t,yp,u):
    """Simulate First-Order Plus Dead-Time (FOPDT) Model"""
    # Initialize u0, yp0, and ns
    u0 = u[0]
    yp0 = yp[0]
    # specify number of steps
    ns = len(t)
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
    # bnds = ((0, None), (0, None), (0.0, None))
    # solution = minimize(objective,x0,bounds=bnds,method='L-BFGS-B')
    x = solution.x
    return x

def sim_foptd_model(t,yp,u,Km,taum,thetam):
    # Initialize u0, yp0, and ns
    u0 = u[0]
    yp0 = yp[0]
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
    def sim_model(Km,taum,thetam):
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
    return sim_model(Km,taum,thetam)

def add_vertical_divider():
    st.markdown("""
    <style>
        .vertical-line {
            border-left: 2px solid rgba(49, 51, 63, 0.2); 
            height: 100vh; 
            margin: 0 auto; 
        }
    </style>
    <div class="vertical-line"></div>
    """, unsafe_allow_html=True)
