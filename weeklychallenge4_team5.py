# -*- coding: utf-8 -*-
"""Weeklychallenge4_Team5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YJyLv1jUFwccsPd4odfv_YnuneW48CEU
"""



from google.colab import drive
drive.mount('/content/drive')

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

# Define the function to load the data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv('/content/drive/MyDrive/Police_Bulk_Data_2014_20241027.csv')
    # Ensure 'offensedate' is in datetime format
    df['offensedate'] = pd.to_datetime(df['offensedate'], errors='coerce')
    return df

# Load the data using the function
file_path = 'Police_Bulk_Data_2014_20241027.csv'  # Update this path as needed
df = load_data(file_path)

# Display a preview of the data in Streamlit
st.write("Dataset Preview:", df.head())

# Set page config
st.set_page_config(page_title="Police Offenses Dashboard", layout="wide")

# Title
st.title("Police Offenses Dashboard")

# Filters
col1, col2 = st.columns(2)

# Date range filter
with col1:
    start_date, end_date = st.date_input(
        "Select Date Range",
        [df['offensedate'].min().date(), df['offensedate'].max().date()],
        key='date_range'
    )

# Offense type filter
with col2:
    offense_type = st.multiselect(
        "Select Offense Type",
        options=df['offensedescription'].unique(),
        key='offense_type'
    )

# Distribution variable filter (State or Race)
distribution_variable = st.selectbox(
    "Select Variable for Distribution Plot",
    options=['offensestate', 'offenserace'],
    key='distribution_variable'
)

# Filter the data based on the user's input
filtered_df = df[(df['offensedate'].dt.date >= start_date) & (df['offensedate'].dt.date <= end_date)]

# If offense types are selected, filter the data further
if offense_type:
    filtered_df = filtered_df[filtered_df['offensedescription'].isin(offense_type)]

# Function to create trend chart
def create_trend_chart(data):
    fig = px.line(data, x='offensedate', title='Trend of Offenses Over Time')
    return fig

# Function to create gender and race count plot
def create_gender_race_countplot(data):
    gender_race_counts = data.groupby(['offensegender', 'offenserace']).size().reset_index(name='count')
    fig = px.bar(
        gender_race_counts,
        x='offensegender',
        y='count',
        color='offenserace',
        title='Offenses by Gender and Race',
        labels={'offensegender': 'Gender', 'count': 'Count', 'offenserace': 'Race'},
        template='plotly',
        barmode='group'
    )
    return fig

# Display charts
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(create_trend_chart(filtered_df), use_container_width=True)
    st.plotly_chart(create_gender_race_countplot(filtered_df), use_container_width=True)

with col2:
    # Additional visualizations can be added here
    st.write("Additional visualizations can be implemented.")

# Correlation heatmap function
def create_correlation_heatmap(data):
    numeric_cols = ['offensereportingarea', 'offensepropertyattackcode', 'offensezip']
    corr_matrix = data[numeric_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu'
    ))
    fig.update_layout(
        title='Correlation Heatmap of Numerical Features',
        height=400
    )
    return fig

st.plotly_chart(create_correlation_heatmap(filtered_df), use_container_width=True)
