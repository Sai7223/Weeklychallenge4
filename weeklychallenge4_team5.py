# -*- coding: utf-8 -*-
"""Weeklychallenge4_Team5.ipynb


"""

import os
import zipfile
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import requests
from io import BytesIO

# Function to download and unzip the dataset from GitHub
def download_and_extract_zip(zip_url, extract_to='data/'):
    # Download the zip file from GitHub
    response = requests.get(zip_url)
    
    # Check if the response is valid
    if response.status_code == 200:
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        st.error("Failed to download the zip file")
        return None

# URL of the zip file in GitHub repository
zip_url = 'https://github.com/Sai7223/weeklychallenge4/raw/main/Police_Bulk_Data_2014_20241027.zip'

# Download and extract the zip file containing the dataset
download_and_extract_zip(zip_url)

# Path to the dataset
dataset_path = 'data/Police_Bulk_Data_2014_20241027.csv'

# Check if the dataset exists
if os.path.exists(dataset_path):
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(dataset_path)

    # Ensure 'offensedate' is in datetime format
    df['offensedate'] = pd.to_datetime(df['offensedate'], errors='coerce')

    # Display a preview of the data in Streamlit
    st.write("Dataset Preview:", df.head())
else:
    st.error(f"Dataset not found at {dataset_path}")

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
    
    # Ensure numeric columns exist in filtered data before calculating correlation matrix.
    if all(col in data.columns for col in numeric_cols):
        corr_matrix = data[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hoverongaps=False,
            colorscale='RdBu'
        ))
        fig.update_layout(
            title='Correlation Heatmap of Numerical Features',
            height=400
        )
        return fig
    
    return go.Figure()  # Return an empty figure if numeric columns are missing

st.plotly_chart(create_correlation_heatmap(filtered_df), use_container_width=True)

