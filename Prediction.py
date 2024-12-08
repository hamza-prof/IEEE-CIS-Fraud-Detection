import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import os
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


st.markdown("""
    <style>
    h1 {
        color: blue;
    }
    </style>
""", unsafe_allow_html=True)


st.title("Fraud Detection")





# Apply custom theme and CSS styling
st.markdown("""
    <style>
    .main {
            background: url('/mnt/data/digital-padlock-with-virtual-screen-on-dark-background-cyber-security-technology-for-fraud-prevention-and-privacy-data-network-protection-concept-vector.jpg') no-repeat center center fixed;
            background-size: cover;
        }   
     .block-container { padding: 3rem; }
    h1 { font-size: 2.5rem; color: #1A7FA2  ; }
    h2 { font-size: 1.75rem; color: #1A7FA2; }
    .stButton>button { background-color: #1A7FA2; color: white; }
    # .stSidebar { background-color: #1A7FA2; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""

    <style>
    .main {
        background: url('/mnt/data/digital-padlock-with-virtual-screen-on-dark-background-cyber-security-technology-for-fraud-prevention-and-privacy-data-network-protection-concept-vector.jpg') no-repeat center center fixed;
        background-size: cover;
    }
    .block-container {
        padding: 3rem;
        background-color: rgba(255, 255, 255, 0.8); /* Optional: Add transparency for content blocks */
    }
    h1 {
        font-size: 2.5rem;
        color: #1A7FA2;
    }
    h2 {
        font-size: 1.75rem;
        color: white;
    }
    .stButton>button {
        background-color: #1A7FA2;
        color: white;
    }
</style>
    """, unsafe_allow_html=True)


# Inject custom CSS to hide link icons for headers
st.markdown("""
    <style>
    .st-emotion-cache-gi0tri {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Page title
st.markdown("""
    <style>
    /* Change the slider's background color */
    div.stSlider > div[data-baseweb = "slider"] > div > div {
        background-color: #1A7FA2 ; /* Set slider background color */
    }
    div.stSlider > div[data-baseweb = "slider"] > div > div {
            background: linear-gradient(to right,  0%, blue 100%, rgba(173, 173, 173, 0.25) 100%, rgba(173, 173, 173, 0.25) 100%);
            } 

    /* Change the color of the slider thumb's value */
    .stSlider .stSlider > div > div .StyledThumbValue {
        color: #1A7FA2 ; /* Change the value text to white */
    }
            
    .st-emotion-cache-na9pmf {
            background-color: white;
    } 
    .st-emotion-cache-1qtecxd {
            color: white;
    }
            
    .st-emotion-cache-16txtl3{
        background-color: #1A7FA2;
        border-right: 3px solid #146374; /* A slightly darker shade for the border */


            }
    .st-emotion-cache-1inwz65{
             color: white;
            }

    .st-emotion-cache-12fmjuu{
        background-color: #1A7FA2;

            }

    .st-emotion-cache-152dyl6 {
             color: white;
            }

    .st-emotion-cache-ugcgyn{
        background-color: #1A7FA2;
        border-right: 3px solid #146374; /* A slightly darker shade for the border */
        

    }

    .st-emotion-cache-x9yr70:hover{
        color:red;
    }

    .st-emotion-cache-rr5e3k{
            background-color: #1A7FA2;

    }
            
    .st-emotion-cache-1vzeuhh{
            background-color:white;
            }
    .st-emotion-cache-1373cj4 {
            color:white
    }
            
    div.stSlider > div[data-baseweb = "slider"] > div > div{
            color:white;
            }
    .st-emotion-cache-10y5sf6{
            color:white;
            }
    
    .st-emotion-cache-l9bjmx p{
            color:white;
            }
    .st-emotion-cache-1puwf6r p{
            color:white;
            }
    .st-emotion-cache-1dj3ksd{
            background-color:white;
            }

    st-ci {
background-image: linear-gradient(to right, rgb(255, 75, 75) 0%, rgb(255, 75, 75) 59.8%, rgba(151, 166, 195, 0.25) 59.8%, rgba(151, 166, 195, 0.25) 100%);}

    .st-emotion-cache-6qob1r{
        background-color: #1A7FA2;
            }
    .st-emotion-cache-18ni7ap{
        background-color: #1A7FA2;
            }

    

    .st-ag st-ah st-ai st-aj st-ak st-al st-am{
            background-color:blue
            }

    /* Change the color of the slider's handle */
    .stSlider .stSlider > div > div {
        background-color: #1A7FA2 ; /* Customize the slider's handle color */
    }

    </style>
""", unsafe_allow_html=True)

# Sidebar header
st.sidebar.header('Input Credit Card Details')

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=['csv'])

# User input for single sample
def user_input():
    # Use sliders for numeric inputs
    TransactionID = st.sidebar.slider('TransactionID', -3663549, 3663549, 5663549)
    TransactionAmt = st.sidebar.slider('TransactionAmt', 0.0, 5000.0, 3500.0)
    card1 = st.sidebar.slider('card1', -5.0, 5.0, 0.0)
    C13 = st.sidebar.slider('C13', 0.0, 100.0, 115.0)
    TransactionDT_Days = st.sidebar.slider('TransactionDT_Days', -100, 100, 7)
    addr1 = st.sidebar.slider('addr1', 0, 100, 6)
    C1 = st.sidebar.slider('C1', -5.0, 5.0, 4.0)
    card2 = st.sidebar.slider('card2', 0, 100, 6)
    C14 = st.sidebar.slider('C14', 0.0, 100.0, 21.0)
    D15 = st.sidebar.slider('D15', 0.0, 100.0, 16.0)
    card5 = st.sidebar.slider('card5', 0, 100, 4)
    P_emaildomain = st.sidebar.slider('P_emaildomain', 0.0, 100.0, 419.0)
    C6 = st.sidebar.slider('C6', -5.0, 5.0, 16.0)
    D1 = st.sidebar.slider('D1', 0.0, 100.0, 4.0)
    D2 = st.sidebar.slider('D2', 0.0, 100.0, 419.0)
    C11 = st.sidebar.slider('C11', 0.0, 100.0, 5.0)
    Transaction_month = st.sidebar.slider('Transaction_month', 1, 12, 7)
    card3 = st.sidebar.slider('card3', 0, 100, 398)
    D4 = st.sidebar.slider('D4', 0.0, 100.0, 6.0)
    C2 = st.sidebar.slider('C2', 0.0, 100.0, 16.0)
    dist1 = st.sidebar.slider('dist1', 0.0, 100.0, 2.0)
    R_emaildomain = st.sidebar.slider('R_emaildomain', 0.0, 100.0, 418.0)
    card6 = st.sidebar.slider('card6', 0.0, 100.0, 6.0)
    D10 = st.sidebar.slider('D10', 0.0, 100.0, 0.0)
    C9 = st.sidebar.slider('C9', 0.0, 100.0, 472.0)
    M5 = st.sidebar.slider('M5', 0.0, 100.0, 37.79)
    id_20 = st.sidebar.slider('id_20', 0.0, 1000.0, 2526.0)
    D8 = st.sidebar.slider('D8', 0.0, 100.0, 47.0)
    C5 = st.sidebar.slider('C5', 0.0, 100.0, 42.0)
    C12 = st.sidebar.slider('C12', 0.0, 100.0, 1.0)
    M6 = st.sidebar.slider('M6', 0.0, 100.0, 37.79)
    D11 = st.sidebar.slider('D11', 0.0, 100.0, 3.0)
    id_01 = st.sidebar.slider('id_01', 0.0, 1000.0, 1.0)
    DeviceInfo = st.sidebar.slider('DeviceInfo', 0.0, 100.0, 0.0)
    id_31 = st.sidebar.slider('id_31', 0.0, 100.0, 47.0)
    id_30 = st.sidebar.slider('id_30', 0.0, 100.0, 42.0)
    V87 = st.sidebar.slider('V87', 0.0, 100.0, 1.0)
    V53 = st.sidebar.slider('V53', 0.0, 100.0, 0.0)
    id_02 = st.sidebar.slider('id_02', 0.0, 1000.0, 125800.5)
    V83 = st.sidebar.slider('V83', 0.0, 100.0, 0.0)
    D3 = st.sidebar.slider('D3', 0.0, 100.0, 0.0)
    ProductCD = st.sidebar.slider('ProductCD', 0.0, 100.0, 3.0)
    V313 = st.sidebar.slider('V313', 0.0, 1000.0, 0.0)
    V70 = st.sidebar.slider('V70', 0.0, 100.0, 47.95)
    V310 = st.sidebar.slider('V310', 0.0, 1000.0, 0.0)
    M4 = st.sidebar.slider('M4', 0.0, 100.0, 1.0)
    EarlyMorningFlag = st.sidebar.slider('EarlyMorningFlag', 0.0, 100.0, 0.0)
    id_05 = st.sidebar.slider('id_05', 0.0, 100.0, 0.0)
    card4 = st.sidebar.slider('card4', 0.0, 100.0, 0.0)
    V296 = st.sidebar.slider('V296', 0.0, 100.0, 0.0)

    data = {
        'TransactionID': TransactionID, 'TransactionAmt': TransactionAmt, 'card1': card1, 'C13': C13,
        'TransactionDT_Days': TransactionDT_Days, 'addr1': addr1, 'C1': C1, 'card2': card2,
        'C14': C14, 'D15': D15, 'card5': card5, 'P_emaildomain': P_emaildomain, 'C6': C6,
        'D1': D1, 'D2': D2, 'C11': C11, 'Transaction_month': Transaction_month, 'card3': card3,
        'D4': D4, 'C2': C2, 'dist1': dist1, 'R_emaildomain': R_emaildomain, 'card6': card6,
        'D10': D10, 'C9': C9, 'M5': M5, 'id_20': id_20, 'D8': D8, 'C5': C5, 'C12': C12,
        'M6': M6, 'D11': D11, 'id_01': id_01, 'DeviceInfo': DeviceInfo, 'id_31': id_31,
        'id_30': id_30, 'V87': V87, 'V53': V53, 'id_02': id_02, 'V83': V83, 'D3': D3,
        'ProductCD': ProductCD, 'V313': V313, 'V70': V70, 'V310': V310, 'M4': M4,
        'EarlyMorningFlag': EarlyMorningFlag, 'id_05': id_05, 'card4': card4, 'V296': V296
    }

    return pd.DataFrame(data, index=[0])

# Load input data
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    input_df = user_input()

# Show data
st.subheader('Credit Card Data')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded.')
    st.write(input_df)

# Load the trained model
# Define the path to the model
model_path = 'savedModels/lightgbm_model.pkl'

# Check if the model file exists
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            load_clf = joblib.load(model_file)
            print("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        load_clf = None
else:
    st.error(f"Model file not found. Please check the path to '{model_path}'.")
    load_clf = None

if load_clf is not None:
    # Make predictions
    predictions = load_clf.predict(input_df)

    # Example threshold for binary classification
    threshold = 0.3176
    y_pred_binary = (predictions > threshold).astype(int)

    # Add prediction to input data
    input_df['isFraud'] = y_pred_binary


# Check if 'isFraud' column is already in the dataset
if 'isFraud' in input_df.columns:
    # Separate fraudulent and genuine transactions
    fraudulent_transactions = input_df[input_df['isFraud'] == 1]
    genuine_transactions = input_df[input_df['isFraud'] == 0]


    
    # Display both tables side by side
    col1, col2 = st.columns(2)

    # Display Genuine Transactions in the first column
    with col1:
        st.subheader('Genuine Transactions')
        st.write(genuine_transactions[['TransactionID', 'isFraud']])

    # Display Fraudulent Transactions in the second column
    with col2:
        st.subheader('Fraudulent Transactions')
        st.write(fraudulent_transactions[['TransactionID', 'isFraud']])
    st.write(f"Fraudulent Transactions: {sum(input_df['isFraud'] == 1)}")
    st.write(f"Genuine Transactions: {sum(input_df['isFraud'] == 0)}")

    st.subheader("Bar Chart")
    fraud_counts = input_df['isFraud'].value_counts()

    # Dynamically set index names based on the number of categories
    fraud_counts.index = ['Genuine' if val == 0 else 'Fraudulent' for val in fraud_counts.index]

    # Plot using Plotly
    fig = px.bar(fraud_counts, x=fraud_counts.index, y=fraud_counts.values, 
                labels={'x': 'Transaction Type', 'y': 'Count'}, 
                title='Transaction Type Counts')
    st.plotly_chart(fig)


    # Box Plot
    st.subheader("Box Plot")
    selected_feature = st.selectbox('Select a feature for Box Plot:', input_df.select_dtypes(include=['float64', 'int64']).columns)
    if selected_feature:
        fig, ax = plt.subplots()
        sns.boxplot(data=input_df, x=selected_feature, ax=ax)
        ax.set_title(f'Box Plot of {selected_feature}')
        st.pyplot(fig)

    
    


    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    correlation_matrix = input_df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)


    # Scatter Plot
    st.subheader("Scatter Plot")
    x_feature = st.selectbox('Select X-axis feature:', input_df.columns)
    y_feature = st.selectbox('Select Y-axis feature:', input_df.columns)
    if x_feature and y_feature:
        fig = px.scatter(input_df, x=x_feature, y=y_feature, color='isFraud', title=f'Scatter Plot of {x_feature} vs {y_feature}')
        st.plotly_chart(fig)

    # Pie Chart
    st.subheader("Pie Chart")
    fig = px.pie(values=fraud_counts.values, names=fraud_counts.index, title='Transaction Type Proportions')
    st.plotly_chart(fig)

    # Feature Importance
    if hasattr(load_clf, 'feature_importances_'):
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': input_df.columns[:-1],
            'Importance': load_clf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', title='Feature Importance')
        st.plotly_chart(fig)
else:
    st.error("Prediction not available. Ensure the model is loaded and 'isFraud' column exists.")



