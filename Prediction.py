import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import os

st.markdown("""
    <style>
    h1 {
        color: blue;
    }
    </style>
""", unsafe_allow_html=True)


st.title("Fraud Detection")


# # Check if the image exists
# image_path = os.path.join("assets", "image.jpg")
# if os.path.exists(image_path):
#     st.image(image_path, use_column_width=True)
# else:
#     st.write("Image not found!")


# Apply custom background image using CSS
# background_image_css = f"""
# <style>
# body {{
#     background-image: url("assets/image.jpg");
#     background-size: 100%;
#     height:100vh;
#     background-position: center;
#     background-repeat: no-repeat;
#     background-attachment: fixed;
# }}


# </style>
# """
# st.markdown(background_image_css, unsafe_allow_html=True)



# Apply custom theme and CSS styling
# st.markdown("""
#     <style>
#     .main { background-color: #1A7FA2; }
#     .block-container { padding: 3rem; }
#     h1 { font-size: 2.5rem; color: #1A7FA2  ; }
#     h2 { font-size: 1.75rem; color: #1A7FA2; }
#     .stButton>button { background-color: #1A7FA2; color: white; }
#     # .stSidebar { background-color: #1A7FA2; }
#     </style>
#     """, unsafe_allow_html=True)

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
    .st-emotion-cache-1inwz65{
             color: white;
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
    V1 = st.sidebar.slider('V1', -5.0, 1.5, 5.0)
    V2 = st.sidebar.slider('V2', -5.0, 1.5, 5.0)
    V3 = st.sidebar.slider('V3', -5.0, 1.5, 5.0)
    V4 = st.sidebar.slider('V4', -5.0, 1.5, 5.0)
    V5 = st.sidebar.slider('V5', -5.0, 1.5, 5.0)
    V6 = st.sidebar.slider('V6', -5.0, 1.5, 5.0)
    V7 = st.sidebar.slider('V7', -5.0, 1.5, 5.0)
    V8 = st.sidebar.slider('V8', -5.0, 1.5, 5.0)
    V9 = st.sidebar.slider('V9', -5.0, 1.5, 5.0)
    V10 = st.sidebar.slider('V10', -5.0, 1.5, 5.0)
    V11 = st.sidebar.slider('V11', -5.0, 1.5, 5.0)
    V12 = st.sidebar.slider('V12', -5.0, 1.5, 5.0)
    V13 = st.sidebar.slider('V13', -5.0, 1.5, 5.0)
    V14 = st.sidebar.slider('V14', -5.0, 1.5, 5.0)
    V15 = st.sidebar.slider('V15', -5.0, 1.5, 5.0)
    V16 = st.sidebar.slider('V16', -5.0, 1.5, 5.0)
    V17 = st.sidebar.slider('V17', -5.0, 1.5, 5.0)
    V18 = st.sidebar.slider('V18', -5.0, 1.5, 5.0)
    V19 = st.sidebar.slider('V19', -5.0, 1.5, 5.0)
    V20 = st.sidebar.slider('V20', -5.0, 1.5, 5.0)
    V21 = st.sidebar.slider('V21', -5.0, 1.5, 5.0)
    V22 = st.sidebar.slider('V22', -5.0, 1.5, 5.0)
    V23 = st.sidebar.slider('V23', -5.0, 1.5, 5.0)
    V24 = st.sidebar.slider('V24', -5.0, 1.5, 5.0)
    V25 = st.sidebar.slider('V25', -5.0, 1.5, 5.0)
    V26 = st.sidebar.slider('V26', -5.0, 1.5, 5.0)
    V27 = st.sidebar.slider('V27', -5.0, 1.5, 5.0)
    V28 = st.sidebar.slider('V28', -5.0, 1.5, 5.0)
    Amount = st.sidebar.number_input('Amount')

    data = {'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6,
            'V7': V7, 'V8': V8, 'V9': V9, 'V10': V10, 'V11': V11, 'V12': V12,
            'V13': V13, 'V14': V14, 'V15': V15, 'V16': V16, 'V17': V17, 'V18': V18,
            'V19': V19, 'V20': V20, 'V21': V21, 'V22': V22, 'V23': V23, 'V24': V24,
            'V25': V25, 'V26': V26, 'V27': V27, 'V28': V28, 'Amount': Amount}
    
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
load_clf = joblib.load(open('savedModels/model.joblib', 'rb'))

# Make predictions
predictions = load_clf.predict(input_df)
prediction_probabilities = load_clf.predict_proba(input_df)

# # Add labels
# input_df['Prediction'] = predictions
# input_df['Prediction Label'] = input_df['Prediction'].apply(lambda x: 'Genuine Transaction' if x == 0 else 'Fraudulent Transaction')
# input_df['Fraud Probability'] = prediction_probabilities[:, 1]

# # Display results
# st.subheader('Prediction Results')
# st.write(input_df[['Prediction Label', 'Fraud Probability']])

# Add labels and probabilities
input_df['Prediction'] = predictions
input_df['Prediction Label'] = input_df['Prediction'].apply(lambda x: 'Genuine Transaction' if x == 0 else 'Fraudulent Transaction')
input_df['Fraud Probability'] = prediction_probabilities[:, 1]

# Separate genuine and fraudulent transactions
genuine_transactions = input_df[input_df['Prediction'] == 0]
fraudulent_transactions = input_df[input_df['Prediction'] == 1]

# Create columns for horizontal display
col1, col2 = st.columns(2)

# Display Genuine Transactions in the first column
with col1:
    st.subheader('Genuine Transactions')
    st.write(genuine_transactions[['Prediction Label', 'Fraud Probability']])

# Display Fraudulent Transactions in the second column
with col2:
    st.subheader('Fraudulent Transactions')
    st.write(fraudulent_transactions[['Prediction Label', 'Fraud Probability']])

# Visualization Section

# Bar chart for prediction counts
st.write("### Fraudulent vs Genuine Transactions")
prediction_counts = input_df['Prediction Label'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=prediction_counts.index, y=prediction_counts.values, ax=ax, palette="coolwarm")
ax.set_title("Transaction Count")
ax.set_ylabel("Count")
ax.set_xlabel("Transaction Type")
st.pyplot(fig)

# Distribution of fraud probabilities
st.write("### Fraud Probability Distribution")
fig, ax = plt.subplots()
sns.histplot(input_df['Fraud Probability'], bins=20, kde=True, color="orange", ax=ax)
ax.set_title("Fraud Probability Distribution")
ax.set_xlabel("Fraud Probability")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Altair chart for Fraud Probability vs Amount
st.write("### Fraud Probability vs Transaction Amount")
chart = alt.Chart(input_df).mark_circle(size=60).encode(
    x='Amount',
    y='Fraud Probability',
    color='Prediction Label',
    tooltip=['Amount', 'Fraud Probability', 'Prediction Label']
).interactive()
st.altair_chart(chart, use_container_width=True)
