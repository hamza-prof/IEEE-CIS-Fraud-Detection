# Fraud Detection ML Project by Hamza

This is a Fraud Detection project built using Streamlit to identify and detect fraudulent activities.

## Setup Instructions

Follow these steps to set up and run the project:

1. **Create a Python Virtual Environment**

   ```bash
   python -m venv ml-env
   ```

2. **Activate the Virtual Environment**

   - On Windows:
     ```bash
     ml-env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source ml-env/bin/activate
     ```

3. **Install Required Libraries**
   Use the `requirements.txt` file to install all necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Project**
   Start the Streamlit application with the following command:
   ```bash
   streamlit run prediction.py
   ```

## Project Overview

This project leverages machine learning to detect potential fraudulent activities based on transaction data. The user-friendly Streamlit interface makes it simple to input data and view predictions.

### Key Features:

- **Real-time Predictions**: Detect fraudulent transactions instantly.
- **Easy-to-Use Interface**: Built with Streamlit for a seamless user experience.
- **Customizable**: Modify the model or features to suit your dataset and requirements.

### Requirements:

Ensure the following are installed on your system:

- Python 3.x
- Streamlit
- Libraries listed in `requirements.txt`
