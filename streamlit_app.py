import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')

st.info('This app builds a machine learning model!')

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Data preview with {df.shape[1]} columns and {df.shape[0]} rows")
    
    with st.expander('Data'):
        st.write('**Raw data**')
        st.write(df)
        
        # Separate features and target if possible
        st.write('**Feature and Target Selection**')
        all_columns = df.columns.tolist()
        target_column = st.selectbox('Select the target column (if any)', ['None'] + all_columns)
        
        if target_column != 'None':
            X_raw = df.drop(target_column, axis=1)
            y_raw = df[target_column]
        else:
            X_raw = df
            y_raw = pd.Series([None] * df.shape[0], name='target')

        st.write('**Features (X)**')
        st.write(X_raw)

        if target_column != 'None':
            st.write('**Target (y)**')
            st.write(y_raw)

    with st.expander('Data visualization'):
        st.write('**Scatter Plot**')
        numerical_cols = X_raw.select_dtypes(include=np.number).columns.tolist()
        if len(numerical_cols) >= 2:
            x_axis = st.selectbox('Select X-axis for scatter plot', numerical_cols)
            y_axis = st.selectbox('Select Y-axis for scatter plot', numerical_cols)
            if x_axis != y_axis:
                st.scatter_chart(data=df, x=x_axis, y=y_axis)
            else:
                st.warning("X-axis and Y-axis must be different for the scatter plot.")
        else:
            st.warning("Not enough numerical columns for scatter plot.")

    # Input features from the sidebar
    with st.sidebar:
        st.header('Input features')
        input_data = {}
        for col in X_raw.columns:
            unique_values = df[col].unique()
            if pd.api.types.is_numeric_dtype(df[col]):
                col_min = float(df[col].min())
                col_max = float(df[col].max())
                col_mean = float(df[col].mean())
                input_data[col] = st.slider(f'{col}', col_min, col_max, col_mean)
            else:
                input_data[col] = st.selectbox(col, unique_values)
        
        # Create a DataFrame for the input features
        input_df = pd.DataFrame([input_data])
        input_data_with_df = pd.concat([input_df, X_raw], axis=0)

    with st.expander('Input features'):
        st.write('**Input data**')
        st.write(input_df)
        st.write('**Combined data**')
        st.write(input_data_with_df)

    # Data preparation
    # Encode categorical features
    df_encoded = pd.get_dummies(input_data_with_df)

    X = df_encoded[1:]
    input_row = df_encoded[:1]

    if target_column != 'None':
        # Encode y (target variable) if it's not None
        if pd.api.types.is_numeric_dtype(y_raw):
            y = y_raw
        else:
            target_mapper = {val: idx for idx, val in enumerate(y_raw.unique())}
            y = y_raw.map(target_mapper)
    else:
        y = pd.Series([None] * df.shape[0])

    with st.expander('Data preparation'):
        st.write('**Encoded X (input data)**')
        st.write(input_row)
        if target_column != 'None':
            st.write('**Encoded y**')
            st.write(y)

    # Model training and inference
    if target_column != 'None' and y.notnull().all():
        clf = RandomForestClassifier()
        clf.fit(X, y)

        prediction = clf.predict(input_row)
        prediction_proba = clf.predict_proba(input_row)

        df_prediction_proba = pd.DataFrame(prediction_proba, columns=target_mapper.keys() if not pd.api.types.is_numeric_dtype(y_raw) else y_raw.unique())

        # Display predicted species
        st.subheader('Predicted Target')
        st.dataframe(df_prediction_proba, hide_index=True)
        if not pd.api.types.is_numeric_dtype(y_raw):
            predicted_label = list(target_mapper.keys())[list(target_mapper.values()).index(prediction[0])]
        else:
            predicted_label = prediction[0]
        st.success(f"Predicted: {predicted_label}")
    else:
        st.warning("No target column selected or target column is not properly defined.")

else:
    st.write("Please upload a CSV file to proceed.")
