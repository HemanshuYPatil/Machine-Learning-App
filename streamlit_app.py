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
    
    with st.expander('Data'):
        st.write('**Raw data**')
        st.write(df)

        st.write('**X**')
        X_raw = df.drop('species', axis=1)
        st.write(X_raw)

        st.write('**y**')
        y_raw = df['species']
        st.write(y_raw)

    with st.expander('Data visualization'):
        st.write('**Scatter Plot**')
        if 'bill_length_mm' in df.columns and 'body_mass_g' in df.columns and 'species' in df.columns:
            st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
        else:
            st.warning("The required columns for the scatter plot are not present in the uploaded file.")
    
    # Input features from the sidebar
    with st.sidebar:
        st.header('Input features')
        island = st.selectbox('Island', df['island'].unique())
        bill_length_mm = st.slider('Bill length (mm)', float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()), float(df['bill_length_mm'].mean()))
        bill_depth_mm = st.slider('Bill depth (mm)', float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()), float(df['bill_depth_mm'].mean()))
        flipper_length_mm = st.slider('Flipper length (mm)', float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()), float(df['flipper_length_mm'].mean()))
        body_mass_g = st.slider('Body mass (g)', float(df['body_mass_g'].min()), float(df['body_mass_g'].max()), float(df['body_mass_g'].mean()))
        gender = st.selectbox('Gender', df['sex'].unique())
        
        # Create a DataFrame for the input features
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': gender}
        input_df = pd.DataFrame(data, index=[0])
        input_penguins = pd.concat([input_df, X_raw], axis=0)

    with st.expander('Input features'):
        st.write('**Input penguin**')
        st.write(input_df)
        st.write('**Combined penguins data**')
        st.write(input_penguins)

    # Data preparation
    # Encode X
    encode = ['island', 'sex']
    df_penguins = pd.get_dummies(input_penguins, prefix=encode)

    X = df_penguins[1:]
    input_row = df_penguins[:1]

    # Encode y
    target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
    def target_encode(val):
        return target_mapper[val]

    y = y_raw.apply(target_encode)

    with st.expander('Data preparation'):
        st.write('**Encoded X (input penguin)**')
        st.write(input_row)
        st.write('**Encoded y**')
        st.write(y)

    # Model training and inference
    clf = RandomForestClassifier()
    clf.fit(X, y)

    prediction = clf.predict(input_row)
    prediction_proba = clf.predict_proba(input_row)

    df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

    # Display predicted species
    st.subheader('Predicted Species')
    st.dataframe(df_prediction_proba, hide_index=True)
    penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
    st.success(str(penguins_species[prediction][0]))

else:
    st.write("Please upload a CSV file to proceed.")
