import os
import base64
from io import BytesIO
import streamlit as st
import pandas as pd
import ydata_profiling
import plotly.express as px
from pycaret.regression import setup, pull
from streamlit_pandas_profiling import st_profile_report
from happytransformer import HappyTextToText, TTSettings






st.markdown("""
    <style>
    .title {
        text-align: center;                 
    }
    </style>
""", unsafe_allow_html=True)


if os.path.exists('dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)
else:
    df = None

with st.sidebar:
    st.image("3D-SMART-FACTORY1-removebg-preview.png")
    st.markdown("<h1 class='title'>~~~~ AutoYounesML ~~~~</h1>", unsafe_allow_html=True)
    st.sidebar.header("Navigation : ")
    choice = st.sidebar.radio("", ["Upload Dataset", "Data Analysis", "Machine Learning Models", "Visualization of Clusters","NLP","Download the Model"])
    st.sidebar.info("This project application helps you to explore your data and give you the best ML models automatically anf finally download it to use it after.")

if choice == "Upload Dataset":
    st.markdown("<h1 class='title'>Upload Your Dataset</h1>", unsafe_allow_html=True)
    file = st.file_uploader("Upload Your Dataset  (CSV file)")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.success("File uploaded successfully!")
        st.dataframe(df)

if choice == "Data Analysis":
    st.markdown("<h1 class='title'>EDA, Data Visualization & Data Wrangling</h1>", unsafe_allow_html=True)
    profile_df = df.profile_report()
    st_profile_report(profile_df)
    
    
if choice == "Machine Learning Models":
    st.markdown("<h1 class='title'>Model Configuration</h1>", unsafe_allow_html=True)
    st.subheader("Problem Type")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    problem_type = st.selectbox('Choose the Problem Type', ['Regression', 'Classification', 'Clustering'])
    if st.button('Run Modelling'):
        if problem_type == 'Regression':
            from pycaret.regression import setup, compare_models,pull, save_model
            setup(df, target=chosen_target, verbose=False)
            st.success('Setup Complete')
            st.subheader('Model Comparison')
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
        elif problem_type == 'Classification':
            from pycaret.classification import setup, compare_models,pull, save_model
            setup(df, target=chosen_target, verbose=False)
            st.success('Setup Complete')
            st.subheader('Model Comparison')
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
        else:
            from pycaret.clustering import setup,pull, save_model,create_model
            setup(df, verbose=False)
            st.success('Setup Complete')
            st.subheader('Model Comparison')
            setup_df = pull()
            st.dataframe(setup_df) 
            best_model = create_model( 'dbscan')
            compare_df = pull()
            st.dataframe(compare_df)
                
        save_model(best_model, 'best_model')


if choice == "Visualization of Clusters":
    st.markdown("<h1 class='title'>Visualization Of Clusters</h1>", unsafe_allow_html=True)
    st.subheader("Problem Type")
    x = st.selectbox('x', df.columns)
    y = st.selectbox('y', df.columns)
    color = st.selectbox('color', df.columns)
    
    if df[x].dtype != 'object' and df[y].dtype != 'object':
        st.subheader('Cluster Visualization')
        fig = px.scatter(df, x=x, y=y, color=color)
        st.plotly_chart(fig)
    else:
        st.warning("Please select numeric columns for x, y")

#*************************************************************************************************************
if choice == "NLP":
    st.markdown("<h1 class='title'> NLP for Spell Check </h1>", unsafe_allow_html=True)
    user_input = st.text_area("Enter Your Sentence To Correct Here:", value="", height=100)
    if st.button('Correct'):
        happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
        args = TTSettings(num_beams=5, min_length=1)
        result = happy_tt.generate_text(user_input, args=args)
        st.write(f"Correction : {result.text}") 


#*************************************************************************************************************************************

if choice == "Download the Model":
    st.markdown("<h1 class='title'>Download the best Model</h1>", unsafe_allow_html=True)
    if os.path.exists('best_model.pkl'):
        st.download_button('Download Model', 'best_model.pkl', file_name="best_model.pkl")
    else:
        st.warning("The model file 'best_model.pkl' does not exist.")



#streamlit run ap.py
#https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png
#https://img.freepik.com/vetores-gratis/coroa-de-ouro-realista-3d-ilustracao-vetorial_97886-286.jpg?size=338&ext=jpg
#https://static.tildacdn.com/tild3263-6531-4764-b532-343666323531/brain1.png
#3D-SMART-FACTORY1-removebg-preview.png

