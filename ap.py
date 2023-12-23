import os
import base64
from io import BytesIO
import re
import streamlit as st
import pandas as pd
import ydata_profiling
import plotly.express as px
from pycaret.regression import setup, pull
from streamlit_pandas_profiling import st_profile_report
from pdfminer.high_level import extract_text
from transformers import pipeline
from help import qadocument
from happytransformer import HappyTextToText, TTSettings

def pdf_to_text(pdf_file):
    text = extract_text(pdf_file)
    return text

def save_text_to_file(text, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(text)

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text_content = file.read()
    return text_content

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
    st.image("https://students.ma/wp-content/uploads/2019/02/Logo_inpt.png")
    st.markdown("<h1 class='title'>~~~~ Auto INPT ML ~~~~</h1>", unsafe_allow_html=True)
    st.sidebar.header("Navigation : ") 
    choice = st.sidebar.radio("", ["Upload Dataset", "Data Analysis", "Data Visualization", "Machine Learning Models","Download the Model","Natural Language Processing"])
    st.sidebar.info("This project application helps you to automate all the process of Machine Learning and NLP tasks aiming to reduce time and increase productivity. This project is made by Younes GUENDOUL. ")

if choice == "Upload Dataset":
    st.markdown("<h1 class='title'>Upload Your Dataset</h1>", unsafe_allow_html=True)
    file = st.file_uploader("Upload Your Dataset  (CSV file)")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.success("File uploaded successfully!")
        st.dataframe(df)

if choice == "Data Analysis":
    st.markdown("<h1 class='title'>Exploratory Data Analysis & Data Wrangling</h1>", unsafe_allow_html=True)
    profile_df = df.profile_report()
    st_profile_report(profile_df)


if choice == "Data Visualization":
    st.markdown("<h1 class='title'>Interactive DashBoards</h1>", unsafe_allow_html=True)
    st.subheader("Choose the Axis: ")
    x = st.selectbox('x', df.columns)
    y = st.selectbox('y', df.columns)
    color = st.selectbox('color', df.columns)
    
    if df[x].dtype != 'object' and df[y].dtype != 'object':
        st.subheader('Data Visualization')
        fig = px.scatter(df, x=x, y=y, color=color)
        st.plotly_chart(fig)
    else:
        st.warning("Please select numeric columns for x, y")    
    
    
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
                
        

#************************************************************************************************************
if choice == "Download the Model":
    st.markdown("<h1 class='title'>Download the best Model</h1>", unsafe_allow_html=True)
    if os.path.exists('best_model.pkl'):
        st.download_button('Download Model', 'best_model.pkl', file_name="best_model.pkl")
    else:
        st.warning("The model file 'best_model.pkl' does not exist.")


#************************************************************************************************************* 

if choice == "Natural Language Processing":
    st.markdown("<h1 class='title'>Natural Language Processing</h1>", unsafe_allow_html=True)
    problem_type = st.selectbox('Choose the Problem Type', ['Document Q&A and Summarazation', 'Sentiment Analysis',"Text Classification", 'Spell Check'])
    if problem_type == 'Document Q&A and Summarazation':
        file = st.file_uploader("Select a PDF file to Upload", type=["pdf"])
        
        
        if file is not None:
            pdf_contents = file.read()
            base64_pdf = base64.b64encode(pdf_contents).decode('utf-8')
            pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="600" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)
            with open("uploaded_file.pdf", "wb") as f:
                f.write(file.getvalue())
            st.success("File saved successfully.")

        user_input = st.text_area("Enter Your Question About The Document Here:", value="", height=100)
        if st.button('answer'):
            st.write(f"Chatbot: {qadocument('uploaded_file.pdf',user_input)}")

        if st.button('summarize'):
            st.write(f"Chatbot: {qadocument('uploaded_file.pdf','can you provide a summary of the document ?')}")
     
    elif problem_type == 'Sentiment Analysis':
        user_input = st.text_area("Enter Your Sentence Here to Predict the Emotion:", value="", height=100)
        if st.button('Predict Sentiment'):
            emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
            sentiment_label = emotion(user_input)[0]['label']
            sentiment_score = emotion(user_input)[0]['score']
            st.write(f"The Sentiment is {sentiment_label} with a Score of {sentiment_score:.2f}")


    elif problem_type == 'Text Classification':
        user_input = st.text_input("Enter Your Sentence To Classify Here:")
        words_for_class = st.text_input("Enter words (separated by commas) for Classification :")
        words = words_for_class.split(',')
        candidate_labels = [word.strip() for word in words]
        if st.button('Classify Sentence'):
            classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
            x = classifier(user_input, candidate_labels)
            st.write(f"The sentence is classified as {x['labels'][0]} with a Score of {x['scores'][0]:.2f}")
        



    elif problem_type == 'Spell Check':
        #st.markdown("<h1 class='title'> NLP for Spell Check </h1>", unsafe_allow_html=True)
        user_input = st.text_area("Enter Your Sentence To Correct Here:", value="", height=100)
        if st.button('Correct'):
            happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
            args = TTSettings(num_beams=5, min_length=1)
            result = happy_tt.generate_text(user_input, args=args)
            st.write(f"Correction : {result.text}") 

#************************************************************************************************************        



#streamlit run ap.py
#https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png
#https://img.freepik.com/vetores-gratis/coroa-de-ouro-realista-3d-ilustracao-vetorial_97886-286.jpg?size=338&ext=jpg
#https://static.tildacdn.com/tild3263-6531-4764-b532-343666323531/brain1.png

