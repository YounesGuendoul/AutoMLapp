from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import SKLearnVectorStore
from langchain import HuggingFaceHub
import os


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"



def qadocument(file_path,query):
# Load the PDF file from current working directory
    loader = PyPDFLoader(file_path)

    # Split the PDF into Pages
    pages = loader.load_and_split()



    # Define chunk size, overlap and separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64,
        separators=['\n\n', '\n', '(?=>\. )', ' ', '']
    )

    # Split the pages into texts as defined above
    texts = text_splitter.split_documents(pages)



    # Load embeddings from HuggingFace
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


    # Set the persisted vector store
    vector_db_path = "./document_vector_db.parquet"

    # Create the vector store
    vector_db = SKLearnVectorStore.from_documents(texts, embedding=embedding, persist_path=vector_db_path,
                                                serializer="parquet")




    llm=HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.1 ,"max_length":512})



    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
                                    return_source_documents=True,
                                    verbose=False,
    )

    # Send question as a query to qa chain
    result = qa({"query": query})
    result = result["result"]
    response_lines = result.split("\n")

# Get the second line (index 1) of the response
    result = response_lines[1]
    return result