import streamlit as st
from backend import extract_text_and_generate_embeddings, store_embeddings, query_similar_text
import psycopg
from psycopg_pool import ConnectionPool
import tempfile

st.set_page_config(page_title="AI-Powered PDF Chat", layout="wide")
st.title("AI-Powered PDF Chat")

# Sidebar for userinputs
st.sidebar.header("Configuration")

# Database inputs
st.sidebar.subheader("PostgreSQL Connection")
st.session_state["db_name"] = st.sidebar.text_input("Database Name", value="")
st.session_state["db_user"] = st.sidebar.text_input("User", value="")
st.session_state["db_password"] = st.sidebar.text_input("Password", type="password", value="")
st.session_state["db_host"] = st.sidebar.text_input("Host", value="")

# OpenAI API key
st.sidebar.subheader("OpenAI API key (Optional)")
st.session_state["openai_api_key"] = st.sidebar.text_input("API key", type="password", value="")

db_info = {"dbname": st.session_state["db_name"], 
            "user": st.session_state["db_user"],
            "password": st.session_state["db_password"],
            "host": st.session_state["db_host"]}
# Function to test the database connection
def test_db_connection():
    try:
        conn = psycopg.connect(**db_info)
        conn.close()
        return True
    except Exception as e:
        return str(e)

if st.sidebar.button("Save Configuration"):
    result = test_db_connection()
    if result is True:
        st.sidebar.success("Succesfully connected to the database")
    else:
        st.sidebar.error(f"Failed to connect: {result}")
        

# Upload and process PDF
uploaded_pdf = st.file_uploader("Upload a PDF", type='pdf')

if uploaded_pdf:
    
    # Save the pdf temporarily 
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        temp_pdf.write(uploaded_pdf.read())
        pdf_path = temp_pdf.name
    
    temp_pdf.close()
    st.write("PDF uploaded successfully!")

    chunks, embeddings = extract_text_and_generate_embeddings(pdf_path)
    
    st.write(f"Extracted {len(chunks)} chunks from PDF!")
    st.write("Generated embeddings successfully!")
    
    # Store embeddings in DB
    store_embeddings(db_info, uploaded_pdf.name, chunks, embeddings)
    st.write("Embeddings stored successfullly in PostgreSQL!")