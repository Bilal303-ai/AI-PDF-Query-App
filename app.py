import streamlit as st
from backend import extract_text_and_generate_embeddings, store_embeddings, query_similar_text, chat_with_llm
import psycopg
import tempfile
import openai

st.set_page_config(page_title="PDF Sage 📜✨", layout="wide")
st.title("PDF Sage 📜✨")

# Sidebar for userinputs
st.sidebar.header("Configuration")

# Database inputs
st.sidebar.subheader("PostgreSQL Connection")
st.session_state["db_url"] =st.sidebar.text_input("Database URL", value="")

# OpenAI API key
st.sidebar.subheader("OpenAI API key (Optional)")
st.session_state["openai_api_key"] = st.sidebar.text_input("API key", type="password", value="")

# Function to test the database connection
def test_db_connection():
    try:
        conn = psycopg.connect(st.session_state["db_url"])
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
    pdf_id = store_embeddings(st.session_state["db_url"], uploaded_pdf.name, chunks, embeddings)
    st.write("Embeddings stored successfullly in PostgreSQL!")
    
    
# Integrate querying
user_query = st.text_input("Enter your query: ")

if user_query:
    retrieved_results = query_similar_text(st.session_state["db_url"], user_query, pdf_id)
    retrieved_text = [row['text'] for row in retrieved_results]
    
    st.write("### Most relevant chunks:")
    for idx, row in enumerate(retrieved_results):
        st.write(f"**{idx+1}.** {row['text']}")
        st.write(f"📊 **Distance:** {row['distance']:.4f}")
        
# Integrate llm
llm_api = st.session_state["openai_api_key"]
query_for_llm = st.text_input("Chat with a bot: ")
if st.button("Explain"):
    if query_for_llm:
        if llm_api:
            try:
                with st.spinner("Thinking..."):
                    llm_response = chat_with_llm(
                                query_for_llm,
                                retrieved_text,
                                llm_api,
                                )
                
                st.write("### Explanation: ")
                st.write(llm_response)
            except openai.AuthenticationError:
                st.error("Invalid API key!")
            except Exception:
                st.error("Oops, we can't connect! please try again")
    
    else:
        st.warning("Please enter a query!")
