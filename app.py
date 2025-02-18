import streamlit as st
from dotenv import load_dotenv
import uuid
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from the uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into chunks for efficient processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# Create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, say: "Sorry, I didn't understand your question. Do you want to connect with a live agent?"
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user input, perform a similarity search, and give an answer
def user_input(user_question):
    response_text = ""

    if 'vector_store' not in st.session_state:
        st.error("Please upload and process PDF files first.")
        return ""

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        docs = new_db.similarity_search(user_question)

        if not docs:
            response_text = "Sorry, I didn't understand your question. Do you want to connect with a live agent?"
        else:
            chain = get_conversational_chain()
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            response_text = response["output_text"]

        # Add to conversation history
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
        
        st.session_state.conversation.append({
            "id": str(uuid.uuid4()),
            "user": user_question,
            "bot": response_text
        })

        return response_text

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return f"An error occurred: {e}"

# Create the chatbot UI
def main():
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    st.set_page_config(page_title="PDF ChatBot", page_icon=":books:")
    
    st.title("Ask Apport Agent 🤖")

    with st.sidebar:
        st.header("PDF Processing")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=['pdf'])
        
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    
                    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    st.session_state.vector_store = FAISS.load_local("faiss_index", st.session_state.embeddings, allow_dangerous_deserialization=True)
                    st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload PDF files first.")

        if st.button("Clear Conversation"):
            st.session_state.conversation = []

    st.header("Conversation")
    
    for chat in st.session_state.conversation:
        st.chat_message("user").write(chat['user'])
        st.chat_message("assistant").write(chat['bot'])

    # Chat input
    user_question = st.chat_input("Ask a question about your PDF...")
    
    if user_question:
        st.chat_message("user").write(user_question)
        
        with st.chat_message("assistant"):
            response = user_input(user_question)
            st.write(response)

if __name__ == "__main__":
    main()