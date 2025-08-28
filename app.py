import streamlit as st
import os
import traceback
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Load Environment Variables ---
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="IqbalGPT - AI Guide to Iqbal's Poetry",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Complete Dark-Themed Redesign ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* General Styling */
    body {
        font-family: 'Inter', sans-serif;
        color: #e2e8f0; /* Default light text color */
    }
    
    .stApp {
        background-color: #0f172a; /* Dark Slate background for the entire app */
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b; /* Slightly lighter dark slate */
        padding: 1.5rem;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #94a3b8; /* Lighter text for descriptions */
        font-size: 0.95rem;
    }
    
    /* Sidebar Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        border: 1px solid #334155;
        background-color: #334155;
        color: #e2e8f0;
        transition: all 0.2s ease-in-out;
        padding: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .stButton>button:hover {
        background-color: #475569;
        color: #ffffff;
        border-color: #475569;
    }

    /* Main Content Area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat Input */
    [data-testid="stChatInput"] {
        background-color: #1e293b;
        border-radius: 1rem;
        border: 1px solid #334155;
    }
    [data-testid="stTextInput"] textarea {
        color: #e2e8f0;
    }
    
    /* Chat Bubbles */
    [data-testid="stChatMessage"] {
        padding: 1.2rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        max-width: 85%;
        border: 1px solid #334155;
    }

    /* Assistant Chat Bubble */
    [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-assistant"]) {
        background-color: #1e293b;
        color: #e2e8f0;
        float: left;
    }

    /* User Chat Bubble */
    [data-testid="stChatMessage"]:has(span[data-testid="chat-avatar-user"]) {
        background-color: #2563eb; /* A vibrant blue */
        color: #ffffff;
        float: right;
    }
    
    /* Clear floats after messages */
    .stChatMessage {
        clear: both;
    }

    /* Header Styling */
    .title-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .title-container h1 {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    
    .title-container p {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }

    /* Welcome Message */
    .welcome-container {
        text-align: center;
        padding: 3rem;
        background-color: #1e293b;
        border-radius: 1rem;
        border: 1px solid #334155;
    }
    .welcome-container h2 {
        color: #ffffff;
    }
    .welcome-container p {
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar Information ---
with st.sidebar:
    st.title("✒️ IqbalGPT")
    st.markdown("---")
    st.header("About")
    st.markdown(
        "**IqbalGPT** is an AI chatbot designed to answer your questions about the poetry and philosophy of Allama Iqbal, a celebrated poet-philosopher of the East."
    )
    st.markdown(
        "It uses a custom knowledge base to provide accurate, context-aware answers."
    )
    st.markdown("---")
    st.subheader("Example Prompts")
    
    example_prompts = [
        "What is the concept of Khudi?",
        "Explain Iqbal's idea of the 'Shaheen' (Eagle)",
        "What is the message of the poem 'Shikwa' (The Complaint)?"
    ]
    
    if 'prompt_from_button' not in st.session_state:
        st.session_state.prompt_from_button = ""

    for prompt in example_prompts:
        if st.button(prompt):
            st.session_state.prompt_from_button = prompt

# --- Constants ---
VECTORSTORE_PATH = "faiss_index_iqbal"
DATA_PATH = "iqbal_knowledge_base.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = "llama3-8b-8192"

# --- Caching Functions for Performance ---
@st.cache_resource
def load_and_embed_data():
    loader = TextLoader(DATA_PATH, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

@st.cache_resource
def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(model_name=LLM_MODEL_NAME, temperature=0.7)
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer from the context, just say that you don't know.
    Provide a detailed and well-structured answer based only on the provided context.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    return qa_chain

# --- Main Application Logic ---
st.markdown("""
<div class="title-container">
    <h1>IqbalGPT</h1>
    <p>Your AI guide to the poetry and philosophy of Allama Iqbal</p>
</div>
""", unsafe_allow_html=True)


if not os.getenv("GROQ_API_KEY"):
    st.error("Groq API key not found. Please create a `.env` file with your Groq key.")
    st.stop()

try:
    if not os.path.exists(VECTORSTORE_PATH):
        with st.spinner("Setting up the knowledge base for the first time... This may take a moment."):
            load_and_embed_data()
    qa_chain = load_qa_chain()
except Exception as e:
    st.error(f"Failed to initialize the application. Error: {e}")
    tb = traceback.format_exc()
    st.error(f"**Traceback:**\n```\n{tb}\n```")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display a welcome message if the chat is empty
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-container">
        <h2>Welcome to IqbalGPT!</h2>
        <p>Ask me anything about Allama Iqbal's work, or select a prompt from the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "✒️"):
        st.markdown(message["content"])

# Handle user input (from chat box or sidebar buttons)
prompt = st.chat_input("Ask a question in English or Roman Urdu...")
if st.session_state.prompt_from_button:
    prompt = st.session_state.prompt_from_button
    st.session_state.prompt_from_button = ""

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✒️"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                response = result['result']
                st.markdown(response)
                with st.expander("View Sources"):
                    st.write("The answer was generated based on these excerpts:")
                    for doc in result["source_documents"]:
                        st.info(doc.page_content)
            except Exception as e:
                tb = traceback.format_exc()
                st.error(f"An error occurred while generating the response:  \n**Error:** {e}  \n\n**Traceback:**\n```\n{tb}\n```")
                response = f"Sorry, an error occurred: {e}"
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()
