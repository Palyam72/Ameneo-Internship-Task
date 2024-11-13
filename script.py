import streamlit as st
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from ctransformers import AutoModelForCausalLM
from io import BytesIO

# Custom CSS for advanced UI styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #f0f4f8, #d9e2ec);
            font-family: Arial, sans-serif;
        }
        .chat-container {
            padding: 15px;
            background-color: #f7fafc;
            border-radius: 20px;
            border: 1px solid #e0e0e0;
            margin-bottom: 15px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
            font-size: 16px;
        }
        .user-bubble, .bot-bubble {
            display: flex;
            align-items: center;
            padding: 10px 15px;
            margin-bottom: 10px;
            font-size: 16px;
            border-radius: 20px;
        }
        .user-bubble {
            background-color: #0078D4;
            color: white;
            border: 2px solid #0078D4;
            border-radius: 20px 20px 0px 20px;
            justify-content: flex-start;
        }
        .bot-bubble {
            background-color: #f1f1f1;
            color: black;
            border: 2px solid #ffc107;
            border-radius: 20px 20px 20px 0px;
            justify-content: flex-start;
        }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: gray;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTextInput > div > div > input, .stTextArea > div > textarea {
            border-radius: 15px;
            padding: 10px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat interaction and vector database
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "vector_database" not in st.session_state:
    st.session_state["vector_database"] = None

# Path to embeddings and Llama model files
model_path = "D:/"
llama_model_path = model_path + "llama-2-7b-chat.Q2_K.gguf"
embeddings_model_path = model_path + "all-MiniLM-L6-v2"

# Load LLaMA 2 model locally for LLM insights with quality-enhancing parameters
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_path, model_type="llama", temperature=0.7, top_k=50
)
# Load SentenceTransformer locally for PDF vectorization
embeddings_model = SentenceTransformerEmbeddings(model_name=embeddings_model_path)

# Function to parse PDF files and convert them into document objects
def parse_pdf(file):
    try:
        reader = PdfReader(BytesIO(file.read()))
        documents = [
            Document(page_content=page.extract_text(), metadata={"page": i + 1})
            for i, page in enumerate(reader.pages) if page.extract_text()
        ]
        return documents
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
        return []

# Retrieve insights from PDF query with the top relevant result
def get_insights_from_query(query_text, top_k=3):  # Increase top_k for better context in comparison
    query_embedding = embeddings_model.embed_query(query_text)
    results = st.session_state["vector_database"].similarity_search_by_vector(query_embedding, k=top_k)
    insights = [result.page_content for result in results] if results else ["No relevant content found."]
    return "\n\n".join(insights) if insights else "No relevant response available."

# Sidebar for uploading PDFs
pdf_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Sidebar menu for options
with st.sidebar:
    option = option_menu("Choose an Action", ["Ask Anything", "Chat With PDFs", "Compare PDFs"], default_index=1)

# Process PDFs and create vector database if not initialized
if pdf_files and st.session_state["vector_database"] is None:
    with st.spinner('Processing your PDFs...'):
        documents = [doc for pdf_file in pdf_files for doc in parse_pdf(pdf_file)]
        if documents:
            st.session_state["vector_database"] = FAISS.from_documents(documents, embeddings_model)
            st.success("PDFs are successfully processed and ready for querying!")

# Chat-like interface display
# Chat-like interface display without bot symbol
def display_chat():
    for i, entry in enumerate(st.session_state["chat_history"]):
        user_query = entry.get("query", "")
        bot_response = entry.get("response", "No response available.")
        
        # User message styling
        st.markdown(
            f'<div class="user-bubble chat-container">'
            f'<img src="https://img.icons8.com/color/48/000000/user.png" class="avatar"/>'
            f'<b>User:</b> {user_query}</div>', unsafe_allow_html=True
        )
        
        # Bot response without icon, in a clean, simple format
        st.markdown(
            f'<div class="bot-bubble chat-container">'
            f'{bot_response}</div>', unsafe_allow_html=True
        )

# Header for the application
st.markdown("<h2 style='text-align: center; color: #333;'>📄 PDF Chat and Analysis Bot</h2>", unsafe_allow_html=True)

# Option: Chat With PDFs - Allows querying the content in uploaded PDFs
if option == "Chat With PDFs":
    st.title("Ask Questions About Your PDFs")
    chat_input = st.text_input("Enter your query")

    if chat_input:
        with st.spinner('Fetching insights...'):
            response = get_insights_from_query(chat_input)
            st.session_state["chat_history"].append({
                "query": chat_input,
                "response": response
            })
        display_chat()

# Option: Compare PDFs - Highlight differences and compare content
elif option == "Compare PDFs":
    st.title("Compare Your PDFs")
    compare_query = st.text_input("Enter a query to compare documents (e.g., revenue differences)")

    if compare_query:
        with st.spinner('Comparing documents...'):
            response = get_insights_from_query(compare_query)
            st.session_state["chat_history"].append({
                "query": compare_query,
                "response": response
            })
        display_chat()

# Option: Ask Anything - General LLM-based Chatbot
elif option == "Ask Anything":
    st.title("Ask Anything")
    essay_input = st.text_area("Enter a topic or question")

    if essay_input:
        with st.spinner('Generating response...'):
            response = llama_model(essay_input, max_new_tokens=200, temperature=0.7)
            st.session_state["chat_history"].append({
                "query": essay_input,
                "response": response
            })
        display_chat()

# Footer for the application
st.markdown("<div class='footer'>Made with ❤️ by Rohith</div>", unsafe_allow_html=True)
