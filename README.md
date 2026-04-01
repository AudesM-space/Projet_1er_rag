# Projet_1er_rag
# Developpement du RAG
***Le fichier du développement du rag est soumis dans la liste des fichiers dans le dépot***
# Interface streamlit 
## ETAPE 1 : Developpement de l'interface streamlit 
```python
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Charger les variables d'environnement
load_dotenv(override=True)

# Prompt template
prompt_template = """
Answer the following question based only on the provided context:
<context>
    {context}
</context>
<question>
    {input}
</question>
"""

# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Configuration de la page
st.set_page_config(page_title="RAG Assistant", layout="wide")

# CSS custom pour un style chic
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }
    h1, h2, h3 {
        color: #1A237E;
        font-family: 'Segoe UI', sans-serif;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1A237E, #37474F);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Tabs pour navigation
tab1, tab2, tab3 = st.tabs([" Loader", " Chatbot", "Settings"])

# --- TAB 1 : Loader ---
with tab1:
    st.header("Data Loader")
    pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)

    if st.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Processing your documents..."):
                content = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        content += page.extract_text()

                splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=512, chunk_overlap=16
                )
                chunks = splitter.split_text(content)

                with st.expander(" Voir les chunks extraits"):
                    st.write(chunks)

                embedding_model = OpenAIEmbeddings()
                vector_store = Chroma.from_texts(
                    chunks,
                    embedding_model,
                    collection_name="data_collection",
                )
                retriever = vector_store.as_retriever(kwargs={"k": 5})
                st.session_state.retriever = retriever

                st.success(" Documents traités avec succès !")
        else:
            st.warning("Veuillez uploader au moins un PDF.")

# --- TAB 2 : Chatbot ---
with tab2:
    st.header("Chatbot - Retrieval Augmented Generation")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.chat_input("Pose ta question...")
    if user_question and "retriever" in st.session_state:
        st.chat_message("user").write(user_question)

        context_docs = st.session_state.retriever.invoke(user_question)
        context_text = ". ".join([d.page_content for d in context_docs])
        prompt = prompt_template.format(context=context_text, input=user_question)

        resp = llm.invoke(prompt)
        st.chat_message("assistant").write(resp.content)

        st.session_state.chat_history.append((user_question, resp.content))
    elif user_question:
        st.warning("Merci de charger des documents avant de poser une question.")

# --- TAB 3 : Settings ---
with tab3:
    st.header("Settings")
    st.write("Paramètres généraux de l'application")
    theme = st.selectbox("Choisir un thème :", ["Light", "Dark", "Tech Blue"])
    st.write(f"Thème sélectionné : {theme}")

```

```bash
streamlit run rag.py
```
## ETAPE 1 : Chargement du document 
<img width="1867" height="475" alt="image" src="https://github.com/user-attachments/assets/fcf7bf40-06e4-4f8d-9371-683cdc4dcb06" />
## ETAPE 2 : Conversation avec le chatbot 
<img width="1877" height="469" alt="image" src="https://github.com/user-attachments/assets/9e048998-0a94-4300-99c2-9e5c7ca4497d" />
