import os
import streamlit as st

# updated import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Load FAISS vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Prompt Template
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load HuggingFace LLM
def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# UI Code
def main():
    st.set_page_config(page_title="Tablet Info Chatbot", page_icon="💊")
    st.title("💊 Tablet Info Summarizer")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask about a tablet or condition...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know — don't make anything up.
        Don't provide anything outside the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk, please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("⚠ Failed to load vector store.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_docs = response["source_documents"]

            # === LAYERED DISPLAY ===
            # Separate content by source
            user_friendly_summary = ""
            clinical_summary = ""

            for doc in source_docs:
                text = doc.page_content
                source_name = doc.metadata.get("source_name", "Unknown")

                if "gale" in source_name.lower():
                    user_friendly_summary += f"{text}\n\n"
                elif "dsm" in source_name.lower() or "icd" in source_name.lower():
                    clinical_summary += f"*Source: {source_name}*\n{text}\n\n"

            st.chat_message("assistant").markdown("Here's what I found:")

            # Show user-friendly section
            st.markdown("### 📘 Description")
            st.markdown(user_friendly_summary if user_friendly_summary else "Not available in current context.")

            # Show clinical reference section
            with st.expander("🔍 Clinical Reference"):
                st.markdown(clinical_summary if clinical_summary else "No clinical reference found in this response.")

            # Save to history
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if _name_ == "_main_":
    main()