import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
@st.cache_resource(show_spinner=False)
def get_ai_models():
    """Loads the LLM and Embedding weights into memory exactly once."""
    llm_instance = ChatOllama(
        model="llama3.2:1b",
        temperature=0.7,
    )
    embeddings_instance = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )
    return llm_instance, embeddings_instance

# Instantiate your models instantly from cache
llm, embeddings = get_ai_models()

@st.cache_data(show_spinner=False)
def load_and_split_urls(urls_list, email):
    """Scrapes the web pages and splits text. Caches the result so reruns are instant."""
    headers_dict = {"User-Agent": f"NewsResearchTool/1.0 ({email})"}
    loader = UnstructuredURLLoader(
        urls=urls_list,
        headers=headers_dict,
        strategy="fast"
    )
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', ' ', '.', ','],
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_documents(data)

st.title("NEWS RESEARCH TOOL")
dynamic_bar=st.empty()

st.sidebar.title("NEWS RESARCH URLS")
urls=[]
with st.sidebar.form(key="input_form"):
    clean_email=st.text_input("Enter Email:")
    for i in range(3):
        url_input=st.text_input(f"url{i+1}")
        if url_input.strip():
            urls.append(url_input)
    click=st.form_submit_button("process urls")



if click:
    if not clean_email or "@" not in clean_email or "." not in clean_email:
        st.sidebar.error("❌ A valid email is required to configure network request headers.")
    elif(len(urls)==0):
        st.sidebar.error("❌ Atleast one valid link is required to configure network request headers.")
    else:
        dynamic_bar.text("Loading the data....")
        docs = load_and_split_urls(urls, clean_email)
        dynamic_bar.text("Vectorizing and Storing the data....")
        vector_db=FAISS.from_documents(docs,embeddings)
        vector_db.save_local("faiss_store")
        dynamic_bar.empty()
        st.success("Embedding Vectors Built and Saved Successfully! Ready to chat.", icon="✅")
query=st.chat_input("CHAT HERE")
if query: 
    if os.path.exists("faiss_store"):
        st.subheader("YOUR QUERY:")
        st.write(query)
        vectorstore = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        system_prompt = (
            "You are an investigative assistant analyst. Use the provided context fragments "
            "extracted from web pages to answer the user's question accurately. "
            "If you don't know the answer, say that you don't know.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        generation_chain = prompt | llm | StrOutputParser()
        rag_pipe = (
            {"context": retriever, "input": RunnablePassthrough()}
            | generation_chain
        )
        with st.spinner("Processing through the chain pipeline..."):
            answer = rag_pipe.invoke(query)
            
            st.subheader("Answer:")
            st.write(answer)
    else:
        st.error("⚠️ No vector database found. Please add URLs and click 'Process URLs' first!")