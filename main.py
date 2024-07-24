import os
import streamlit as st
import pickle
import time
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

os.environ['OPENAI_API_KEY'] = "sk-proj-6X5PaV6KwEkilgaeZn9MT3BlbkFJ2baHh1m6sjf2OGIOPBi5"

st.title("News and Document Research Tool 📈📄")

st.sidebar.title("News Article URLs")
urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

if st.sidebar.button("Clear Cache and Reset"):
    st.cache_data.clear()
    st.session_state.clear()
    st.experimental_rerun()

st.sidebar.title("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Choose PDF Files", accept_multiple_files=True)

process_clicked = st.sidebar.button("Process URLs and PDFs")

file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0, max_tokens=1000)


system_prompt = """You are an AI assistant tasked with answering questions by analysing the provided context from news articles and documents. Your goal is to provide accurate, concise, and relevant answers and strictly adhere to the provided content.

When answering:
1. Use the information from the given context.
2. Strictly, If the answer is not in the context, say "I don't have enough information to answer this question."
3. If asked for an opinion, clarify that you're an AI and don't have personal opinions, but can provide information from the sources.

Context: {context}

Question: {question}

Answer:"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt
)

# import shutil
# try:
#     if os.path.exists(file_path):
#         shutil.rmtree(file_path, ignore_errors=True)
#         st.sidebar.success("Cleared existing index. Please process new documents.")
# except Exception as e:
#     st.sidebar.warning(f"Unable to clear existing index: {str(e)}")

if process_clicked:
    docs = []
    main_placeholder.text("Processing started...✅✅✅")
    
    # Process URLs
    if urls:
        loader = UnstructuredURLLoader(urls=urls)
        url_docs = loader.load()
        for doc in url_docs:
            doc.metadata['source'] = doc.metadata.get('source', 'Unknown URL')
            doc.metadata['page'] = 'N/A'
        docs.extend(url_docs)
        main_placeholder.text("URLs processed...✅✅✅")
    
    # Process PDFs
    if uploaded_files:
        for uploaded_file in uploaded_files:
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_file_path)
            pdf_docs = loader.load()
            for doc in pdf_docs:
                doc.metadata['source'] = uploaded_file.name
            docs.extend(pdf_docs)
            os.remove(temp_file_path)
        main_placeholder.text("PDFs processed...✅✅✅")
    
    if not docs:
        main_placeholder.text("No URLs or PDFs to process. Please input at least one source.")
    else:
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitting...Started...✅✅✅")
        docs = text_splitter.split_documents(docs)
        
        # Create embeddings and save to FAISS index
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index
        vectorstore_openai.save_local(file_path)
        main_placeholder.text("Processing complete! You can now ask questions.✅✅✅")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        vectorstore = FAISS.load_local(file_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt_template
            }
        )
        
        result = qa_chain({"query": query})
        
        st.header("Answer")
        st.write(result["result"])

        if result["source_documents"]:
            st.subheader("Sources:")
            
            unique_sources = {}
            
            for doc in result["source_documents"]:
                source = doc.metadata['source']
                page = doc.metadata.get('page', 'N/A')
                
                is_url = source.startswith('http://') or source.startswith('https://')
                
                if is_url:
                    unique_sources[source] = None
                else:
                    if page != 'N/A':
                        page = int(page) + 1  # Convert to 1-based page numbering
                    
                    if source in unique_sources:
                        if page not in unique_sources[source] and page != 'N/A':
                            unique_sources[source].append(page)
                    else:
                        unique_sources[source] = [page] if page != 'N/A' else []
            
            # Display the unique sources
            for source, pages in unique_sources.items():
                if unique_sources[source] is None:
                    st.write(f"- {source}")
                elif not pages:
                    st.write(f"- Document: {source}")
                elif len(pages) == 1:
                    st.write(f"- Document: {source}, Page: {pages[0]}")
                else:
                    pages_str = ', '.join(map(str, sorted(pages)))
                    st.write(f"- Document: {source}, Pages: {pages_str}")
