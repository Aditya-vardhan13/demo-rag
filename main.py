import os
import io
import re
import streamlit as st
import pickle
import time
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tiktoken import encoding_for_model

st.title("News and Document Research Tool 📈📄")

def clean_text(text):
    text = re.sub(r'[.\s]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text.strip()


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
llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")

system_prompt = """You are an AI assistant tasked with answering questions by analysing the provided context. Your goal is to provide accurate, concise, and relevant answers and strictly adhere to the provided content.

When answering:
1. Use the information from the given context.
2. Importantly, If the answer is not in the context, say "I don't have enough information to answer this question."
3. Do not answer from your own database, if the answer to the question is not present in the context provided to you.

Context: {context}

Question: {question}

Answer:"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt
)

def tiktoken_len(text):
    tokenizer = encoding_for_model("gpt-3.5-turbo")
    tokens = tokenizer.encode(text=text)
    return len(tokens)

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
            doc.page_content = clean_text(doc.page_content)  # Apply cleaning here
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
                doc.page_content = clean_text(doc.page_content)  # Apply cleaning here
            docs.extend(pdf_docs)
            os.remove(temp_file_path)
        main_placeholder.text("PDFs processed...✅✅✅")
    
    if not docs:
        main_placeholder.text("No URLs or PDFs to process. Please input at least one source.")
    else:
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '. ', ', ', '! ', '? ', ';', ':', ' - ', '—', '–'],
            chunk_size=1500,
            chunk_overlap=300,
            length_function=tiktoken_len,
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

        # Prepare chunks for download
        chunks_text = ""
        if result["source_documents"]:
            st.subheader("Sources:")
            
            unique_sources = {}
            
            for i, doc in enumerate(result["source_documents"], 1):
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
                
                # Add chunk to the download text
                chunks_text += f"Chunk {i}:\n"
                chunks_text += f"Source: {source}\n"
                chunks_text += f"Page: {page}\n"
                chunks_text += f"Content:\n{doc.page_content}\n\n"
            
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
        
        # Create download button
        if chunks_text:
            bio = io.BytesIO(chunks_text.encode())
            st.download_button(
                label="Download Chunks",
                data=bio,
                file_name="answer_chunks.txt",
                mime="text/plain"
            )
