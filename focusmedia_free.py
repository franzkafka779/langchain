import streamlit as st
import tiktoken
from loguru import logger
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.memory import StreamlitChatMessageHistory

# 더 작은 모델 설정
MODEL = 'beomi/KoAlpaca-Polyglot-3.9B'

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16
    ).to(device="cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU를 사용하도록 설정
    )
except ImportError as e:
    logger.error(f"Error importing model: {e}")
    st.error(f"Error importing model: {e}")

def ask(question, context=''):
    result = pipe(
        f"### 질문: {question}\n\n### 맥락: {context}\n\n### 답변:" if context else f"### 질문: {question}\n\n### 답변:",
        do_sample=True,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2,
    )
    return result[0]['generated_text']

def main():
    st.set_page_config(
        page_title="DirChat",
        page_icon=":books:"
    )

    st.title("_Private Data :red[QA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        process = st.button("Process")
    
    if process:
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                         "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # 채팅 로직
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            if chain is None:
                st.error("Conversation chain is not initialized. Please process the documents first.")
                return

            logger.info(f"User query: {query}")
            logger.info(f"Conversation chain: {chain}")

            with st.spinner("Thinking..."):
                try:
                    # KoAlpaca 모델을 사용하여 질문에 답변
                    result = chain({"question": query})
                    context = " ".join([doc.page_content for doc in result['source_documents']])
                    response = ask(query, context)
                    source_documents = result['source_documents']

                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:
                            st.markdown(doc.metadata['source'], help=doc.page_content)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Error during chain call: {str(e)}")

                st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(file_name)
        elif file_name.endswith('.docx'):
            loader = Docx2txtLoader(file_name)
        elif file_name.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(file_name)
        else:
            st.error(f"Unsupported file format: {file_name}")
            continue
        documents = loader.load_and_split()
        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorstore):
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=None,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()
