import streamlit as st
import tiktoken
from loguru import logger

from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

class HuggingFaceLLM:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
    
    def __call__(self, inputs):
        question, context = inputs.split("Context: ")
        result = self.pipeline(question=question.strip(), context=context.strip())
        return result['answer']

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

            logger.info(f"User query: {query}")  # 디버그 출력 추가
            logger.info(f"Conversation chain: {chain}")  # 디버그 출력 추가

            with st.spinner("Thinking..."):
                try:
                    result = chain({"question": query})
                    st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    source_documents = result['source_documents']

                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:  # 문서 출력 개선
                            st.markdown(doc.metadata['source'], help=doc.page_content)

                except Exception as e:  # 예외 처리 추가
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
        elif file_name.endswith('.pptx'):  # pptx 파일에 대한 지원 추가
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

def get_conversation_chain(vectorstore):  # vetorestore 변수명 변경
    model_name = "deepset/roberta-base-squad2"
    llm = HuggingFaceLLM(model_name=model_name)
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="Question: {question}\nContext: {context}\nAnswer:"
    )

    qa_chain = load_qa_with_sources_chain(llm=llm, prompt=prompt_template)

    conversation_chain = ConversationalRetrievalChain(
        combine_docs_chain=qa_chain,
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()
