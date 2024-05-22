import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.memory import StreamlitChatMessageHistory
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

class HuggingFaceLLM:
    def __init__(self, model_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, **kwargs)

    def generate_text(self, prompt):
        result = self.pipeline(prompt, max_length=150, num_return_sequences=1)
        return result[0]['generated_text']

def main():
    st.set_page_config(page_title="DirChat", page_icon=":books:")
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
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        translated_response = "죄송합니다, 응답을 생성하지 못했습니다."  # 초기화
        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            if chain is None:
                st.error("Conversation chain is not initialized. Please process the documents first.")
                return

            logger.info(f"User query: {query}")
            logger.info(f"Conversation chain: {chain}")

            with st.spinner("Thinking..."):
                try:
                    result = chain({"question": query})
                    response = result['answer']
                    source_documents = result['source_documents']

                    # 번역기 사용
                    translator = Translator()
                    translated_response = translator.translate(response, src='en', dest='ko').text

                    st.markdown(translated_response)
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:
                            st.markdown(doc.metadata['source'], help=doc.page_content)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Error during chain call: {str(e)}")

                st.session_state.messages.append({"role": "assistant", "content": translated_response})

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100, length_function=tiktoken_len)
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorstore):
    llm = HuggingFaceLLM(model_name="EleutherAI/gpt-neo-2.7B")
    prompt_template = PromptTemplate(input_variables=["context", "question"], template="{context}\n\nQ: {question}\nA:")

    def custom_llm_chain(question):
        context = " ".join([doc.page_content for doc in vectorstore.similarity_search(question, k=3)])
        prompt = prompt_template.format(context=context, question=question)
        return llm.generate_text(prompt)
    
    conversation_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        combine_docs_chain=custom_llm_chain,
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True),
        verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()
