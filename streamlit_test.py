import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from googletrans import Translator

st.header('한기대 AI 휴먼 강의 프로젝트', divider='rainbow')
st.markdown('''궁금한게 있으시면 질문해보세요!! :balloon:''')

# 환경 변수 로드
load_dotenv()

# 벡터 데이터베이스와 모델 초기화
persist_directory = 'db'
embedding = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

class ChatGuide:
    def __init__(self, model_name='llama3'):
        self.model = ChatOllama(model=model_name)
        self.prompt = PromptTemplate.from_template(
            """
            당신은 학교 온라인 강의 플랫폼에서 학생들의 질문에 답변하는 AI 튜터입니다.
            <지침>
            1. 제공된 컨텍스트를 바탕으로 학생의 질문에 대해 최대한 정확하고 상세하게 답변해 주세요.
            2. 답변은 학생의 수준에 맞는 쉬운 언어를 사용하되, 전문적인 지식을 바탕으로 작성해 주세요.
            3. 답변에 관련된 핵심 개념이나 용어가 있다면 간단히 설명해 주세요.
            4. 실생활 예시나 시각 자료(이미지, 그래프, 다이어그램 등)를 활용하여 이해를 돕는 것이 좋습니다. 예시는 <예시></예시> 태그로 감싸 주세요.
            5. 추가 학습에 도움이 될 만한 자료나 참고 문헌이 있다면 링크 또는 출처를 제공해 주세요.
            6. 제공된 컨텍스트만으로 답변하기 어려운 경우, 관련 정보가 부족함을 알리고 추가 컨텍스트를 요청해 주세요.
            7. 반드시 답변은 한국어로만 해주세요.
            8. 모든 답변은 한국어로 작성해 주세요. 다른 언어를 사용하지 마세요.
            </지침>
            <컨텍스트>
            {context}
            </컨텍스트>
            <학생 질문>
            {question}
            </학생 질문>
            <답변>
            """
        )
        
        # 기존의 벡터 데이터베이스를 사용하여 retriever 설정
        self.retriever = vectordb.as_retriever(search_kwargs={"k": 2})

        # QA 체인을 구성
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model  
            | StrOutputParser())
        
    def ask(self, query: str):
        return self.chain.invoke(query)

# ChatGuide 초기화
if "chat_guide" not in st.session_state:
    st.session_state["chat_guide"] = ChatGuide(model_name="llama3")

# 번역기 초기화
translator = Translator()

# Streamlit 애플리케이션 구성
st.title("경영정보시스템개론 Chatbot")

with st.chat_message("user"):
    st.write("MIS가 무엇입니까?!")
    st.write("MIS가 최근 대두되는 이유는 무엇인가?")
    st.write("MIS를 도입한 기업들의 사례를 알려줘")
                 
# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def model_res_generator(query):
    response = st.session_state["chat_guide"].ask(query)
    return response

# 대화 기록 표시
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # 사용자 입력 기록
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 모델 응답 생성
    response = model_res_generator(prompt)
    
    # 모델 응답을 한국어로 번역
    translated_response = translator.translate(response, src='en', dest='ko').text
    
    with st.chat_message("assistant"):
        st.markdown(translated_response)
        
    st.session_state["messages"].append({"role": "assistant", "content": translated_response})
