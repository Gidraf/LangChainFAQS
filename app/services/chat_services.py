import bs4
from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain, create_history_aware_retriever 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import ConfigurableFieldSpec
import os
from langchain_community.chat_message_histories import RedisChatMessageHistory
from psycopg_pool import ConnectionPool
import uuid

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory
import psycopg


from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


conn_info = os.environ.get("SERVICE_DATABASE_URL") # Fill in with your connection info
sync_connection = psycopg.connect(conn_info)
service_url = os.environ.get("SERVICE_URL")
# Create the table schema (only needs to be done once)
table_name = "chat_history"
PostgresChatMessageHistory.create_tables(sync_connection, table_name)

# Initialize the chat history manager


# model = "tinydolphin"

# llm = ChatOllama(model=model)


service_url = os.environ.get("WEB_URL")

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=(f"{service_url}/#services",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            name=("main")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,  add_start_index=True)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

template = """You are the bot agent managing our services CVPAP, Use the following pieces of context to answer the questions about our services and strictly act as the whatsapp bot agent because you are.
Please answer by telling the client who we are from the context, If the converation is casual be casual and answer based on the conversation e.g if the question is greetings answer with greetings to, if the question is introduction answer with our introductions too. In case the question is not understandable this I mean it's not in the context of this business context, kindly search it from the context list of services. If it's not available, just say: "Thank you for your interest in our services I can not process your request. Kindly contact our customer care through customer@cvpap.com", then list the services from context, List the services as a numbered list.
Always be kind, polite and professional to the customers when resppnding.
If the question is about knowing how to revamp the cv, always add catlogue link when saying our process of cv creation, use the context information and reply telling our client how they can revamp their cv including all the details if possible use emojis to express emotions..

{context}

Helpful Answer:"""



custom_rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)



contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, always return response based on the user question and our cuurent context \
The answers should be a follow up of most previous answer or question from the user, \
Return the answer as per organisation context and the user questions and responses."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


question_answer_chain = create_stuff_documents_chain(llm, custom_rag_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# _history = []

store = {}


def get_message_history(session_id: str) -> PostgresChatMessageHistory:
    return PostgresChatMessageHistory(
    table_name,
    session_id,
    sync_connection=sync_connection)


with_message_history = RunnableWithMessageHistory(
    rag_chain,
    get_message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_key="output",
 
)


# @celery_service.task
def process_enquiery(question, session_id):
    
    chat_history = PostgresChatMessageHistory(
    table_name,
    session_id,
    sync_connection=sync_connection)
    _history = []
    if chat_history.get_messages():
        _history = chat_history.get_messages()
    else:
        _history = []
    answer = with_message_history.invoke(
        { "input": question, "chat_history": _history,  "output_key":'result'},
        config={"configurable": {"session_id": session_id}},
    )
    chat_history.add_messages([
    SystemMessage(content=contextualize_q_system_prompt),
    AIMessage(content=answer["answer"]),
    HumanMessage(content=question),
    ])
    return answer["answer"]
    

