import os
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory



os.environ['API_KEY'] = "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"

model = ChatOpenAI(
    model = "gpt-3.5-turbo",
    api_key = os.getenv("API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

prompt = ChatPromptTemplate([
    ("system","You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}")
])

parser = StrOutputParser()

chain = prompt | model | parser

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {"configurable":{ "session_id" : "nitin-001" }}

r1 = chain_with_memory.invoke(
    input = {"input":"My name is Nitin Singh and i love to play football"},
    config = config
)
print(r1)

r2 = chain_with_memory.invoke(
    input = {"input":"yes , my favorite team is man united. what was my name "},
    config = config
)
print(r2)

r3 = chain_with_memory.invoke(
    {"input": "What sport did I just mention I love?"},
    config={"configurable": {"session_id": "other-user"}}
)
print(r3)

# older version of langchain
# from langchain_classic.memory import ConversationSummaryBufferMemory
#
# memory = ConversationSummaryBufferMemory(
#     llm = model,
#     max_token_limit=200,
#     return_messages = True
# )
#
# memory.save_context(
#     {"input": "My name is Nitin. I'm learning LangChain."},
#     {"output": "Great to meet you Nitin! LangChain is powerful."}
# )
#
# memory.save_context(
#     {"input": "I've already done RAG and LCEL."},
#     {"output": "Excellent progress! Memory is next."}
# )
#
# print(memory.moving_summary_buffer)

from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START , MessagesState , StateGraph

trimmer = trim_messages(
    max_tokens = 500,
    token_counter = model ,
    strategy = "last",
    start_on = "human",
    include_system = True,
    allow_partial = False
)

def call_model(state: MessagesState):
    trimmed = trimmer.invoke(state["messages"])
    response = model.invoke(trimmed)
    return {"messages": [response]}

workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START,"model")
workflow.add_node("model", call_model)

checkpoint = MemorySaver()
app = workflow.compile(checkpointer=checkpoint)

from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "nitin-001"}}

# Turn 1
r1 = app.invoke(
    {"messages": [HumanMessage("My name is Nitin. I'm learning LangChain.")]},
    config=config
)
print(r1["messages"][-1].content)
# "Nice to meet you Nitin! LangChain is powerful."

# Turn 2 — history is loaded automatically via checkpointer
r2 = app.invoke(
    {"messages": [HumanMessage("What's my name?")]},
    config=config
)
print(r2["messages"][-1].content)

# See the full state of a thread at any point
state = app.get_state(config)

for msg in state.values["messages"]:
    role = "User" if msg.type == "human" else "Bot"
    print(f"{role}: {msg.content}")
# "Your name is Nitin."

#
# prompt = ChatPromptTemplate([
#     ("system", "You are a helpful assistant."),
#     MessagesPlaceholder(variable_name="history"),
#     ("human", "{input}")
# ])
#
# chain = (
#     RunnablePassthrough.assign(
#         history=RunnableLambda(
#             lambda x: trimmer.invoke(x["history"])
#         )
#     )
#     | prompt
#     | model
#     | parser
# )
# result = chain.invoke({
#     "input" : "hello my name is nitin i love to play football",
#     "history":"[]"
# })
# print(result)