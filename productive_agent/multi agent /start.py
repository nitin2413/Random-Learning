from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing import TypedDict , Annotated
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel , RunnableLambda
from langgraph.graph import StateGraph , START , END


api = "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"

llm = ChatOpenAI(
    model = "gpt-3.5-turbo",
    base_url="https://openrouter.ai/api/v1",
    api_key = api
)

class AgentState(TypedDict):
    messages : Annotated[list , add_messages]

    sub_questions : list[str]
    research_finds : dict[str , str ]

    selection : str

    generated_code : str
    generated_research: str

    retry_count: int  # ✅ track attempts

    critic_feedback: str
    approved : bool

############################# Agent Function #############################

def orchestrator(state : AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system","analyse the human prompt and only return 'code' or 'research' nothing else. if not able to decide then, only written 'research'"),
        ("human", "{question}")
    ])
    last_message = state["messages"][-1].content
    message1 = prompt.format_messages(question = last_message)
    result = llm.invoke(message1)
    return {
        "selection": result.content.strip()
    }

def query_breaker(state : AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system","Understand the question and break it into small sub questions, return them as a numbered list, one per line "),
        ("human", "{question}")
    ])
    question = state["messages"][-1].content
    message1 = prompt.format_messages(question = question)
    result = llm.invoke(message1)

    lines = result.content.strip().split("\n")
    sub_question = [line.split(". ", 1)[-1] for line in lines if line.strip()]

    return {"sub_questions": sub_question}

'''"1. What is AI?" -> split(". ", 1) -> ["1", "What is AI?"] -> [-1] -> "What is AI?"'''

def researcher(state : AgentState):
    sub_questions = state["sub_questions"]

    # each sub_question gets its own chain
    # x -> og input which we ignore, q -> current value of q :q means return y
    # This is because Python lambdas capture the variable q, not its value at that moment.
    # By the time you call the lambda, the loop is done and q is stuck at the last value.
    # y=q fixes this by binding the current value as a default argument at creation time.
    chains = {
        q: RunnableLambda(lambda x, q=q : q) | llm # y
        for q in sub_questions
    }
    result = RunnableParallel(chains).invoke({})
    research_finds = {q : res.content for q , res in result.items()}
    return {"research_finds" : research_finds}

def research_gen(state : AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system" , "i want you to use the provided context and provide a clean and detail explanation of concept."),
        ("human" , "original question{question} \n\n Research Context{context}")
    ])
    question = state["messages"][-1].content
    context = dict_to_string(state["research_finds"])
    message1 = prompt.format_messages(
        question = question,
        context = context
    )
    result = llm.invoke(message1)
    return {"generated_research": result.content}

def code_gen(state : AgentState):
    if state["retry_count"] == 0:
        human_msg = "Original Query: {query}\n\nResearch Context:\n{context}"
    else:
        human_msg = "Original Query: {query}\n\nResearch Context:\n{context}\n\nCritic Feedback:\n{critic}"
    prompt = ChatPromptTemplate.from_messages([
        ("system" , "Your role is to read and understand provided context and generate code accordingly."),
        ("human" , human_msg)
    ])

    context = dict_to_string(state["research_finds"])
    query = state["messages"][-1].content

    fmt_kwargs = {"query": query , "context":context}
    if state["retry_count"] > 0:
        fmt_kwargs["critic"] = state["critic_feedback"]

    message1 = prompt.format_messages(**fmt_kwargs)
    result = llm.invoke(message1)

    return {"generated_code": result.content}

def critic_feedback(state : AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system","You are an expert in code debugging and optimization. Review the provided code and give improvement areas. If code is correct return 'good' else return 'bad' followed by your feedback"),
        ("human", "Original Question: {question}\n\nGenerated Code:\n{review}")
    ])
    review = state["generated_code"]
    question = state["messages"][-1].content
    message1 = prompt.format_messages(
        question = question,
        review = review
    )
    result = llm.invoke(message1)
    feedback = result.content.strip()
    approved = feedback.upper().startswith("GOOD")

    return {"critic_feedback" : feedback, "approved" : approved}

############################### helping Function #####################

def increment_retry(state : AgentState):
    return {"retry_count": state["retry_count"] + 1}

def dict_to_string(research_finds : dict) -> str:
    return "\n".join([
        f"Question : {key} \n Answer {value}"
        for key , value in research_finds.items()
    ])

############################### Routers ################################

def router(state: AgentState) -> str:
    return state["selection"]

def review_router(state : AgentState):
    if state["approved"]:
        return "approved"
    if state["retry_count"] >= 3 :
        return "approved"

    return "retry"

############################### Graph ###################################

"""
START
  ↓
Orchestrator
  ↓
Query Breaker
  ↓
Researcher
      ↓
 ┌───────────────┐
 │               │
Research      Code
 │               │
 ↓               ↓
ResearchGen   CodeGen
                 ↓
              Critic
                 ↓
           ReviewRouter
            ↙       ↘
        Retry      Approved
          ↓            ↓
   IncrementRetry     END
          ↓
       CodeGen
"""

graph_builder = StateGraph(AgentState)
graph_builder.add_node("orchestrator" , orchestrator)
graph_builder.add_node("query_breaker" , query_breaker)
graph_builder.add_node("researcher" , researcher)
graph_builder.add_node("code_gen" , code_gen)
graph_builder.add_node("research_gen" ,research_gen)
graph_builder.add_node("critic_feedback",critic_feedback)
graph_builder.add_node("increment_retry" , increment_retry)


# graph_builder.set_entry_point("orchestrator") below line do the same thing
graph_builder.add_edge(START , "orchestrator")
graph_builder.add_edge("orchestrator" , "query_breaker")
graph_builder.add_edge("query_breaker" , "researcher")

graph_builder.add_conditional_edges(
    "researcher",
    router ,
    {
        "research" : "research_gen",
        "code" : "code_gen",
    }
)

graph_builder.add_edge("research_gen", END)
graph_builder.add_edge("code_gen","critic_feedback")

graph_builder.add_conditional_edges(
    "critic_feedback",
    review_router,
    {
        "approved": END,
        "retry" : "increment_retry"
    }
)

graph_builder.add_edge("increment_retry","code_gen")

memory = MemorySaver()
agent = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id" : "nitin-103"}}
result1 = agent.invoke(
    {
        "messages": [HumanMessage(content="write code for u-architecture diffusion model(ddpm) using pythorch.")],
        "retry_count":0
    },
    config = config
)
print(result1["selection"])
print(result1)
print(result1["generated_code"])