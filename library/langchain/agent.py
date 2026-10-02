from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage , SystemMessage
from langchain_core.tools import tool
from typing import TypedDict , Annotated
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph , END
from langgraph.graph.message import add_messages
import operator
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = "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"   # from smith.langchain.com
os.environ["LANGCHAIN_PROJECT"]    = "langchain/agent"

@tool
def add(a:int , b:int) -> int :
    """Add two integers."""
    return a + b

@tool
def multipy(a : int , b : int )-> int :
    """Multiply two integers."""
    return a * b

tools = [add , multipy]
api = "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"
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key= api , base_url="https://openrouter.ai/api/v1")
llm_with_tools = llm.bind_tools(tools)

# Every node returns {"messages": [new_msg]} and LangGraph merges it in.
class AgentState(TypedDict):
    messages : Annotated[list , operator.add]

def agent_node(state : AgentState):
    # state["messages"] is the full conversation so far
    response = llm_with_tools.invoke(state["messages"])
    # returning a list triggers the operator.add reducer — it APPENDS
    return {"messages":[response]}

# This is the routing logic. LangGraph calls this after agent_node
def should_continue(state : AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # → goes to ToolNode
    return END

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent" , agent_node)
graph_builder.add_node("tools" , ToolNode(tools)) # ToolNode handles execution automatic

graph_builder.set_entry_point("agent")
# Conditional edge: after agent_node, call should_continue to decide where to go
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools":"tools", # if should_continue returns "tools" → go to tools node
        END : END
    }
)
# After tools execute, always go back to agent
graph_builder.add_edge("tools","agent")

memory = MemorySaver()
agent = graph_builder.compile(checkpointer=memory)

config = {"configurable": { "thread_id" : "user_001" }}

response = agent.invoke(
    {"messages":[HumanMessage(content = "What is 6 * 7, then add 3 to that?")]},
    config = config
)
print(response["messages"][-1])

####################################################################################
# Custom state — adding your own keys #

class AgentStates(TypedDict):
    messages : Annotated[list , add_messages]
    username : str
    task_count : int
    last_tool_use : str

def agent_node(state : AgentStates):
    name = state["username"]
    system = f"you are helping {name}"
    response =llm_with_tools.invoke([SystemMessage(system)] + state["messages"])
    return {
        "messages": [response],
        "last_tool_use":""
    }

def tool_node_wrapper(state : AgentStates):
    tool_result = ToolNode(tools).invoke(state)
    last = state["messages"][-1]
    tool_name = last.tool_calls[0]["name"] if last.tool_calls else ""
    return {
        **tool_result,
        "last_tool_use": tool_name,
        "task_count" : state["task_count"] + 1
    }

def should_continue(state : AgentStates):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # → goes to ToolNode
    return END

graph_builders = StateGraph(AgentStates)
graph_builders.add_node("agent" , agent_node)
graph_builders.add_node("tools" , tool_node_wrapper) # ToolNode handles execution automatic

graph_builders.set_entry_point("agent")
# Conditional edge: after agent_node, call should_continue to decide where to go
graph_builders.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools":"tools", # if should_continue returns "tools" → go to tools node
        END : END
    }
)
# After tools execute, always go back to agent
graph_builders.add_edge("tools","agent")

memory = MemorySaver()
agent = graph_builders.compile(checkpointer=memory)

responce = agent.invoke({
    "messages":[HumanMessage("How much is 6 * 7 then + 3?")],
    "username" : "NITIN SINGH",
    "task_count": 0,
    "last_tool_use" : ""
}, config = config)

print(responce["messages"][-1].content)
print("Task Count:", responce["task_count"])
print("Last Tool Used:", responce["last_tool_use"])

