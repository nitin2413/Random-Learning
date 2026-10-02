import os
from langchain_openai import ChatOpenAI
from langchain_community.tools import tool
from langchain_core.messages import HumanMessage , ToolMessage

os.environ["API_KEY"] = "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"
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
@tool
def addition(a:int , b:int) -> int:
    """add two integers."""
    return a + b

@tool
def stock_price(stock : str) -> str:
    """Get the current stock price for a ticker symbol."""
    prices = {"ONGC":300 , "ATGL":600 , "AAPL":500}
    return str(prices.get(stock.upper(),"unknown stock"))

llm_with_tools = llm.bind_tools([addition,stock_price])

def run_with_tools(user_input):
    messages = [HumanMessage(content=user_input)]

    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        tools_map = {"addition": addition, "stock_price":stock_price}
        for tc in response.tool_calls:
            result = tools_map[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(result),tool_call_id = tc["id"]))

print(run_with_tools("What is 42 + 58?"))
print(run_with_tools("What's the stock price of Apple?"))
print(run_with_tools("What is the price of ongc plus the price of atgl?"))


from langchain_community.tools import DuckDuckGoSearchResults
search = DuckDuckGoSearchResults()
result = search.invoke("current share price of ATGL")
print(result )

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
result = wiki.invoke("transformer neural network")
print(result)