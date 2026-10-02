from langchain_community.tools import DuckDuckGoSearchRun , Tool

def wen_search(query):
    duck = DuckDuckGoSearchRun()
    return duck.run(query)

web_search = Tool(
    func= wen_search,
    name = "web_search",
    description="search web"
)


