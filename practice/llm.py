from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from practice.rag import get_retriver
from practice.websearch import web_search

llm = ChatOpenAI(
    model = model ,
    api_key = api_key,
    base_url = base_url
)
query = "rectify and find the new ways to implement langchain"
web_search = web_search.run(query)
retriver = get_retriver()


prompt = ChatPromptTemplate.from_template(
    """
    You Are a helpful agent who will use the given tools for rectifying the problem. You may face while helping me.
    Question : {query}
    WEB_finding: {web_search} 
    retrival: {rag}
    """
)

result = llm.invoke({"query":query , "web_search":web_search, "rag":retriver})
