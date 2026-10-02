from langchain_openai import ChatOpenAI , OpenAI
import os

os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(model="gpt-3.5-turbo",
             api_key=os.getenv("OPENAI_API_KEY"),
             base_url="https://openrouter.ai/api/v1",
             )
# result = llm.invoke("what is ml?")
# print(result)

from langchain_core.messages import (
HumanMessage,AIMessage,SystemMessage,ToolMessage
)

messages = [
    SystemMessage("you are a helpful assistant"),
    HumanMessage("Capital of India"),
    AIMessage("Delhi"),
    HumanMessage("What language do they speak?")
]

# result1 = llm.invoke(messages)
# print(result1)

from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system","you are an expert in {domain}"),
    ("human","{question}"),
])

message1 = template.format_messages(
    domain = "ML",
    question = "explain diffusion model architecture in detail for deep learning ,and also each and every phase of it."
)

# result2 = llm.invoke(message1)
# print(type(result2))

from langchain_core.output_parsers import StrOutputParser , JsonOutputParser
from pydantic import Field , BaseModel

parser = StrOutputParser()
# result = llm.invoke(message1)
# answer = parser.invoke(result)
# print(answer)

class MovieReview(BaseModel):
    title: str = Field(description="movie-title"),
    rating: int = Field(description="Rate between 0-10"),
    summary: str = Field(description="one line summary of the movie")

json_parser = JsonOutputParser(pydantic_object=MovieReview)

template1 = ChatPromptTemplate.from_messages([
    ("system" ,"Return a movie review as JSON. {format_instructions}" ),
    ("human","Review the movie: {movie}")
])

message = template1.format_messages(
    format_instructions = json_parser.get_format_instructions(),
    movie = "inception"
)

# raw = llm.invoke(message)
# result = json_parser.invoke(raw)
# print(result)
########################################################################################################################


translate_prompt = ChatPromptTemplate([
    ("system","Translate to the following {language}"),
    ("human", "{text}")
])

translate_chain = translate_prompt | llm | parser

answer1 = translate_chain.invoke({
    "language" : "Hindi",
    "text" : "hey there my name is nitin singh"
})

print(answer1)

summarize_prompt = ChatPromptTemplate([
    ("system","Summarize the following in one line"),
    ("human", " {text}")
])

from langchain_core.runnables import RunnableLambda

def strtotext(s : str) -> dict :
    return {"text" : s}

# final_chain = translate_chain | RunnableLambda(strtotext) | summarize_prompt
#
# answer2 = final_chain.invoke({
#     "language":"Hindi",
#     "text":"nural network are inter connected node use to predict output for a model by adjusting their weights"
# })
#
# print(answer2)

from langchain_core.runnables import RunnableParallel

cons_prompt = ChatPromptTemplate([
    ("system","list cons on {topic} give 5 points")
])

pros_prompt = ChatPromptTemplate([
    ("system","list pros on {topic} give 5 points")
])

# cons_chain = cons_prompt | llm | parser
# pros_chain = pros_prompt | llm | parser
#
# combine_chain = RunnableParallel(
#     pros = pros_chain,
#     cons = cons_chain
# )
#
# result = combine_chain.invoke({"topic":"langchain"})
# print(result["pros"])
# print(result["cons"])

from langchain_core.runnables import RunnablePassthrough

qa_prompt = ChatPromptTemplate([
    ("system","answer the question precisely."),
    ("human","{text}")
])

qa_chain = qa_prompt | llm | parser

full_chain = RunnableParallel(
    question  =  RunnablePassthrough(),
    answer    =  qa_chain
)

result = full_chain.invoke({"text":"can you do coding."})

# print(result)

for chunk in translate_chain.stream({"language":"Spanish","text":"Good Morning"}):
    print(chunk , end = " ", flush=True)

result = translate_chain.batch([
    {"language": "Spanish", "text": "Good Morning"},
    {"language":"Hindi","text":"Good Morning"},
    {"language":"protuges","text":"Good Morning"}
])
print(result)

explain_prompt = ChatPromptTemplate([
    ("system","Explain {topic} in one clear paragraph. No bullet points."),
])

explain_chain = explain_prompt | llm | parser

quiz_prompt = ChatPromptTemplate([
    ("system","Based on the following explanation, write one multiple-choice quiz question with 4 options and mark the correct answer.\n\nExplanation: {explanation}"),
])

quiz_chain = quiz_prompt | llm | parser
# { output of stage one
#    "explanation": "Neural networks are computational models...",
#    "topic": {
#        "topic": "Neural Networks"
#    }
# }
final_chain = (
    RunnableParallel(
        explanation = explain_chain,
        topic = RunnablePassthrough()
    )
    | RunnableLambda (lambda x:{ # formating output as per are need (Dictionary format )
          "explanation": x["explanation"],
          "topic_original" : x["topic"]["topic"]
    })
    | RunnableParallel (
           explanation = RunnablePassthrough() | RunnableLambda(lambda x:x["explanation"]), # formating (return only the explanation )
           quiz = RunnableLambda(lambda x:{"explanation":x["explanation"]}) | quiz_chain
    )
)

result = final_chain.invoke({"topic": "gradient descent"})
print("=== EXPLANATION ===")
print(result["explanation"])
print("\n=== QUIZ QUESTION ===")
print(result["quiz"])