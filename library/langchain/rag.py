from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough , RunnableParallel
from langchain_core.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = "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"

def build_chain(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key = api_key,
        base_url =  "https://openrouter.ai/api/v1"
    )
    if os.path.exists("faiss1_index"):
        vector_store = FAISS.load_local("faiss1_index" ,
                embedding,
                allow_dangerous_deserialization=True
        )
    else:
        vector_store = FAISS.from_documents(chunks , embedding)
        vector_store.save_local("faiss1_index")

    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":5}
    )

    llm = ChatOpenAI(
        model= "gpt-3.5-turbo",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer the question using the provided context if can't be answer using the context use your knowlage 
    If the answer isn't in the context, give your answer 
    
    Context:
    {context}
    
    Question: {question}        
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parser = StrOutputParser()

    chain = (
        RunnableParallel({
        "question" : RunnablePassthrough(),
        "context" : retriever | format_docs,
    })
    | prompt
    | llm
    | parser
    )

    return chain

if __name__ == "__main__":
    chain = build_chain("diffusion.pdf")
    print("Document loaded. Ask questions (Ctrl+C to quit):\n")
    while(True):
        q = input("YOU: ")
        print("BOT: ",end="",flush=True)
        for chunk in chain.stream(q):
            print(chunk,end="",flush=True)
        print("\n")