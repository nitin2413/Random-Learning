from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough , RunnableParallel

def get_retriver(file_path):

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunk = splitter.split_documents(docs)
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key="xyz",
        base_url="https://openrouter.ai/api/v1"
    )
    faiss_store = FAISS.from_documents(chunk , embedding)
    faiss_store.save_local("fissa.idx")
    retriever = faiss_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    # llm = ChatOpenAI(
    #     model="gpt-3.5-turbo",
    #     api_key=api_key,
    #     base_url="https://openrouter.ai/api/v1"
    # )
    #
    # prompt = ChatPromptTemplate.from_template("""
    #     You are a helpful assistant. Answer the question using the provided context if can't be answer using the context use your knowlage
    #     If the answer isn't in the context, give your answer
    #
    #     Context:
    #     {context}
    #
    #     Question: {question}
    #     """)
    #
    #
    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)
    #
    #
    # parser = StrOutputParser()
    #
    # chain = (
    #         RunnableParallel({
    #             "question": RunnablePassthrough(),
    #             "context": retriever | format_docs,
    #         })
    #         | prompt
    #         | llm
    #         | parser
    # )
    return retriever
