from langchain_community.document_loaders import PyPDFLoader

pdf_files = [
    "diffusion.pdf"
    # "ddpm.pdf",
    # "latent_diffusion.pdf",
    # "stable_diffusion.pdf"
]

docs = []

for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    docs.extend(loader.load())

print("Total pages loaded:", len(docs))
print(docs[0].metadata)

############# Chunking the Documents #############

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter  = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap= 150,
    separators= ["\n\n", "\n", ".", " "]  # split priority order
)

chunks = splitter.split_documents(docs)

print(f"Total chunks: {len(chunks)}")
# this part return all the chunks created splitter
# for i, chunk in enumerate(chunks):
#     print(f"\n--- Chunk {i+1} ---")
#     print(f"Length: {len(chunk.page_content)} chars")
#     print(f"Content: {chunk.page_content}")
#     print(f"Metadata:{chunks[i].metadata}")

############# Embedding Chunks #############

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

embedding = HuggingFaceEmbeddings(
    model_name = "nomic-ai/nomic-embed-text-v1.5",
    model_kwargs = {"trust_remote_code": True} #“If this model has custom Python code, allow it to execute.”
)

# it only store vector does not convert query text to embedding , it also find the match vector for query vector
if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local(
        "faiss_index",
        embedding,
        allow_dangerous_deserialization=True
    )
else:
    vector_store = FAISS.from_documents(chunks, embedding)
    vector_store.save_local("faiss_index")

print("Vector store ready")
print(f"Total vectors stored: {vector_store.index.ntotal}")

############# Retrieval #############

retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":5} # return top 5 chunks
)

query = "What is the forward diffusion process?"
retrieved_chunks = retriever.invoke(query)
# this part return top 5 related chunks
for i, chunk in enumerate(retrieved_chunks):
    print(f"\n--- Retrieved Chunk {i+1} ---")
    print(chunk.page_content)
    print(f"Source: {chunk.metadata['source']}")

# os.environ["OPENAI_API_KEY"] = "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"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# ==========================================================
# 1) Configure API key
# ----------------------------------------------------------
# Store your API key in an environment variable so the LLM
# client can access it securely.
# Note:
# - 'sk-or-v1-...' is typically an OpenRouter key.
# - Therefore we also specify OpenRouter's base_url below.
# ==========================================================
os.environ["OPENAI_API_KEY"] = "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"


# ==========================================================
# 2) Initialize the Language Model
# ----------------------------------------------------------
'''LangChain builds HTTP request
↓
Send request to base_url
↓
Attach api_key in headers
↓
Include model name in payload
↓
Server authenticates key
↓
Server selects model
↓
Model generates output
↓
Response returned'''
# openroute app : is use to generate api key so base url direct their
# temperature   : 0 => deterministic / consistent answers
# max_tokens    : maximum response length
# ==========================================================
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=1500
)

# ==========================================================
# 3) Create Prompt Template
# ----------------------------------------------------------
# This prompt instructs the model:
# - answer ONLY from retrieved context
# - avoid hallucination
# - present answer in structured format
# - return "I don't know" if context lacks answer
# ==========================================================

prompt = ChatPromptTemplate.from_template('''
You are helpful AI assistant.

You must answer using ONLY the given content.

Format answer like this:
### Summary ###
(medium summary)

### Explanation ###
(detailed explanation)

### Key Points ###
-point1
-point2
-point3
-point4
-point5

### Formula / Equation (if any) ###
(write equation clearly)

If answer is not in context, say:
"I don't know the answer."

Context:
{context}

Question:
{question}
''')

# ==========================================================
# Alternative Prompt (JSON Output)
# ----------------------------------------------------------
# Uncomment this if you want structured JSON output instead
# of markdown-style formatted text.
# ==========================================================
'''
prompt = ChatPromptTemplate.from_template("""
Return JSON format:

{
  "summary": "...",
  "explanation": "...",
  "key_points": ["...", "..."],
  "formula": "..."
}

Context:
{context}

Question:
{question}
""")
'''


# ==========================================================
# 4) Convert Retrieved Chunks into Single Context String
# ----------------------------------------------------------
# Retriever returns multiple document chunks.
# This function combines them into one large text block
# so the LLM can read all retrieved context together.
# ==========================================================
def format_chunks(retrieved):
    return "/n/n".join(chunk.page_content for chunk in retrieved)

# ==========================================================
# 5) Build RAG Pipeline
# ----------------------------------------------------------
# Flow:
# User Question
#      ↓
# Retriever finds relevant chunks
#      ↓
# format_chunks merges retrieved chunks
#      ↓
# Prompt inserts:
#   - context
#   - question
#      ↓
# LLM generates answer
#      ↓
# Output parser converts response to plain string
# ==========================================================
rag_chain = (
    {
        "context": retriever | format_chunks,
        "question": RunnablePassthrough() # make copy of "What happens in diffusion forward pass? Explain briefly in 200 words." and past it
    }
    | prompt
    | llm # returns AIMessage(content="Forward diffusion gradually adds Gaussian noise...")
    | StrOutputParser() # convert AImessage to plain Python string.
)


# ==========================================================
# 6) Run Query
# ----------------------------------------------------------
# Send user question into RAG chain and print final answer.
# ==========================================================
answer = rag_chain.invoke(
    "What happens in diffusion forward pass? Explain briefly in 200 words."
)
print("\n\n")
print("Answer:\n", answer)