# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
# from langchain_core.prompts import PromptTemplate
# from youtube_transcript_api import YouTubeTranscriptApi
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableParallel, RunnableLambda

# # Initialize environment and models
# load_dotenv()
# llm = HuggingFaceEndpoint(
#     repo_id = 'deepseek-ai/DeepSeek-V4-Pro',
#     task='text-generation'
#     )
# model = ChatHuggingFace(llm=llm)
# strparser = StrOutputParser()

# # ---------------------------------------------------------
# # 1. Indexing the video transcript
# # ---------------------------------------------------------

# print("Fetching transcript for video ID\n\n")
# vid_id = "DsewHeVbL-0"

# def merge_transcript(transcript_list):
#     return "\n\n".join([t['text'] for t in transcript_list])

# try:
#     # Use get_transcript and extract the 'text' key from the returned dictionaries
#     yt_api = YouTubeTranscriptApi()
#     transcript_list = yt_api.fetch(vid_id)
#     # transcript = " ".join([t.text for t in transcript_list])
#     print('--------------------------------')
#     print("Transcript fetched successfully!")
#     print('--------------------------------')
# except Exception as e:
#     print('--------------------------------')
#     print(f"Error fetching transcript: {e}")
#     print('--------------------------------')
#     exit()

# # Chunking the text
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks = splitter.split_text(transcript)

# # Creating the Vectorstore (using HuggingFaceEmbeddings and FAISS)
# embeddings = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-l-v2.0")
# vectorstore = FAISS.from_texts(chunks, embeddings)

# # ---------------------------------------------------------
# # 2. Query Rewriting & Retrieval
# # ---------------------------------------------------------
# print("Generating optimized search query...\n\n")
# query_raw = "What is AGI and how does it work?"

# # Removed the live python object from the prompt template
# prompt_to_generate_correct_query = PromptTemplate(
#     template='''You are an expert in semantic search. Your task is to rewrite the user's query so that it is optimized for retrieving relevant information from a vector database. 
#     If the query is already well-structured, return it exactly as is. Do not answer the query, just rewrite it.
    
#     Original Query: {query}
#     Optimized Query:''',
#     input_variables=['query']
# )

# # Generate the optimized query
# chain_to_query = prompt_to_generate_correct_query | model | strparser
# query = chain_to_query.invoke({"query": query_raw})
# print("--------------------------------")
# print(f"Optimized search query: {query.strip()}")
# print("--------------------------------")

# # Retrieve documents (pass string directly to invoke)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
# retrieved_docs = retriever.invoke(query.strip())
# context = "\n\n".join([doc.page_content for doc in retrieved_docs])
# print("\n\n--------------------------------")
# print(f"Retrieved context:\n\n{context}")
# print("--------------------------------")

# # ---------------------------------------------------------
# # 3. Generating the final answer
# # ---------------------------------------------------------
# print("Generating final answer...\n\n")
# prompt = PromptTemplate(
#     template="""You are an expert in processing vast amounts of text from YouTube video transcripts. Provide a clear, concise answer to the question based ONLY on the provided video transcript. If the answer cannot be found in the context, say "I don't know based on this transcript."
    
#     Question: {question}
    
#     Video Transcript Context: 
#     {context}
#     """,
#     input_variables=['question', 'context']
# )

# # Execute final chain
# chain_to_answer = prompt | model | strparser
# answer = chain_to_answer.invoke({"question": query, "context": context})


# # parallel chain to generate question + context for the model
# par_chain = RunnableParallel({
#     "question": chain_to_query | strparser,
#     "context": chain_to_query| retriever | strparser 
# })

# final_chain = par_chain | chain_to_answer


# print("\n--- Final Answer ---")
# print(answer)
# print("--------------------")


import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

# Load environment variables (Ensure HUGGINGFACEHUB_API_TOKEN is set in .env)
load_dotenv()

def initialize_models():
    """Initializes the LLM and Embedding models via Hugging Face."""
    print("[*] Initializing Models...")
    # Initialize the LLM (DeepSeek for deep reasoning capabilities)
    llm = HuggingFaceEndpoint(
        repo_id='deepseek-ai/DeepSeek-V4-Pro',
        task='text-generation',
    )
    chat_model = ChatHuggingFace(llm=llm)
    
    # Initialize the Embedding Model (Snowflake for high-quality semantic vectors)
    embeddings = HuggingFaceEmbeddings(
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0"
    )
    
    return chat_model, embeddings

def fetch_and_chunk_transcript(vid_id: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Fetches the YouTube transcript and splits it into manageable text chunks."""
    print(f"[*] Fetching transcript for video ID: {vid_id}...")
    try:
        yt_api = YouTubeTranscriptApi()
        transcript_list = yt_api.fetch(vid_id)
        transcript_text = " ".join([t.text for t in transcript_list])
        print("✅ Transcript fetched successfully!")
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        exit(1)

    print("[*] Chunking text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(transcript_text)

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    vid_id = "DsewHeVbL-0"
    user_query = "What is AGI and how does it work?"
    strparser = StrOutputParser()
    
    model, embeddings = initialize_models()
    chunks = fetch_and_chunk_transcript(vid_id)
    
    print("[*] Indexing vectors into FAISS...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ---------------------------------------------------------
    # 2. Defining the LCEL Chains
    # ---------------------------------------------------------
    print(f"\n[*] Constructing LCEL Pipeline for query: '{user_query}'")
    
    # Chain A: Query Optimizer
    prompt_to_generate_correct_query = PromptTemplate(
        template='''You are an expert in semantic search. Rewrite the user's query so that it is optimized for retrieving relevant information from a vector database. 
        Return ONLY the optimized query string. Do not include any introductory words, explanations, or quotes.
        
        Original Query: {query}
        Optimized Query:''',
        input_variables=['query']
    )
    optimized_query_chain = prompt_to_generate_correct_query | model | strparser

    # Chain B: Document Retrieval
    retrieval_chain = retriever | RunnableLambda(format_docs)

    # Chain C: Parallel Orchestrator
    # This automatically passes the optimized query into both the prompt's 'question' slot 
    # AND the retriever to fetch the context.
    par_chain = RunnableParallel({
        "context" : retrieval_chain,
        "question" : RunnablePassthrough()
    })

    # Chain D: Final Answer Generation
    final_prompt = PromptTemplate(
        template="""You are an expert in processing vast amounts of text from YouTube video transcripts. Provide a clear, concise answer to the question based ONLY on the provided video transcript. If the answer cannot be found in the context, say "I don't know based on this transcript."
        
        Question: {question}
        
        Video Transcript Context: 
        {context}
        """,
        input_variables=['question', 'context']
    )

    # The Master Chain
    final_chain = optimized_query_chain | par_chain | final_prompt | model | strparser

    # ---------------------------------------------------------
    # 3. Execution
    # ---------------------------------------------------------
    print("[*] Executing RAG Pipeline...\n")
    # A single invoke triggers the entire orchestrated pipeline
    final_answer = final_chain.invoke({"query": user_query})

    print("="*60)
    print("🎯 FINAL ANSWER:")
    print("="*60)
    print(final_answer.strip())
    print("="*60 + "\n")

if __name__ == "__main__":
    main()