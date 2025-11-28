import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

def create_manual_expert(pdf_path):
    print(f"Loading {pdf_path}...")
    
    # 2. LOAD & SPLIT: Break the PDF into digestible chunks
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # We split text so the AI doesn't get overwhelmed. 
    # Chunk size 1000 is a good balance for technical manuals.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split manual into {len(texts)} chunks.")

    # 3. VECTORIZE: Turn text into numbers (Embeddings)
    # This is where the 'physics' happens. We map text to a high-dimensional vector space.
    embeddings = OpenAIEmbeddings()
    
    # Create a local vector database (FAISS) to store these numbers
    print("Building vector database (this might take a moment)...")
    vector_db = FAISS.from_documents(texts, embeddings)
    
    return vector_db

def ask_question(vector_db, query):
    # 4. RETRIEVE: Find the most relevant chunks for the question
    # We use a 'chain' that combines retrieval (finding text) with generation (AI answering)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever()
    )
    
    response = qa_chain.invoke(query)
    return response['result']

if __name__ == "__main__":
    # Test it out!
    # Download a sample PDF manual (e.g., for a microwave or router) and name it 'manual.pdf'
    my_pdf = "manual.pdf" 
    
    if os.path.exists(my_pdf):
        # Build the brain
        brain = create_manual_expert(my_pdf)
        
        # Ask a question
        user_query = input("Ask the manual a question: ")
        answer = ask_question(brain, user_query)
        
        print("\n--- AI Answer ---")
        print(answer)
    else:
        print(f"Error: Please put a file named '{my_pdf}' in this folder.")