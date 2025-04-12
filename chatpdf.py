import argparse
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM as Ollama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--pdf", required=True, help="Path to the PDF file.")
        parser.add_argument("--model", required=True, help="The model to use.")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")

        args = parser.parse_args()

        pdf_path = args.pdf
        model = args.model
        verbose = args.verbose

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Load and split the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = OllamaEmbeddings(model=model)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        # Initialize the LLM and memory
        llm = Ollama(model=model)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create the conversation chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            verbose=verbose
        )

        print(f"Chat initialized with PDF: {pdf_path}")
        print("You can start asking questions about the document.")

        while True:
            request = input("Put your request here (or 'exit') : ")
            if request.lower() == "exit":
                break
            try:
                response = qa_chain.invoke({"question": request})
                print(response["answer"])
                continue
            except Exception as e:
                print(f"An error occurred : {e}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
