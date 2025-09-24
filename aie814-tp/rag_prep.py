from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.pdf import OnlinePDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
import webvtt
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import LlamafileEmbeddings


def get_vector_store():
    # load env
    load_dotenv()

    # connect to Mistral API to user embedding model
#    embeddings = MistralAIEmbeddings(model="mistral-embed")
    embeddings = LlamafileEmbeddings()

    # in memory vector linked to embedding model
    vector_store = InMemoryVectorStore(embeddings)

    # Split file into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )

    return (vector_store, text_splitter)

def add_pdf_to_store(input_file: str, vector_store: InMemoryVectorStore, text_splitter: TextSplitter):

    # Load file
    loader = OnlinePDFLoader(file_path=input_file)
    docs = loader.load()
    assert 1 == len(docs)
    doc_nb_chars = len(docs[0].page_content)
    assert doc_nb_chars > 0
    print(f'doc has {doc_nb_chars} chars')
    all_splits = text_splitter.split_documents(documents=docs)

    print(f"Split PDF into {len(all_splits)} sub-documents.")

    # Store document
    vector_store.add_documents(documents=all_splits)
    return vector_store



def add_vttfile_to_store(input_file: str, vector_store: InMemoryVectorStore, text_splitter: TextSplitter):
    # convert vtt file into simple text file

    vttfile = webvtt.read(input_file)
    rawvttfile = f'raw-{input_file}'
    with open(rawvttfile, 'w') as f:
        for caption in vttfile.captions:
            f.write(caption.text)
            f.write('\n')

    # Load file
    loader = TextLoader(file_path=rawvttfile)
    docs = loader.load()
    assert 1 == len(docs)
    doc_nb_chars = len(docs[0].page_content)
    assert doc_nb_chars > 0
    print(f'doc has {doc_nb_chars} chars')


    all_splits = text_splitter.split_documents(documents=docs)

    print(f"Split WebVTT into {len(all_splits)} sub-documents.")

    # Store document
    vector_store.add_documents(documents=all_splits)

    return vector_store

def add_raw_text(input_content: str, vector_store: InMemoryVectorStore, text_splitter: TextSplitter):
    document = Document(
        page_content=input_content,
        metadata={"source": "chat"}
    )
    all_splits = text_splitter.split_documents(documents=[document])

    print(f"Split Chat input into {len(all_splits)} sub-documents.")

    # Store document
    vector_store.add_documents(documents=all_splits)

    return vector_store    

if __name__ == '__main__':
    vector_store, text_splitter = get_vector_store()
    vector_store = add_vttfile_to_store('truncated_generated_captions.vtt', vector_store, text_splitter)
    vector_store = add_pdf_to_store('module-architecture-ia.pdf', vector_store, text_splitter)