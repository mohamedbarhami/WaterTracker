from dotenv import load_dotenv
from langchain_community.llms.llamafile import Llamafile
from langchain.chat_models import init_chat_model
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from rag_prep import get_vector_store, add_pdf_to_store, add_vttfile_to_store, add_raw_text
import httpx

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Read API Keys from .env
load_dotenv()

# get vector store
vector_store, text_splitter = get_vector_store()
vector_store = add_vttfile_to_store('auto_generated_captions.vtt', vector_store, text_splitter)
vector_store = add_pdf_to_store('module-architecture-ia.pdf', vector_store, text_splitter)

# Init the connection to the selected model
# model = init_chat_model("mistral-large-latest", model_provider="mistralai", temperature=0.6)
model = Llamafile()
# Configure the expected response
template = """Répondre à la question posée par 3 phrases maximum contenant maximum 15 mots. Agir en tant que formateur en informatique bienveillant. Basé la réponse uniquement sur le contexte. Si le contexte ne contient pas la réponse, alors dire que vous ne connaissez pas la réponse.

Context: {context}

Question: {question}

"""


prompt_template = PromptTemplate.from_template(template)

# Manage history size

trimmer = trim_messages(
    max_tokens=10000,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Include vector db

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

# Message persistance


# Define the function that calls the model
def call_model(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state['context'])
    prompt = prompt_template.invoke({'question': state['question'], 'context': docs_content})
    response = model.invoke(prompt)
    return {'answer': response}

# Define a new graph
workflow = StateGraph(state_schema=State).add_sequence([retrieve, call_model])


# Define the (single) node in the graph
workflow.add_edge(START, "retrieve")
#workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

username = input('Quel est votre nom ? ')
config = {"configurable": {"thread_id": username.lower()}}
messages_history = []

while True:
    raw_message = input('#> ')
    if raw_message == 'upload':
        raw_content = input('content to add#> ')
        vector_store = add_raw_text(raw_content, vector_store, text_splitter)
    else:
        #human_message = HumanMessage(raw_message)
        try:
            # for chunk, metadata in app.stream({"question": raw_message}, config, stream_mode='messages'):
            #     print(chunk.content, end='')
            # print()
            response = app.invoke({"question": raw_message}, config)
            print(response['answer'])
        except httpx.HTTPStatusError as e:
            print('##### Error')
            print(e)