from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from time import sleep
import asyncio

# Read API Keys from .env
load_dotenv()

# Init the connection to the selected model
model = init_chat_model("mistral-large-latest", model_provider="mistralai", temperature=0.8)

# Configure the expected response
system_template = "Répondre par des phrases courtes, maximum 15 mots, et pas plus de 2 phrases. Agir en tant que formateur en informatique bienveillant."

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_template,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Manage history size

trimmer = trim_messages(
    max_tokens=120,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Message persistance
# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state['messages'])
    prompt = prompt_template.invoke({"messages": trimmed_messages})
    response = model.invoke(prompt)
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)



# username = input('Quel est votre nom ? ')
# #config = {"configurable": {"thread_id": username.lower()}}
# messages_history = []
# while True:
#     raw_message = input('#> ')
#     human_message = HumanMessage(raw_message)
#     for chunk, metadata in app.stream({"messages": human_message}, config, stream_mode='messages'):
#         print(chunk.content, end='|')

messages = [
    'Comment est-ce que je m\'appelle ?',
    'Bonjour je m\'appelle {0}',
    'Comment est-ce que je m\'appelle ?'
]

async def do_chat(username: str, config: str):
    print(f'start chat by {username}')
    config = {"configurable": {"thread_id": config}}
    for msg in messages:
        human_message = HumanMessage(msg.format(username.capitalize()))
        response = await app.ainvoke({"messages": human_message}, config)
        print(f'{human_message.content} - {response["messages"][-1].content}')
        sleep(3)

# Config shared by both conversations
asyncio.run(do_chat('gael', 't123'))
asyncio.run(do_chat('mickey', 't123'))

# Config specific to each conversation
asyncio.run(do_chat('gael', 'gael'))
asyncio.run(do_chat('mickey', 'mickey'))
