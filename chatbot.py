import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from database import connection_pool,db_url
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")

if not db_url:
    raise ValueError("DATABASE_URL not found in .env file!")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

repo_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    huggingfacehub_api_token=hf_token,
    timeout=120,
    streaming=True
    )

model = ChatHuggingFace(llm=llm)

def chat_node(state: ChatState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

memory = PostgresSaver(connection_pool)
memory.setup()
workflow = graph.compile(checkpointer=memory)