import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

repo_id="deepseek-ai/DeepSeek-V3.2"
llm=HuggingFaceEndpoint(repo_id=repo_id,
                        task="text-generation",
                    max_new_tokens=512,
                 do_sample=False,
                        )
model=ChatHuggingFace(llm=llm)

def chat_node(state:ChatState):
    response=model.invoke(state["messages"])
    return {"messages":[response]}

graph=StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)
workflow=graph.compile()





