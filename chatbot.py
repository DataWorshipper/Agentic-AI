import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()

hf_token=os.getenv("HUGGINGFACE_API_KEY")
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

repo_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
llm=HuggingFaceEndpoint(repo_id=repo_id,
                        task="text-generation",
                    max_new_tokens=512,
                 do_sample=False,
                 huggingfacehub_api_token=hf_token,
                 streaming=True
                        )
model=ChatHuggingFace(llm=llm)
memory=MemorySaver()
def chat_node(state:ChatState):
    response=model.invoke(state["messages"])
    return {"messages":[response]}

graph=StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)
workflow=graph.compile(checkpointer=memory)


thread_id='1'

if __name__ == "__main__":
    print("Graph compiled with MemorySaver! (Type 'quit' or 'q' to exit)")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "q"]:
            print("Exiting...")
            break
            
        input_message = {"messages": [HumanMessage(content=user_input)]}
        for event in workflow.stream(input_message, config=config):
            for node_name, node_state in event.items():
                ai_message = node_state["messages"][-1]
                print(f"\nChatbot: {ai_message.content}")





