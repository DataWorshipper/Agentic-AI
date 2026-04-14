import chainlit as cl
from utilities import generate_thread_id
from chatbot import workflow
from langchain_core.messages import HumanMessage, AIMessage

@cl.on_chat_start
async def on_chat_start():
    thread_id=generate_thread_id()
    config={"configurable":{"thread_id":thread_id}}
    cl.user_session.set("config",config)
    await cl.Message(content="Hello! I am your Agentic AI. How can I help you?").send()
    
    
@cl.on_message
async def on_message(message: cl.Message):
    config = cl.user_session.get("config")
    input_message = {"messages": [HumanMessage(content=message.content)]}
    
    ui_message = cl.Message(content="")
    await ui_message.send()
    
   
    async for chunk, metadata in workflow.astream(
        input_message, 
        config=config, 
        stream_mode="messages" 
    ):
        
        if metadata.get("langgraph_node") == "chat_node" and chunk.content:
            ui_message.content += chunk.content 
            await ui_message.update()