import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from chatbot import workflow
from utilities import generate_thread_id, get_all_threads, update_thread_name

st.set_page_config(page_title="Agentic AI", page_icon="🤖", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

with st.sidebar:
    st.header("💬 Conversations")
    
    if st.button("➕ Start a New Chat", use_container_width=True):
        st.session_state.thread_id = generate_thread_id()
        st.rerun()
        
    st.divider()
    st.subheader("Recent")
    
    threads = get_all_threads()
    for t_id, t_name in threads.items():
        button_label = t_name
        
        is_active = (t_id == st.session_state.thread_id)
        button_type = "primary" if is_active else "secondary"
        
        if st.button(button_label, key=t_id, type=button_type, use_container_width=True):
            st.session_state.thread_id = t_id
            st.rerun()

st.title("🤖 Agentic AI Chatbot")

config = {"configurable": {"thread_id": st.session_state.thread_id}}

snapshot = workflow.get_state(config)
if "messages" in snapshot.values:
    for msg in snapshot.values["messages"]:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

if user_input := st.chat_input("Type your message here..."):
    
    update_thread_name(st.session_state.thread_id, user_input)
    
    st.chat_message("user").write(user_input)
    
    input_message = {"messages": [HumanMessage(content=user_input)]}
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        for chunk, metadata in workflow.stream(
            input_message, 
            config=config, 
            stream_mode="messages"
        ):
            if metadata.get("langgraph_node") == "chat_node" and chunk.content:
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    st.rerun()