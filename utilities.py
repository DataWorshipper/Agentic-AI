import uuid
import os
import json
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

TITLES_FILE = "chat_titles.json"

def load_threads():
    if os.path.exists(TITLES_FILE):
        with open(TITLES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_threads(threads_dict):
    with open(TITLES_FILE, "w") as f:
        json.dump(threads_dict, f, indent=4)

ACTIVE_THREADS = load_threads()

def generate_thread_id() -> str:
    thread_id = str(uuid.uuid4())
    global ACTIVE_THREADS
    ACTIVE_THREADS = load_threads()
    ACTIVE_THREADS[thread_id] = "New Chat"
    save_threads(ACTIVE_THREADS)
    return thread_id

def get_all_threads():
    return load_threads()

def generate_title_with_llm(user_message: str) -> str:
    try:
        hf_token = os.getenv("HUGGINGFACE_API_KEY")
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct", 
            task="text-generation",
            max_new_tokens=15, 
            temperature=0.1, 
            do_sample=False,
            huggingfacehub_api_token=hf_token
        )
        chat_model = ChatHuggingFace(llm=llm)
        
        messages = [
            SystemMessage(content="You are a title generator. Summarize the user's message in 2 to 4 words to be used as a chat UI title. Output ONLY the words for the title. Do not include quotes, punctuation, or any conversational filler."),
            HumanMessage(content=user_message)
        ]
        
        response = chat_model.invoke(messages)
        title = response.content.strip().strip('"').strip("'")
        
        if len(title) > 35:
            return title[:32] + "..."
            
        return title
        
    except Exception as e:
        print(f"Title generation failed: {e}")
        return user_message[:25] + "..."

def update_thread_name(thread_id: str, message_content: str):
    global ACTIVE_THREADS
    ACTIVE_THREADS = load_threads()
    
    if thread_id in ACTIVE_THREADS and ACTIVE_THREADS[thread_id] == "New Chat":
        title = generate_title_with_llm(message_content)
        ACTIVE_THREADS[thread_id] = title
        save_threads(ACTIVE_THREADS)