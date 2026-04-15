import uuid
import os
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from database import connection_pool

load_dotenv()

def init_db():
    with connection_pool.connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

init_db()

def generate_thread_id() -> str:
    thread_id = str(uuid.uuid4())
    with connection_pool.connection() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (thread_id, title) VALUES (%s, %s)",
            (thread_id, "New Chat")
        )
    return thread_id

def get_all_threads():
    threads = {}
    with connection_pool.connection() as conn:
        cursor = conn.execute("SELECT thread_id, title FROM chat_sessions ORDER BY created_at DESC")
        for row in cursor.fetchall():
            threads[row[0]] = row[1]
    return threads

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
    with connection_pool.connection() as conn:
        cursor = conn.execute("SELECT title FROM chat_sessions WHERE thread_id = %s", (thread_id,))
        result = cursor.fetchone()

        if result and result[0] == "New Chat":
            new_title = generate_title_with_llm(message_content)

            conn.execute(
                "UPDATE chat_sessions SET title = %s WHERE thread_id = %s",
                (new_title, thread_id)
            )