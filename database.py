import os
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URI")
from psycopg_pool import ConnectionPool
connection_pool = ConnectionPool(
    conninfo=db_url,
    max_size=20,
    kwargs={"autocommit": True} 
)