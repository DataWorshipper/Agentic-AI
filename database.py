import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
load_dotenv()

db_url = os.getenv("DATABASE_URI")
connection_pool = ConnectionPool(
    conninfo=db_url,
    max_size=20,
    kwargs={"autocommit": True} 
)