import sqlite3
from datetime import datetime

DB = "agent.db"

def get_conn():
    return sqlite3.connect(DB)

def init_db():
    conn = get_conn()
    c = conn.cursor()

    # Tasks: what you need to do
    c.execute("""
        CREATE TABLE IF NOT EXISTS tasks(
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT NOT NULL,
            priority   TEXT DEFAULT 'medium', -- low / medium / high
            deadline   TEXT,
            est_hours  REAL,
            done       INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            done_at    TEXT
        )
    """)

    # Habits: things you want to do regularly

    c.execute("""
        CREATE TABLE IF NOT EXISTS habits(
           id        INTEGER PRIMARY KEY AUTOINCREMENT,
           name      TEXT NOT NULL ,
           frequency TEXT DEFAULT 'daily' -- daily / weekly 
        )   
    """)

    # Habit logs: each time you check in

    c.execute("""
        CREATE TABLE IF NOT EXISTS habits_log(
          id        INTEGER PRIMARY KEY AUTOINCREMENT,
          habit_id  INTEGER NOT NULL,
          date      TEXT NOT NULL,
          completed INTEGER DEFAULT 1
        )  
    """)

    # Goals: big picture things you're working toward

    c.execute("""
        CREATE TABLE IF NOT EXISTS goals(
          id INTEGER PRIMARY KEY AUTOINCREMENT ,
          title TEXT NOT NULL,
          deadline TEXT,
          progress INTEGER DEFAULT 0, -- 0 to 100
          done INTEGER DEFAULT 0
        )  
    """)

    conn.commit()
    conn.close()
