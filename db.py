import sqlite3
import json
import logging
from typing import Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class DB:
    """
    A simple, persistent key-value store using SQLite.
    Values are serialized to JSON to preserve their type information.
    """
    def __init__(self, db_file: str):
        self.db_file = db_file
        self._init_db()

    def _init_db(self):
        """Initializes the database and creates the necessary table if it doesn't exist."""
        with self._connect() as con:
            con.execute('''
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')
        logger.info("Database initialized.")

    def _connect(self) -> sqlite3.Connection:
        """Creates and returns a new database connection."""
        return sqlite3.connect(self.db_file)

    def set_state(self, key: str, value: Any):
        """
        Saves a key-value pair to the database.
        The value is serialized to a JSON string.
        """
        value_json = json.dumps(value)
        
        with self._connect() as con:
            con.execute(
                "REPLACE INTO state (key, value) VALUES (?, ?)",
                (key, value_json)
            )

    def connect(self) -> sqlite3.Connection:
        """Creates and returns a new database connection."""
        return sqlite3.connect(self.db_file)

    def get_state(self, key: str) -> Optional[Any]:
        """
        Retrieves a value from the database for a given key.
        The value is deserialized from a JSON string.
        """
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("SELECT value FROM state WHERE key = ?", (key,))
            result: Optional[Tuple[str]] = cur.fetchone()
        
        if result:
            # result is a tuple, e.g., ('{"my_list": [1, 2]}',)
            # Deserialize the JSON string back into a Python object
            return json.loads(result[0])
        
        return None # Return None if the key doesn't exist