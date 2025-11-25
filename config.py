import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_file):
            print(f"Config file {self.config_file} not found. Creating default.")
            return {"servers": {}}
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding {self.config_file}: {e}")
            return {"servers": {}}

    def get_server_config(self, guild_id: int) -> Optional[Dict[str, Any]]:
        return self.config.get("servers", {}).get(str(guild_id))

    def get_bot_channel_id(self, guild_id: int) -> Optional[int]:
        server_config = self.get_server_config(guild_id)
        if server_config:
            return server_config.get("bot_channel_id")
        return None

    def get_student_role_id(self, guild_id: int) -> Optional[int]:
        server_config = self.get_server_config(guild_id)
        if server_config:
            return server_config.get("student_role_id")
        return None
