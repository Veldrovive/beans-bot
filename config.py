import json
import os
from typing import Dict, Any, Optional
import yaml
from typing import List

class ConfigManager:
    def __init__(self, config_file: str = 'config.yaml'):
        self.config_file = config_file
        # Check we have a json or yaml config file
        if self.config_file.endswith(".json"):
            self.config = self.load_json_config()
        elif self.config_file.endswith(".yaml"):
            self.config = self.load_yaml_config()
        else:
            raise ValueError("Config file must be a json or yaml file.")

    def load_json_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_file):
            print(f"Config file {self.config_file} not found. Creating default.")
            return {"servers": {}}
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding {self.config_file}: {e}")
            return {"servers": {}}

    def load_yaml_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_file):
            print(f"Config file {self.config_file} not found. Creating default.")
            return {"servers": {}}
        
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error decoding {self.config_file}: {e}")
            return {"servers": {}}

    def get_all_server_ids(self) -> List[int]:
        return list(self.config.get("servers", {}).keys())

    def get_server_config(self, guild_id: int) -> Optional[Dict[str, Any]]:
        return self.config.get("servers", {}).get(int(guild_id))

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

    def get_model_device(self) -> str:
        return self.config.get("model_device", "cpu")

    def get_torch_cpu_threads(self) -> int:
        return self.config.get("torch_cpu_threads", 1)

    def get_classifier_config(self, guild_id: int) -> Optional[Dict[str, Any]]:
        server_config = self.get_server_config(guild_id)
        if server_config:
            return server_config.get("classifier_confs")

    def get_council_of_teds_config(self, guild_id: int) -> Optional[Dict[str, Any]]:
        server_config = self.get_server_config(guild_id)
        if server_config:
            return server_config.get("council_of_teds_confs")
