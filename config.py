import json
from pathlib import Path
import os
from typing import Dict, Any, Optional
import yaml
from typing import List
from contextlib import contextmanager
from tinydb import TinyDB
from peewee import SqliteDatabase

class ConfigManager:
    def __init__(self, config_file: Path | str = 'config.yaml'):
        self.config_file = Path(config_file)
        # Check we have a json or yaml config file
        if self.config_file.suffix == ".json":
            self.config = self.load_json_config()
        elif self.config_file.suffix == ".yaml":
            self.config = self.load_yaml_config()
        else:
            raise ValueError("Config file must be a json or yaml file.")
        
        assert "lazy_store_folder" in self.config, "Config file must contain lazy_store_folder."
        self.data_store_path = Path(self.config.get("lazy_store_folder"))
        self.data_store_path.mkdir(parents=True, exist_ok=True)

    def get_data_store_path(self, server_id: int, cog_id: str) -> Path:
        return self.data_store_path / str(server_id) / cog_id

    @contextmanager
    def open_data_store(self, server_id: int, cog_id: str, filename: str, mode: str = "r"):
        cog_path = self.get_data_store_path(server_id, cog_id)
        cog_path.mkdir(parents=True, exist_ok=True)
        file_path = cog_path / filename
        with open(file_path, mode) as f:
            yield f

    def open_tinydb_store(self, server_id: int, cog_id: str, filename: str):
        cog_path = self.get_data_store_path(server_id, cog_id)
        cog_path.mkdir(parents=True, exist_ok=True)
        file_path = cog_path / filename
        return TinyDB(file_path)

    def open_peewee_store(self, filename: str):
        file_path = self.data_store_path / filename
        return SqliteDatabase(file_path)

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

    def get_jail_config(self, guild_id: int) -> Optional[Dict[str, Any]]:
        server_config = self.get_server_config(guild_id)
        if server_config:
            return server_config.get("jail_confs")

    def get_birthday_config(self, guild_id: int) -> Optional[Dict[str, Any]]:
        server_config = self.get_server_config(guild_id)
        if server_config:
            return server_config.get("birthday_confs")
        return None
        