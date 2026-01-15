import discord
import os
from discord.ext import commands
from dotenv import load_dotenv
from typing import Optional, List, Tuple
import time
import asyncio
from pathlib import Path
import logging
import json

from db import DB
from cogs.basic_cog import BasicCog
from cogs.robotics_identity import RoleNameCog
from cogs.classifier_cog import ClassifierCog
# from cogs.council_of_teds_cog import CouncilOfTedsCog

from config import ConfigManager

# --- Configuration Loading ---
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)-15s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

BOT_TOKEN = os.getenv('BOT_TOKEN')
DB_FILE = Path(__file__).parent / os.getenv('DB_FILE', 'db.sqlite3')

if not all([BOT_TOKEN, DB_FILE]):
    raise ValueError("One or more required environment variables are not set.")


# --- Bot Class ---
class Bot(commands.Bot):
    def __init__(self, config_path: Path):
        # We need specific "intents" to allow the bot to read messages and see reactions
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message content
        intents.reactions = True      # Required to see reactions
        # intents = discord.Intents.all()  # Enable all intents for simplicity

        super().__init__(command_prefix='!', intents=intents)

        self.config_manager = ConfigManager(config_file=config_path)
        logging.info(f"Loading config from {config_path}")
        logging.info(f"****************\nConfig: {json.dumps(self.config_manager.config, indent=2)}\n****************")

        self.db = DB(DB_FILE)

        if self.db.get_state("message_count") is None:
            self.db.set_state("message_count", 0)

    async def on_ready(self):
        """Called when the bot is connected and ready."""
        logging.info(f'Logged in as {self.user} (ID: {self.user.id})')
        logging.info('------')
        await self.add_cog(BasicCog(self))
        await self.add_cog(RoleNameCog(self))
        await self.add_cog(ClassifierCog(self))
        # await self.add_cog(CouncilOfTedsCog(self))


# --- Main Execution ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the bot.')
    parser.add_argument('config_path', type=Path, help='Path to the config file', default=Path(__file__).parent / 'configs/config_dev.yaml')
    args = parser.parse_args()
    
    bot = Bot(args.config_path)
    bot.run(BOT_TOKEN)