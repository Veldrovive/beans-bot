import discord
import os
from discord.ext import commands
from dotenv import load_dotenv
from typing import Optional, List, Tuple
import time
import asyncio
from pathlib import Path

from db import DB
from cogs.basic_cog import BasicCog
from cogs.robotics_identity import RoleNameCog
from cogs.classifier_cog import ClassifierCog

# --- Configuration Loading ---
load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
GUILD_ID = os.getenv('GUILD_ID')
BOT_CHANNEL_ID = os.getenv('BOT_CHANNEL_ID')
STUDENT_ROLE_ID = os.getenv('ROBOTICS_STUDENT_ROLE_ID')
DB_FILE = Path(__file__).parent / os.getenv('DB_FILE', 'db.sqlite3')

if not all([BOT_TOKEN, GUILD_ID, BOT_CHANNEL_ID, STUDENT_ROLE_ID, DB_FILE]):
    raise ValueError("One or more required environment variables are not set.")


# --- Bot Class ---
class Bot(commands.Bot):
    def __init__(self):
        # We need specific "intents" to allow the bot to read messages and see reactions
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message content
        intents.reactions = True      # Required to see reactions
        # intents = discord.Intents.all()  # Enable all intents for simplicity

        super().__init__(command_prefix='!', intents=intents)

        # Store IDs as integers after validation
        self.guild_id = int(GUILD_ID)
        self.channel_id = int(BOT_CHANNEL_ID)
        self.student_role_id = int(STUDENT_ROLE_ID)

        self.db = DB(DB_FILE)

        if self.db.get_state("message_count") is None:
            self.db.set_state("message_count", 0)

    async def on_ready(self):
        """Called when the bot is connected and ready."""
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')
        await self.add_cog(BasicCog(self))
        await self.add_cog(RoleNameCog(self))
        await self.add_cog(ClassifierCog(self))


# --- Main Execution ---
if __name__ == '__main__':
    bot = Bot()
    bot.run(BOT_TOKEN)