"""
The council of teds serves the mother of bean.


1. Decide whether cringe or based
2. Mother of bean gets veto

"""

from discord.ext import commands
from typing import Optional, List, Tuple
import discord
import asyncio
import logging

class CouncilOfTedsCog(commands.Cog, name="Council of Teds"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.logger = logging.getLogger("CouncilOfTeds")
        self.active = False

        self.cog_config = self.bot.config_manager.get_council_of_teds_config()
        if self.cog_config is None:
            self.logger.error("Council of Teds config not found")
            return
        self.active = True

    # async def _get_council_reactions(self, )

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        if not self.active:
            self.logger.debug("Council of Teds is not active")
            return
        

        pass

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not self.active:
            self.logger.debug("Council of Teds is not active")
            return
        
        pass