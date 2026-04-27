"""
## Fat cow
Records stats related to Palace Tang.

`!fatcow`
Alias for `!fatcow help`

`!fatcow help`
Shows help.

`!fatcow start`
Alerts everyone to put in their fat cow amount. And that if they went and didn't eat any fat cow to just do fatcow without any argument.

`!fatcow consumed [amount]?`
Records amount of fat cow eaten (in meters).
If no argument records 0.
Bot reacts when it has recorded.

`!fatcow went`
Records that you went without eating any fat cow.
Also reacts when recorded.

`!fatcow leaderboard`
Reports amount of fat cow that everyone has eaten, the number of times attended, and meters per tang.
"""

HELP = """
Records stats related to Palace Tang.

`!fatcow`
Alias for `!fatcow help`

`!fatcow help`
Shows help.

`!fatcow start`
Records that we went to palace tang today.

`!fatcow consumed [amount]?`
Records amount of fat cow eaten (in meters).

`!fatcow went`
Records that you went without eating any fat cow.

`!fatcow leaderboard`
Returns stats for everyone.
"""

ON_START_MESSAGE = """
The ritual of the meter fat cow has begun...

In solumn accordance, all ye who are of the league of the fat cow, log your consumption.
`!fatcow consumed [amount]`

Those not of the cow, submit your attendance.
`!fatcow went`
"""


import discord
from discord.ext import commands, tasks
import logging
from dataclasses import dataclass
from typing import Optional
import time
import random
from peewee import *
import re
from pydantic import BaseModel
import emoji

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import Bot

db_proxy = Proxy()

# Used to store the times when we went to palace tang
class FatCowVisitEntry(Model):
    server_id = BigIntegerField()
    timestamp = BigIntegerField()

    class Meta:
        database = db_proxy

# DB model for storing amount of fat cow eaten by each user 
class FatCowUserVisitEntry(Model):
    server_id = BigIntegerField()
    user_id = BigIntegerField()
    amount = FloatField()
    timestamp = BigIntegerField()

    class Meta:
        database = db_proxy

class FatCowReactionSpec(BaseModel):
    no_fat_cow_emoji: str = ""
    low_fat_cow_threshold: float = 0.5  # Below this is low
    low_fat_cow_emoji: str = "cow"
    medium_fat_cow_threshold: float = 1.0  # Below this is medium
    medium_fat_cow_emoji: str = ""
    high_fat_cow_threshold: float = 1.5  # Below this is high
    high_fat_cow_emoji: str = ""
    very_high_fat_cow_emoji: str = ""

class FatCowCogConfig(BaseModel):
    reactions: FatCowReactionSpec


class FatCowCog(commands.Cog):
    def __init__(self, bot: "Bot"):
        self.cog_id = "FatCowCog"
        self.bot: "Bot" = bot
        self.logger = logging.getLogger(self.cog_id)

        self.db = self.bot.config_manager.open_peewee_store("fatcow_cog.db")
        db_proxy.initialize(self.db)
        self.db.connect()
        self.db.create_tables([FatCowVisitEntry, FatCowUserVisitEntry])

    def get_config(self, guild_id: int) -> Optional[FatCowCogConfig]:
        server_config = self.bot.config_manager.get_fat_cow_config(guild_id)
        if server_config is None:
            return None
        return FatCowCogConfig(**server_config)

    def _get_reaction_emoji(self, config: FatCowReactionSpec, amount: float) -> str:
        if amount == 0:
            return config.no_fat_cow_emoji
        elif amount < config.low_fat_cow_threshold:
            return config.low_fat_cow_emoji
        elif amount < config.medium_fat_cow_threshold:
            return config.medium_fat_cow_emoji
        elif amount < config.high_fat_cow_threshold:
            return config.high_fat_cow_emoji
        else:
            return config.very_high_fat_cow_emoji

    async def respond_with_help(self, ctx: commands.Context):
        await ctx.send(HELP)
    
    async def start_fatcow(self, ctx: commands.Context):
        assert ctx.guild is not None, "Called start_fatcow without a server"
        FatCowVisitEntry.create(
            server_id=ctx.guild.id,
            timestamp=time.time()
        )
        await ctx.send(ON_START_MESSAGE)

    async def record_fat_cow_consumed(self, ctx: commands.Context, server_config: FatCowCogConfig, amount: float):
        assert ctx.guild is not None, "Called record_fat_cow_consumed without a server"
        FatCowUserVisitEntry.create(
            server_id=ctx.guild.id,
            user_id=ctx.author.id,
            amount=amount,
            timestamp=time.time()
        )
        reaction_emoji = self._get_reaction_emoji(server_config.reactions, amount)
        if reaction_emoji:
            emojis = [e.strip() for e in reaction_emoji.split(",")]
            for e in emojis:
                # 1. Strip colons to get the raw name (e.g., ":custom_cow:" becomes "custom_cow")
                raw_name = e.strip(":")
                
                # 2. Try to find a custom emoji in the server matching this name
                custom_emoji = discord.utils.get(ctx.guild.emojis, name=raw_name)
                
                if custom_emoji:
                    # Found a custom server emoji! Pass the discord.Emoji object directly.
                    actual_reaction = custom_emoji
                elif raw_name.isascii():
                    # 3. Fallback to standard Unicode emoji
                    shortcode = f":{raw_name}:"
                    actual_reaction = emoji.emojize(shortcode, language='alias')
                else:
                    # Then this is already a unicode emoji
                    actual_reaction = raw_name
                
                try:
                    await ctx.message.add_reaction(actual_reaction)
                except discord.HTTPException as err:
                    self.logger.error(f"Failed to add reaction {actual_reaction} in guild {ctx.guild.id}. Error: {err}")
        else:
            self.logger.error(f"No reaction emoji found for amount {amount} in guild {ctx.guild.id}")

    async def report_leaderboard(self, ctx: commands.Context, server_config: FatCowCogConfig):
        assert ctx.guild is not None, "Called report_leaderboard without a server"
        total_times_went = FatCowVisitEntry.select().where(FatCowVisitEntry.server_id == ctx.guild.id).count()

        if total_times_went == 0:
            await ctx.send("Leaderboard:\nNo fat cow visits recorded yet.")
            return

        output_lines = ["Total visits to the Palace:"]
        output_lines.append(f"{total_times_went}")
        output_lines.append("")

        # Perform aggregation and sorting at the database level using peewee's fn.SUM
        user_stats = (
            FatCowUserVisitEntry.select(
                FatCowUserVisitEntry.user_id,
                fn.SUM(FatCowUserVisitEntry.amount).alias('total_amount'),
                fn.COUNT(FatCowUserVisitEntry.user_id).alias('times_went')
            )
            .where(FatCowUserVisitEntry.server_id == ctx.guild.id)
            .group_by(FatCowUserVisitEntry.user_id)
            .order_by(fn.SUM(FatCowUserVisitEntry.amount).desc())
        )
        
        output_lines.append("Leaderboard:")
        for stat in user_stats:
            member = ctx.guild.get_member(stat.user_id)
            member_name = member.display_name if member else f"Unknown User ({stat.user_id})"
            
            meters_per_tang = stat.total_amount / stat.times_went
            
            output_lines.append(f"{member_name}: {stat.total_amount:.2f} meters total, {stat.times_went} attendances, {meters_per_tang:.2f} meters per tang")
            
        await ctx.send("\n".join(output_lines))


    def cog_check(self, ctx: commands.Context) -> bool:
        """Automatically runs before any command in this cog."""
        if ctx.guild is None:
            raise commands.NoPrivateMessage("This command can only be used in a server.")
        
        if self.get_config(ctx.guild.id) is None:
            raise commands.CheckFailure("Fatcow cog is not configured for this server.")
            
        return True

    async def cog_command_error(self, ctx: commands.Context, error: Exception):
        """Catches the errors from cog_check and sends the messages to the user."""
        if isinstance(error, commands.NoPrivateMessage):
            await ctx.send(str(error))
        elif isinstance(error, commands.CheckFailure) and str(error) == "Fatcow cog is not configured for this server.":
            await ctx.send(str(error))
        raise error

    def _get_active_config(self, ctx: commands.Context) -> FatCowCogConfig:
        """Helper to satisfy type-checkers since cog_check guarantees these aren't None."""
        assert ctx.guild is not None
        config = self.get_config(ctx.guild.id)
        assert config is not None
        return config

    @commands.group(name="fatcow", invoke_without_command=True)
    async def fatcow(self, ctx: commands.Context):
        await self.respond_with_help(ctx)

    @fatcow.command(name="help")
    async def help_cmd(self, ctx: commands.Context):
        await self.respond_with_help(ctx)

    @fatcow.command(name="start")
    async def start(self, ctx: commands.Context):
        await self.start_fatcow(ctx)

    @fatcow.command(name="consumed")
    async def consumed(self, ctx: commands.Context, *, amount: float = 0.0):
        await self.record_fat_cow_consumed(ctx, self._get_active_config(ctx), amount)

    @fatcow.command(name="went")
    async def went(self, ctx: commands.Context):
        await self.record_fat_cow_consumed(ctx, self._get_active_config(ctx), 0)

    @fatcow.command(name="leaderboard")
    async def leaderboard(self, ctx: commands.Context):
        await self.report_leaderboard(ctx, self._get_active_config(ctx))