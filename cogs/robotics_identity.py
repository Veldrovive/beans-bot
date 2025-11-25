from discord.ext import commands
from typing import Optional, List, Tuple
import discord
import asyncio
import logging

class RoleNameCog(commands.Cog, name="Robotics Identity"):
    required_positive_reactions = 1
    allow_self_voting = True
    positive_reactions = ['👍', '🎉', '❤️', '🔥']
    negative_reactions = ['👎']

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.logger = logging.getLogger("RoboticsIdentity")
        self.role_change_message_table_name = "role_change_messages"

        with self.bot.db.connect() as con:
            con.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.role_change_message_table_name} (
                    guild_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    change_poll_message_id INTEGER PRIMARY KEY,
                    original_message_id INTEGER NOT NULL,
                    change_initiator_user_id INTEGER NOT NULL,
                    new_adjective TEXT NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    active INTEGER DEFAULT 1
                )
            ''')

            self.bot.loop.create_task(self.periodic_poll_updater())

    async def periodic_poll_updater(self):
        await self.bot.wait_until_ready()
        while not self.bot.is_closed():
            await self.update_active_polls()
            await asyncio.sleep(60)  # Check every 60 seconds
            

    def persist_role_change_poll_message(
        self,
        guild_id: int,
        channel_id: int,
        change_poll_message_id: int,
        original_message_id: int,
        change_initiator_user_id: int,
        new_adjective: str
    ):
        with self.bot.db.connect() as con:
            con.execute(f'''
                INSERT INTO {self.role_change_message_table_name} (
                    guild_id,
                    channel_id,
                    change_poll_message_id,
                    original_message_id,
                    change_initiator_user_id,
                    new_adjective,
                    active
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
            ''', (
                guild_id,
                channel_id,
                change_poll_message_id,
                original_message_id,
                change_initiator_user_id,
                new_adjective
            )
    )

    def recall_role_change_poll_message(
        self,
        change_poll_message_id: int
    ) -> Optional[Tuple[int, int, int, int, str, int]]:
        with self.bot.db.connect() as con:
            cur = con.cursor()
            cur.execute(f'''
                SELECT guild_id, channel_id, original_message_id, change_initiator_user_id, new_adjective, active
                FROM {self.role_change_message_table_name}
                WHERE change_poll_message_id = ?
            ''', (change_poll_message_id,))
            result = cur.fetchone()
            return result  # Returns a tuple or None

    def set_poll_active_status(
        self,
        change_poll_message_id: int,
        active: int
    ):
        with self.bot.db.connect() as con:
            con.execute(f'''
                UPDATE {self.role_change_message_table_name}
                SET active = ?
                WHERE change_poll_message_id = ?
            ''', (active, change_poll_message_id))

    def get_active_polls(self) -> List[Tuple[int, int, int, int, str]]:
        with self.bot.db.connect() as con:
            cur = con.cursor()
            cur.execute(f'''
                SELECT guild_id, channel_id, change_poll_message_id, original_message_id, change_initiator_user_id, new_adjective
                FROM {self.role_change_message_table_name}
                WHERE active = 1
            ''')
            results = cur.fetchall()
            return results  # Returns a list of tuples

    def validate_adjective(self, adjective: str) -> Optional[str]:
        """
        Validates and sanitizes the adjective input.
        Adjectives must be non-empty and less than 20 characters.
        Adjectives will be stripped of leading/trailing whitespace and converted to Title Case.
        """
        adjective = adjective.strip()
        if not adjective:
            return None
        if len(adjective) > 20:
            return None
        # Further sanitization can be added here (e.g., removing special characters)
        return adjective.title()

    def construct_role_change_poll_message(self, sanitized_adjective, required_votes):
        return f"React with 👍 or 👎 to rename the robotics students to \"{sanitized_adjective} Robotics Student\" (required votes: {required_votes})"

    async def process_role_change_poll_message(
        self,
        change_poll_message_id: int,
    ):
        """
        Processes the role change poll message to update its status or content.
        """
        record = self.recall_role_change_poll_message(change_poll_message_id)
        if record is None:
            self.logger.debug('Not a role change poll message.')
            return  # Not a tracked message
        guild_id, channel_id, original_message_id, change_initiator_user_id, new_adjective, active = record
        # Get the message object
        guild = self.bot.get_guild(guild_id)
        channel = guild.get_channel(channel_id)
        try:
            message = await channel.fetch_message(change_poll_message_id)
        except discord.NotFound:
            self.logger.warning(f"Message ID {change_poll_message_id} not found in channel ID {channel_id}.")
            return

        if active == 0:
            self.logger.info('Poll is no longer active.')
            return  # Poll is inactive

        # Count the number of unique users who reacted positively and negatively
        positive_voters = set()
        negative_voters = set()
        for reaction in message.reactions:
            async for user in reaction.users():
                if user.id == self.bot.user.id:
                    continue  # Skip the bot's own reactions
                if not self.allow_self_voting and user.id == change_initiator_user_id:
                    continue  # Skip the initiator's own votes if self-voting is not allowed
                if str(reaction.emoji) in self.positive_reactions:
                    positive_voters.add(user.id)
                elif str(reaction.emoji) in self.negative_reactions:
                    negative_voters.add(user.id)

        # We compute the required number of votes left as votes_required + negative_votes - positive_votes
        net_positive_votes = len(positive_voters) - len(negative_voters)
        votes_needed = self.required_positive_reactions - net_positive_votes

        # Check if the poll has passed
        if votes_needed <= 0:
            # Change the role adjective
            student_role_id = self.bot.config_manager.get_student_role_id(guild.id)
            if student_role_id is None:
                self.logger.warning(f"Student role ID not configured for guild {guild.id}")
                return
            role = guild.get_role(student_role_id)
            if role is None:
                self.logger.warning("Role not found.")
                return
            await role.edit(name=f"{new_adjective} Robotics Student")
            # Mark the poll as inactive
            self.set_poll_active_status(change_poll_message_id, 0)
            # Edit the message to indicate success
            await message.edit(content=f"Poll passed! The robotics student role has been renamed to \"{new_adjective} Robotics Student\".")
        else:
            # Edit the message to update the vote count
            await message.edit(content=self.construct_role_change_poll_message(
                new_adjective,
                votes_needed
            ))

    async def update_active_polls(self):
        """Periodically checks and updates all active polls."""
        active_polls = self.get_active_polls()
        for poll in active_polls:
            guild_id, channel_id, change_poll_message_id, original_message_id, change_initiator_user_id, new_adjective = poll
            await self.process_role_change_poll_message(change_poll_message_id)


    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """Handles reactions added to messages, even if the message is not in cache."""
        if payload.user_id == self.bot.user.id:
            return

        await self.process_role_change_poll_message(
            change_poll_message_id=payload.message_id
        )

    @commands.Cog.listener()
    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent):
        """Handles reactions removed from messages, even if the message is not in cache."""
        if payload.user_id == self.bot.user.id:
            return

        await self.process_role_change_poll_message(
            change_poll_message_id=payload.message_id
        )

    @commands.command(
        name="robrole",
        help="Starts a poll to change your robotics student role adjective.",
    )
    async def robrole(self, ctx: commands.Context, *, new_adjective: str):
        """Starts a poll to change your robotics student role adjective."""
        # Check if we are in the correct channel
        bot_channel_id = self.bot.config_manager.get_bot_channel_id(ctx.guild.id)
        if bot_channel_id and ctx.channel.id != bot_channel_id:
            # await ctx.send(f"Please use this command in <#{bot_channel_id}>.")
            return

        sanitized_adjective = self.validate_adjective(new_adjective)
        if sanitized_adjective is None:
            await ctx.send("Invalid adjective. Please provide a non-empty adjective less than 20 characters.")
            return

        message_ctx = await ctx.send(self.construct_role_change_poll_message(
            sanitized_adjective,
            self.required_positive_reactions
        ))
        # Save the poll message info to the database
        self.persist_role_change_poll_message(
            guild_id=ctx.guild.id,
            channel_id=ctx.channel.id,
            change_poll_message_id=message_ctx.id,
            original_message_id=ctx.message.id,
            change_initiator_user_id=ctx.author.id,
            new_adjective=sanitized_adjective
        )