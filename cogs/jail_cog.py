"""
You get put in stocks if enough people react with the :tojail: emoji in the quotes channel.
If above threshold then the person who got quoted gets the "Jail" role and the bot says "Hark, [NAME] has been sent to the stocks for heinous speech!"
The first time they speak after being put in the stocks the bot replies to their comment saying it is throwing a tomato at them
and reacts to the message with the :tomato: emoji.
Further, every subsequent message the bot just adds the reaction and does not reply.
This lasts for N hours.

If you tomato somebody who is not in jail, the bot replies stuff like "HALT! You have tomatoed an innocent! I will remember this!"

When a person is sent to jail, the bot should add the "Jail" role to them. 

If we cannot figure out who the quote is of, connor gets sent to jail.
"""

import discord
from discord.ext import commands, tasks
import logging
from dataclasses import dataclass, asdict
from dataclass_wizard import JSONWizard
from typing import Optional
import time
import json
import random

@dataclass
class JailData(JSONWizard):
    user_id: int
    channel_id: int
    offending_message_id: int
    start_time: int  # ms since epoch
    end_time: int  # ms since epoch
    has_been_humiliated: bool = False  # Whether the user has chatted and been told by the bot that they are in jail

@dataclass
class JailCogConfig:
    jail_emoji: str
    tomato_emoji: str
    to_jail_threshold: int
    jail_length_ms: int
    jailed_role_id: int
    on_jailed_scripts: list[str]  # May have the format string "{jailed_user}"
    humiliate_scripts: list[str]  # May have the format string "{jailed_user}"
    assult_innocent_scripts: list[str]  # May have the format string "{reacting_user}" and "{attacked_user}"
    already_in_jail_scripts: list[str]  # May have the format string "{jailed_user}"
    release_scripts: list[str]  # May have the format string "{jailed_user}"

class JailCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.cog_id = "JailCog"
        self.bot = bot
        self.logger = logging.getLogger(self.cog_id)

        self.currently_jailed_user_ids: dict[int, list[int]] = {}  # guild_id -> list of user_ids
        self.currently_jailed_data: dict[int, dict[int, JailData]] = {}  # guild_id -> user_id -> JailData
        self.historical_jailed_data: dict[int, list[JailData]] = {}  # guild_id -> list of JailData
        self.tomato_counters: dict[int, dict[int, int]] = {}  # guild_id -> user_id -> tomato_count

        self.used_messages: dict[int, set[int]] = {}  # guild_id -> set of message_ids that have already been used to jail someone.

        # Start the background task to check for jail releases
        self.check_jail_timers.start()
        self.logger.info("JailCog initialized")

    def cog_unload(self):
        self.check_jail_timers.cancel()

    def get_config(self, guild_id: int) -> Optional[JailCogConfig]:
        # This assumes config_manager is attached to bot and returns a dict
        if not hasattr(self.bot, "config_manager"):
            self.logger.error("Bot does not have config_manager")
            return None
            
        server_config = self.bot.config_manager.get_jail_config(guild_id)
        if server_config is None:
            self.logger.error(f"JailCogConfig not found for guild {guild_id}")
            return None
        return JailCogConfig(**server_config)

    async def data_set_user_free(self, guild_id: int, user_id: int) -> bool:
        """
        Updates the data stores to mark a user as free. 
        """
        config = self.get_config(guild_id)
        if config is None:
            return False

        if user_id not in self.currently_jailed_user_ids.get(guild_id, []):
            self.logger.warning(f"User {user_id} is not jailed in guild {guild_id}")
            return False

        # Move data to historical
        jail_data = self.currently_jailed_data[guild_id][user_id]
        if guild_id not in self.historical_jailed_data:
            self.historical_jailed_data[guild_id] = []
        
        self.historical_jailed_data[guild_id].append(jail_data)
        
        # Remove from current
        del self.currently_jailed_data[guild_id][user_id]
        self.currently_jailed_user_ids[guild_id].remove(user_id)

        self.dump_data()
        return True

    def data_set_user_jailed(self, guild_id: int, user_id: int, channel_id: int, offending_message_id: int) -> bool:
        """
        Updates the data stores to mark a user as jailed.
        """
        config = self.get_config(guild_id)
        if config is None:
            return False

        if guild_id not in self.currently_jailed_user_ids:
            self.currently_jailed_user_ids[guild_id] = []
        if guild_id not in self.currently_jailed_data:
            self.currently_jailed_data[guild_id] = {}
        if guild_id not in self.used_messages:
            self.used_messages[guild_id] = set()

        if user_id in self.currently_jailed_user_ids[guild_id]:
            return False

        self.currently_jailed_user_ids[guild_id].append(user_id)
        self.currently_jailed_data[guild_id][user_id] = JailData(
            user_id=user_id,
            channel_id=channel_id,
            offending_message_id=offending_message_id,
            start_time=int(time.time() * 1000),
            end_time=int(time.time() * 1000) + config.jail_length_ms,
            has_been_humiliated=False
        )
        self.used_messages[guild_id].add(offending_message_id)

        self.dump_data()
        return True

    def dump_data(self):
        """
        Saves all data to disk. Converts JailData objects to dicts for JSON serialization.
        """
        for guild_id in self.currently_jailed_user_ids:
            # Safely get data or default to empty
            jailed_user_ids = self.currently_jailed_user_ids.get(guild_id, [])
            
            # Convert JailData objects to dicts
            jailed_data_raw = self.currently_jailed_data.get(guild_id, {})
            jailed_data_json = {str(uid): data.to_dict() for uid, data in jailed_data_raw.items()}
            
            # Convert Historical JailData objects to dicts
            historical_raw = self.historical_jailed_data.get(guild_id, [])
            historical_json = [data.to_dict() for data in historical_raw]
            
            used_messages = list(self.used_messages.get(guild_id, set()))
            tomato_counters = self.tomato_counters.get(guild_id, {})

            # Helper to write
            def write_json(filename, data):
                with self.bot.config_manager.open_data_store(guild_id, self.cog_id, filename, "w") as f:
                    json.dump(data, f, indent=4)

            try:
                write_json("currently_jailed_user_ids.json", jailed_user_ids)
                write_json("currently_jailed_data.json", jailed_data_json)
                write_json("historical_jailed_data.json", historical_json)
                write_json("used_messages.json", used_messages)
                write_json("tomato_counters.json", tomato_counters)
            except Exception as e:
                self.logger.error(f"Failed to dump data for guild {guild_id}: {e}")

    def load_data(self):
        """
        Loads all data from disk.
        """
        for guild in self.bot.guilds:
            guild_id = guild.id
            self._init_guild_storage(guild_id)

            data_store_path = self.bot.config_manager.get_data_store_path(guild_id, self.cog_id)
            if not data_store_path.exists():
                continue

            try:
                # Load User IDs
                with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "currently_jailed_user_ids.json", "r") as f:
                    self.currently_jailed_user_ids[guild_id] = json.load(f)

                # Load Jailed Data (Convert dict back to JailData)
                with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "currently_jailed_data.json", "r") as f:
                    raw_data = json.load(f)
                    self.currently_jailed_data[guild_id] = {
                        int(uid): JailData.from_dict(d) for uid, d in raw_data.items()
                    }

                # Load Historical Data
                with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "historical_jailed_data.json", "r") as f:
                    raw_hist = json.load(f)
                    self.historical_jailed_data[guild_id] = [JailData.from_dict(d) for d in raw_hist]

                # Load Used Messages
                with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "used_messages.json", "r") as f:
                    self.used_messages[guild_id] = set(json.load(f))

                # Load Tomato Counters
                with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "tomato_counters.json", "r") as f:
                    # JSON keys are strings, convert back to int
                    raw_tomato = json.load(f)
                    self.tomato_counters[guild_id] = {int(uid): count for uid, count in raw_tomato.items()}

            except FileNotFoundError:
                self.logger.warning(f"Some data files missing for guild {guild_id}, starting with partial defaults.")
            except Exception as e:
                self.logger.error(f"Error loading data for guild {guild_id}: {e}")

    def _init_guild_storage(self, guild_id: int):
        if guild_id not in self.currently_jailed_user_ids:
            self.currently_jailed_user_ids[guild_id] = []
        if guild_id not in self.currently_jailed_data:
            self.currently_jailed_data[guild_id] = {}
        if guild_id not in self.historical_jailed_data:
            self.historical_jailed_data[guild_id] = []
        if guild_id not in self.used_messages:
            self.used_messages[guild_id] = set()
        if guild_id not in self.tomato_counters:
            self.tomato_counters[guild_id] = {}

    def get_is_jailed(self, guild_id: int, user_id: int) -> bool:
        return user_id in self.currently_jailed_user_ids.get(guild_id, [])

    async def try_jail_user(self, guild_id: int, user_id: int, offending_message: discord.Message) -> bool:
        """
        Called when a user should be jailed.
        """
        config = self.get_config(guild_id)
        if config is None:
            return False

        # 1. Check if message already used
        if offending_message.id in self.used_messages.get(guild_id, set()):
            return False

        # 2. Check if user already jailed
        if self.get_is_jailed(guild_id, user_id):
            if config.already_in_jail_scripts:
                script = random.choice(config.already_in_jail_scripts)
                await offending_message.channel.send(script.format(jailed_user=offending_message.author.mention))
            return False

        # 3. Apply Role
        guild = offending_message.guild
        member = guild.get_member(user_id)
        if not member:
            # Try to fetch if not in cache
            try:
                member = await guild.fetch_member(user_id)
            except discord.NotFound:
                self.logger.error(f"Could not find member {user_id} to jail.")
                return False

        role = guild.get_role(config.jailed_role_id)
        if role:
            try:
                await member.add_roles(role, reason="Sent to the stocks by popular vote.")
            except discord.Forbidden:
                await offending_message.reply("I lack the power to jail this person (Missing Permissions).")
                return False
        else:
            self.logger.error(f"Jail role {config.jailed_role_id} not found in guild {guild_id}")
            await offending_message.reply("System Error: Jail role not found.")
            return False

        # 4. Update Data
        self.data_set_user_jailed(guild_id, user_id, offending_message.channel.id, offending_message.id)

        # 5. Announcement and Tomato
        await offending_message.add_reaction(config.tomato_emoji)
        
        # Standard "Hark!" message
        if config.on_jailed_scripts:
            script = random.choice(config.on_jailed_scripts)
            await offending_message.reply(script.format(jailed_user=member.mention))

        return True

    async def on_jail_reaction(self, payload: discord.RawReactionActionEvent):
        """
        Handles a :tojail: reaction being added.
        """
        config = self.get_config(payload.guild_id)
        if not config:
            return

        channel = self.bot.get_channel(payload.channel_id)
        if not channel:
            return

        try:
            message = await channel.fetch_message(payload.message_id)
        except discord.NotFound:
            return

        # Count specific jail emojis
        jail_reaction_count = 0
        for reaction in message.reactions:
            # Check if custom emoji or standard unicode emoji matches config
            if str(reaction.emoji) == config.jail_emoji or getattr(reaction.emoji, 'name', '') == config.jail_emoji:
                jail_reaction_count = reaction.count
                break

        if jail_reaction_count >= config.to_jail_threshold:
            # Target the message author
            await self.try_jail_user(payload.guild_id, message.author.id, message)

    async def on_tomato_reaction(self, payload: discord.RawReactionActionEvent):
        """
        Handles a :tomato: reaction.
        """
        config = self.get_config(payload.guild_id)
        if not config:
            return

        # Determine target user
        # We need the message author.
        channel = self.bot.get_channel(payload.channel_id)
        if not channel:
            return
        try:
            message = await channel.fetch_message(payload.message_id)
        except discord.NotFound:
            return

        target_user_id = message.author.id
        reacting_user_id = payload.user_id

        # 1. If target is in jail -> Increment counter
        if self.get_is_jailed(payload.guild_id, target_user_id):
            if payload.guild_id not in self.tomato_counters:
                self.tomato_counters[payload.guild_id] = {}
            
            current_count = self.tomato_counters[payload.guild_id].get(target_user_id, 0)
            self.tomato_counters[payload.guild_id][target_user_id] = current_count + 1
            self.dump_data()
        
        # 2. If target is NOT in jail -> Warn the thrower
        else:
            if config.assult_innocent_scripts:
                script = random.choice(config.assult_innocent_scripts)
                # Format arguments: reacting user and attacked user
                formatted_msg = script.format(
                    reacting_user=f"<@{reacting_user_id}>",
                    attacked_user=f"<@{target_user_id}>"
                )
                await channel.send(formatted_msg)

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        if payload.user_id == self.bot.user.id:
            return

        config = self.get_config(payload.guild_id)
        if config is None:
            return

        # Check emoji equality (handling both Unicode and Custom emojis)
        emoji_name = payload.emoji.name
        # config.jail_emoji might be the actual unicode char or the name string
        is_jail = (str(payload.emoji) == config.jail_emoji) or (emoji_name == config.jail_emoji)
        is_tomato = (str(payload.emoji) == config.tomato_emoji) or (emoji_name == config.tomato_emoji)
        self.logger.info(f"Emoji: {payload.emoji}, Emoji name: {emoji_name}, Is jail: {is_jail}, Is tomato: {is_tomato}")

        if is_jail:
            self.logger.info(f"Jail emoji detected. Processing jail reaction.")
            await self.on_jail_reaction(payload)
        elif is_tomato:
            self.logger.info(f"Tomato emoji detected. Processing tomato reaction.")
            await self.on_tomato_reaction(payload)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        
        if not self.get_is_jailed(message.guild.id, message.author.id):
            self.logger.info(f"User {message.author.id} is not in jail. Not throwing tomato.")
            return

        config = self.get_config(message.guild.id)
        if config is None:
            self.logger.info(f"Config not found for guild {message.guild.id}. Not throwing tomato.")
            return

        # User is in jail: Throw tomato
        try:
            self.logger.info(f"Throwing tomato at message {message.id} in guild {message.guild.id}.")
            await message.add_reaction(config.tomato_emoji)
        except Exception:
            self.logger.info(f"Failed to add tomato to message {message.id} in guild {message.guild.id}. Not throwing tomato.")
            pass # Fail silently if permissions issue or emoji invalid

        # Check humilitation
        jail_data = self.currently_jailed_data[message.guild.id][message.author.id]
        
        if not jail_data.has_been_humiliated:
            self.logger.info(f"User {message.author.id} is in jail and has not been humiliated. Humiliating...")
            jail_data.has_been_humiliated = True
            
            # Send humiliation message
            if config.humiliate_scripts:
                script = random.choice(config.humiliate_scripts)
                try:
                    self.logger.info(f"Humiliating user {message.author.id} in guild {message.guild.id}.")
                    await message.reply(script.format(jailed_user=message.author.mention))
                except discord.HTTPException:
                    # Fallback if reply fails
                    self.logger.info(f"Failed to reply to message {message.id} in guild {message.guild.id}. Not humiliating.")
                    await message.channel.send(script.format(jailed_user=message.author.mention))
            
            self.dump_data()

    @tasks.loop(seconds=60)
    async def check_jail_timers(self):
        """
        Periodically checks if users should be released from jail.
        """
        self.logger.info("Checking jail timers...")
        current_time_ms = int(time.time() * 1000)

        # Iterate over a copy of keys to avoid modification issues during iteration
        guild_ids = list(self.currently_jailed_data.keys())

        for guild_id in guild_ids:
            config = self.get_config(guild_id)
            if not config:
                continue

            guild = self.bot.get_guild(guild_id)
            if not guild:
                continue

            # Identify users to release
            users_to_release = []
            for user_id, data in self.currently_jailed_data[guild_id].items():
                if current_time_ms >= data.end_time:
                    users_to_release.append(user_id)

            if len(users_to_release) > 0:
                self.logger.info(f"Users to release: {users_to_release}")

            # Release them
            for user_id in users_to_release:
                member = guild.get_member(user_id)
                if not member:
                    # Attempt fetch
                    try:
                        member = await guild.fetch_member(user_id)
                    except discord.NotFound:
                        pass # User left server, just clean up data
                
                # Remove Role
                if member:
                    role = guild.get_role(config.jailed_role_id)
                    if role:
                        try:
                            await member.remove_roles(role, reason="Served their time in the stocks.")
                        except discord.Forbidden:
                            self.logger.warning(f"Could not remove jail role from {user_id} in {guild_id}")

                    # Send Release Message
                    if config.release_scripts:
                        script = random.choice(config.release_scripts)
                        # Try to send in the channel they were jailed in, fallback to system channel or nowhere
                        channel = guild.get_channel(self.currently_jailed_data[guild_id][user_id].channel_id)
                        if channel:
                            await channel.send(script.format(jailed_user=member.mention))

                # Update Data
                await self.data_set_user_free(guild_id, user_id)

    @check_jail_timers.before_loop
    async def before_check_jail_timers(self):
        await self.bot.wait_until_ready()

# import discord
# from discord.ext import commands
# import logging
# from dataclasses import dataclass
# from dataclass_wizard import JSONWizard
# from typing import Optional
# import time

# @dataclass
# class JailData(JSONWizard):
#     user_id: int
#     channel_id: int
#     offending_message_id: int
#     start_time: int  # ms since epoch
#     end_time: int  # ms since epoch
#     has_been_humiliated: bool = False  # Whether the user has chatted and been told by the bot that they are in jail

# @dataclass
# class JailCogConfig:
#     jail_emoji: str
#     tomato_emoji: str
#     to_jail_threshold: int
#     jail_length_ms: int
#     jailed_role_id: int
#     humiliate_scripts: list[str]  # May have the format string "{jailed_user}"
#     assult_innocent_scripts: list[str]  # May have the format string "{reacting_user}" and "{attacked_user}"
#     already_in_jail_scripts: list[str]  # May have the format string "{jailed_user}"
#     release_scripts: list[str]  # May have the format string "{jailed_user}"

# class JailCog(commands.Cog):
#     def __init__(self, bot: commands.Bot):
#         self.cog_id = "JailCog"
#         self.bot = bot
#         self.logger = logging.getLogger(self.cog_id)

#         self.currently_jailed_user_ids: dict[int, list[int]] = {}  # guild_id -> list of user_ids
#         self.currently_jailed_data: dict[int, dict[int, JailData]] = {}  # guild_id -> user_id -> JailData
#         self.historical_jailed_data: dict[int, list[JailData]] = {}  # guild_id -> list of JailData
#         self.tomato_counters: dict[int, dict[int, int]] = {}  # guild_id -> user_id -> tomato_count

#         self.used_messages: dict[int, set[int]] = {}  # guild_id -> set of message_ids that have already been used to jail someone. More jail emojis on these does nothing.

#     def get_config(self, guild_id: int) -> Optional[JailCogConfig]:
#         server_config = self.bot.config_manager.get_jail_config(guild_id)
#         if server_config is None:
#             self.logger.error(f"JailCogConfig not found for guild {guild_id}")
#             return None
#         return JailCogConfig(**server_config)

#     def data_set_user_free(self, guild_id: int, user_id: int) -> bool:
#         """
#         Updates the data stores to mark a user as free. Does not have side effects besides triggering a data dump.
#         """
#         config = self.get_config(guild_id)
#         if config is None:
#             self.logger.error(f"JailCogConfig not found for guild {guild_id}")
#             return False

#         if user_id not in self.currently_jailed_user_ids.get(guild_id, []):
#             self.logger.warning(f"User {user_id} is not jailed in guild {guild_id}")
#             return False

#         # Then we move the jailed data from currently_jailed_data to historical_jailed_data and
#         # remove the user from currently_jailed_user_ids
#         self.historical_jailed_data[guild_id].append(self.currently_jailed_data[guild_id][user_id])
#         del self.currently_jailed_data[guild_id][user_id]
#         self.currently_jailed_user_ids[guild_id].remove(user_id)

#         self.dump_data()
#         return True

#     def data_set_user_jailed(self, guild_id: int, user_id: int, offending_message_id: int) -> bool:
#         """
#         Updates the data stores to mark a user as jailed. Does not have side effects besides triggering a data dump.

#         Adds the user to currently_jailed_user_ids and currently_jailed_data, and adds the message to used_messages.
#         """
#         config = self.get_config(guild_id)
#         if config is None:
#             self.logger.error(f"JailCogConfig not found for guild {guild_id}")
#             return False

#         if user_id in self.currently_jailed_user_ids.get(guild_id, []):
#             self.logger.warning(f"User {user_id} is already jailed in guild {guild_id}")
#             return False

#         self.currently_jailed_user_ids[guild_id].append(user_id)
#         self.currently_jailed_data[guild_id][user_id] = JailData(
#             user_id=user_id,
#             offending_message_id=offending_message_id,
#             start_time=int(time.time() * 1000),
#             end_time=int(time.time() * 1000) + config.jail_length_ms,
#             has_been_humiliated=False
#         )

#         self.dump_data()
#         return True

#     def dump_data(self):
#         """
#         Saves all data to disk.
#         """
#         for guild_id in self.currently_jailed_user_ids:
#             jailed_user_ids = self.currently_jailed_user_ids[guild_id]
#             jailed_data = self.currently_jailed_data[guild_id]
#             historical_jailed_data = self.historical_jailed_data[guild_id]
#             used_messages = self.used_messages[guild_id]
#             tomato_counters = self.tomato_counters[guild_id]

#             with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "currently_jailed_user_ids.json", "w") as f:
#                 json.dump(jailed_user_ids, f)
#             with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "currently_jailed_data.json", "w") as f:
#                 json.dump(jailed_data, f)
#             with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "historical_jailed_data.json", "w") as f:
#                 json.dump(historical_jailed_data, f)
#             with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "used_messages.json", "w") as f:
#                 json.dump(list(used_messages), f)
#             with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "tomato_counters.json", "w") as f:
#                 json.dump(tomato_counters, f)

#     def load_data(self):
#         """
#         Loads all data from disk.
#         """
#         for guild in self.bot.guilds:
#             guild_id = guild.id

#             self.currently_jailed_user_ids[guild_id] = []
#             self.currently_jailed_data[guild_id] = {}
#             self.historical_jailed_data[guild_id] = []
#             self.used_messages[guild_id] = set()
#             self.tomato_counters[guild_id] = {}

#             data_store_path = self.bot.config_manager.get_data_store_path(guild_id, self.cog_id)
#             if data_store_path.exists():
#                 with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "currently_jailed_user_ids.json", "r") as f:
#                     self.currently_jailed_user_ids[guild_id] = json.load(f)
#                 with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "currently_jailed_data.json", "r") as f:
#                     self.currently_jailed_data[guild_id] = json.load(f)
#                 with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "historical_jailed_data.json", "r") as f:
#                     self.historical_jailed_data[guild_id] = json.load(f)
#                 with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "used_messages.json", "r") as f:
#                     self.used_messages[guild_id] = set(json.load(f))
#                 with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "tomato_counters.json", "r") as f:
#                     self.tomato_counters[guild_id] = json.load(f)

#     def get_is_jailed(self, guild_id: int, user_id: int) -> bool:
#         """
#         Returns True if the user is currently jailed.
#         """
#         return user_id in self.currently_jailed_user_ids.get(guild_id, [])

#     async def try_jail_user(self, guild_id: int, user_id: int, offending_message: discord.Message) -> bool:
#         """
#         Called when a user should be jailed. Updates the data and sends messages to the appropriate channels.
#         """
#         # Check if this message has already been used to jail someone
#         if offending_message.id in self.used_messages.get(guild_id, []):
#             return False

#         # Check if the user is already jailed
#         if self.get_is_jailed(guild_id, user_id):
#             raise NotImplementedError("TODO send a message from the already_in_jail_scripts")
#             return False

#         config = self.get_config(guild_id)
#         if config is None:
#             self.logger.error(f"JailCogConfig not found for guild {guild_id}")
#             return False

#         self.data_set_user_jailed(guild_id, user_id, offending_message.id)
#         await offending_message.add_reaction(config.tomato_emoji)
#         return True

#     async def on_jail_reaction(self, payload: discord.RawReactionActionEvent):
#         """
#         Handles a :tojail: reaction being added to a message.
#         Checks how many :tojail: reactions have been added to the message. If it exceeds the to_jail_threshold then the user is sent to jail.
#         This means adding the data, adding the tomato reaction, and sending a message from the jail_scripts.
#         """
#         self.logger.info(f"Jail reaction added to message {payload.message_id} by user {payload.user_id}")
    
#     async def on_tomato_reaction(self, payload: discord.RawReactionActionEvent):
#         """
#         Handles a :tomato: reaction being added to a message.
#         If the person who had the tomato thrown at them is in jail, then we increment their tomato counter.
#         If they are not, then we issue a warning to the thrower from the assult_innocent_scripts.
#         """
#         self.logger.info(f"Tomato reaction added to message {payload.message_id} by user {payload.user_id}")

#     @commands.Cog.listener()
#     async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
#         """
#         Checks if the :tojail: or :tomato: reaction was added to a message.
#         """
#         if payload.user_id == self.bot.user.id:
#             return

#         config = self.get_config(payload.guild_id)
#         if config is None:
#             self.logger.error(f"JailCogConfig not found for guild {payload.guild_id}")
#             return

#         if payload.emoji.name == config.to_jail_emoji:
#             await self.on_jail_reaction(payload)
#         elif payload.emoji.name == config.tomato_emoji:
#             await self.on_tomato_reaction(payload)

#     @commands.Cog.listener()
#     async def on_message(self, message: discord.Message):
#         """
#         Check if the person sending the message is in jail. If they are, throw a tomato at them.
#         If has_been_humiliated is false, then set it to true and send a message from the humiliate_scripts.
#         """
#         if message.author.bot:
#             return
        
#         if not self.get_is_jailed(message.guild.id, message.author.id):
#             return

#         config = self.get_config(message.guild.id)
#         if config is None:
#             self.logger.error(f"JailCogConfig not found for guild {message.guild.id}")
#             return

#         # Otherwise we know the user is in jail, so we throw a tomato at them
#         await message.add_reaction(config.tomato_emoji)

#         # Check if the user has been humiliated yet
#         jail_data = self.currently_jailed_data[message.guild.id][message.author.id]
#         if not jail_data.has_been_humiliated:
#             jail_data.has_been_humiliated = True
#             self.dump_data()
#             raise NotImplementedError("TODO send a message from the humiliate_scripts")

#     # TODO have a interval running that checks whether people should be released from jail
            