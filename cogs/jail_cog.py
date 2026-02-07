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
    scot_free_scripts: list[str]  # Format: "{jailed_user}" - User found with role but no data, set free
    forgotten_prisoner_scripts: list[str] # Format: "{jailed_user}" - User found with role but no data, re-jailed

class JailCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.cog_id = "JailCog"
        self.bot = bot
        self.logger = logging.getLogger(self.cog_id)

        self.currently_jailed_user_ids: dict[int, list[int]] = {}  # guild_id -> list of user_ids
        self.currently_jailed_data: dict[int, dict[int, JailData]] = {}  # guild_id -> user_id -> JailData
        self.historical_jailed_data: dict[int, list[JailData]] = {}  # guild_id -> list of JailData
        self.tomato_counters: dict[int, dict[int, int]] = {}  # guild_id -> user_id -> tomato_count
        self.tomato_history: dict[int, list[tuple[int, int, int, int, bool]]] = {}  # guild_id -> list of (thrower_user_id, attacked_user_id, message_id, timestamp, is_innocent)

        self.used_messages: dict[int, set[int]] = {}  # guild_id -> set of message_ids that have already been used to jail someone.

        self.load_data()

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
            tomato_history = self.tomato_history.get(guild_id, [])

            # Helper to write
            def write_json(filename, data):
                self.logger.info(f"Writing {filename} for guild {guild_id}")
                with self.bot.config_manager.open_data_store(guild_id, self.cog_id, filename, "w") as f:
                    json.dump(data, f, indent=4)

            try:
                write_json("currently_jailed_user_ids.json", jailed_user_ids)
                write_json("currently_jailed_data.json", jailed_data_json)
                write_json("historical_jailed_data.json", historical_json)
                write_json("used_messages.json", used_messages)
                write_json("tomato_counters.json", tomato_counters)
                write_json("tomato_history.json", tomato_history)
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
                self.logger.info(f"Loading currently_jailed_user_ids.json for guild {guild_id}")
                with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "currently_jailed_user_ids.json", "r") as f:
                    self.currently_jailed_user_ids[guild_id] = json.load(f)
                self.logger.info(f"Loaded {len(self.currently_jailed_user_ids[guild_id])} currently jailed users for guild {guild_id}")

                # Load Jailed Data (Convert dict back to JailData)
                self.logger.info(f"Loading currently_jailed_data.json for guild {guild_id}")
                with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "currently_jailed_data.json", "r") as f:
                    raw_data = json.load(f)
                    self.currently_jailed_data[guild_id] = {
                        int(uid): JailData.from_dict(d) for uid, d in raw_data.items()
                    }
                self.logger.info(f"Loaded {len(self.currently_jailed_data[guild_id])} currently jailed data for guild {guild_id}")

                # Load Historical Data
                self.logger.info(f"Loading historical_jailed_data.json for guild {guild_id}")
                with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "historical_jailed_data.json", "r") as f:
                    raw_hist = json.load(f)
                    self.historical_jailed_data[guild_id] = [JailData.from_dict(d) for d in raw_hist]
                self.logger.info(f"Loaded {len(self.historical_jailed_data[guild_id])} historical jailed data for guild {guild_id}")

                # Load Used Messages
                self.logger.info(f"Loading used_messages.json for guild {guild_id}")
                with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "used_messages.json", "r") as f:
                    self.used_messages[guild_id] = set(json.load(f))
                self.logger.info(f"Loaded {len(self.used_messages[guild_id])} used messages for guild {guild_id}")

                # Load Tomato Counters
                self.logger.info(f"Loading tomato_counters.json for guild {guild_id}")
                with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "tomato_counters.json", "r") as f:
                    # JSON keys are strings, convert back to int
                    raw_tomato = json.load(f)
                    self.tomato_counters[guild_id] = {int(uid): count for uid, count in raw_tomato.items()}
                self.logger.info(f"Loaded {len(self.tomato_counters[guild_id])} tomato counters for guild {guild_id}")

                # Load Tomato History
                self.logger.info(f"Loading tomato_history.json for guild {guild_id}")
                with self.bot.config_manager.open_data_store(guild_id, self.cog_id, "tomato_history.json", "r") as f:
                    raw_history = json.load(f)
                    self.tomato_history[guild_id] = [
                        (thrower, attacked, msg_id, ts, bool(innocent)) 
                        for thrower, attacked, msg_id, ts, innocent in raw_history
                    ]
                self.logger.info(f"Loaded {len(self.tomato_history[guild_id])} tomato history for guild {guild_id}")
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
        if guild_id not in self.tomato_history:
            self.tomato_history[guild_id] = []

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

        self.logger.info(f"Jail reaction count for message {message.id} in guild {payload.guild_id}: {jail_reaction_count}")

        if jail_reaction_count >= config.to_jail_threshold:
            # Target the message author
            self.logger.info(f"Jail reaction count for message {message.id} in guild {payload.guild_id} is >= {config.to_jail_threshold}. Jailing user {message.author.id}.")
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

        is_jailed = self.get_is_jailed(payload.guild_id, target_user_id)

        if payload.guild_id not in self.tomato_history:
            self.tomato_history[payload.guild_id] = []

        self.tomato_history[payload.guild_id].append((
            reacting_user_id,
            target_user_id,
            payload.message_id,
            int(time.time() * 1000),
            not is_jailed  # is_innocent
        ))

        # 1. If target is in jail -> Increment counter
        if is_jailed:
            self.logger.info(f"Tomato thrown at jailed user {target_user_id} in guild {payload.guild_id}")
            if payload.guild_id not in self.tomato_counters:
                self.tomato_counters[payload.guild_id] = {}
            
            current_count = self.tomato_counters[payload.guild_id].get(target_user_id, 0)
            self.tomato_counters[payload.guild_id][target_user_id] = current_count + 1
            self.dump_data()
        # 2. If target is NOT in jail -> Warn the thrower
        else:
            self.logger.info(f"Tomato thrown at innocent user {target_user_id} in guild {payload.guild_id}")
            self.dump_data()
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

    @tasks.loop(seconds=10)
    async def check_jail_timers(self):
        """
        Periodically checks if users should be released from jail.
        Also checks for users who have the role but are not in the DB (desync).
        """
        self.logger.info("Checking jail timers and clerical errors...")
        current_time_ms = int(time.time() * 1000)

        # Iterate over known guilds to clean up DB entries
        guild_ids_in_db = list(self.currently_jailed_data.keys())

        # 1. Standard Release Logic (Existing DB entries)
        for guild_id in guild_ids_in_db:
            config = self.get_config(guild_id)
            if not config:
                continue

            guild = self.bot.get_guild(guild_id)
            if not guild:
                continue

            # Identify users to release based on time
            users_to_release = []
            for user_id, data in self.currently_jailed_data[guild_id].items():
                if current_time_ms >= data.end_time:
                    users_to_release.append(user_id)

            # Release them
            for user_id in users_to_release:
                member = guild.get_member(user_id)
                if not member:
                    try:
                        member = await guild.fetch_member(user_id)
                    except discord.NotFound:
                        pass 
                
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
                        channel = guild.get_channel(self.currently_jailed_data[guild_id][user_id].channel_id)
                        if channel:
                            await channel.send(script.format(jailed_user=member.mention))

                # Update Data
                await self.data_set_user_free(guild_id, user_id)

        # 2. Desynchronization Check (Users with role but NO DB entry)
        # Iterate over all guilds the bot is in, not just ones in DB
        for guild in self.bot.guilds:
            config = self.get_config(guild.id)
            if not config:
                continue

            role = guild.get_role(config.jailed_role_id)
            if not role:
                continue

            # We need a channel to announce clerical errors in.
            server_config = self.bot.config_manager.get_server_config(guild.id)
            announce_channel = guild.get_channel(server_config["bot_channel_id"])
            if not announce_channel:
                self.logger.warning(f"Could not find bot channel for guild {guild.id}. Not checking for clerical errors.")
                continue

            # Check all members with the jail role
            self.logger.info(f"Checking jail ({config.jailed_role_id}) status for {len(role.members)} members in guild {guild.id}.")
            for member in role.members:
                self.logger.info(f"Checking jail status for user {member.id} in guild {guild.id}.")
                # If they have the role, but `get_is_jailed` returns False, they are desynced
                if not self.get_is_jailed(guild.id, member.id):
                    self.logger.info(f"User {member.id} found with jail role but no DB entry in {guild.id}.")
                    
                    # 50% Chance
                    if random.random() < 0.5:
                        # === OPTION A: Set them free ===
                        try:
                            await member.remove_roles(role, reason="Clerical error: User was not in jail DB.")
                            if config.scot_free_scripts and announce_channel:
                                script = random.choice(config.scot_free_scripts)
                                await announce_channel.send(script.format(jailed_user=member.mention))
                        except discord.Forbidden:
                            self.logger.error(f"Cannot remove role from {member.id} (Permission Error)")

                    else:
                        # === OPTION B: Re-jail them ===
                        # We don't have an offending message ID or channel, so we use placeholders (0 and announce_channel)
                        target_channel_id = announce_channel.id if announce_channel else 0
                        
                        # Log them into the DB (This sets start_time to NOW, so they restart their sentence)
                        success = self.data_set_user_jailed(guild.id, member.id, target_channel_id, 0)
                        
                        if success and config.forgotten_prisoner_scripts and announce_channel:
                            script = random.choice(config.forgotten_prisoner_scripts)
                            await announce_channel.send(script.format(jailed_user=member.mention))

    @check_jail_timers.before_loop
    async def before_check_jail_timers(self):
        await self.bot.wait_until_ready()
