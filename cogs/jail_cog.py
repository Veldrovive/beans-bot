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
from dataclasses import dataclass
from typing import Optional
import time
import random
from peewee import *

db_proxy = Proxy()

class JailedUser(Model):
    server_id = IntegerField()
    user_id = IntegerField()
    channel_id = IntegerField()
    offending_message_id = IntegerField()
    start_time = IntegerField()
    end_time = IntegerField()
    has_been_humiliated = BooleanField(default=False)

    class Meta:
        database = db_proxy

class HistoricalJailedUser(Model):
    server_id = IntegerField()
    user_id = IntegerField()
    channel_id = IntegerField()
    offending_message_id = IntegerField()
    start_time = IntegerField()
    end_time = IntegerField()
    has_been_humiliated = BooleanField(default=False)

    class Meta:
        database = db_proxy

class TomatoCounter(Model):
    server_id = IntegerField()
    user_id = IntegerField()
    count = IntegerField(default=0)

    class Meta:
        database = db_proxy
        primary_key = CompositeKey('server_id', 'user_id')

class TomatoHistory(Model):
    server_id = IntegerField()
    thrower_user_id = IntegerField()
    attacked_user_id = IntegerField()
    message_id = IntegerField()
    timestamp = IntegerField()
    is_innocent = BooleanField()

    class Meta:
        database = db_proxy

class UsedMessage(Model):
    server_id = IntegerField()
    message_id = IntegerField()

    class Meta:
        database = db_proxy


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

        self.db = self.bot.config_manager.open_peewee_store("jail_cog.db")
        db_proxy.initialize(self.db)
        self.db.connect()
        self.db.create_tables([JailedUser, HistoricalJailedUser, TomatoCounter, TomatoHistory, UsedMessage])

        # Start the background task to check for jail releases
        self.check_jail_timers.start()
        self.logger.info("JailCog initialized")

    async def cog_unload(self):
        self.check_jail_timers.cancel()

    def get_config(self, guild_id: int) -> Optional[JailCogConfig]:
        # This assumes config_manager is attached to bot and returns a dict
        if not hasattr(self.bot, "config_manager"):
            self.logger.error("Bot does not have config_manager")
            return None
            
        server_config = self.bot.config_manager.get_jail_config(guild_id)
        if server_config is None:
            # self.logger.error(f"JailCogConfig not found for guild {guild_id}")
            return None
        return JailCogConfig(**server_config)

    async def data_set_user_free(self, guild_id: int, user_id: int) -> bool:
        """
        Updates the data stores to mark a user as free. 
        """
        config = self.get_config(guild_id)
        if config is None:
            return False

        try:
            jailed_user = JailedUser.get((JailedUser.server_id == guild_id) & (JailedUser.user_id == user_id))
        except JailedUser.DoesNotExist:
            self.logger.warning(f"User {user_id} is not jailed in guild {guild_id}")
            return False

        # Move data to historical
        HistoricalJailedUser.create(
            server_id=guild_id,
            user_id=jailed_user.user_id,
            channel_id=jailed_user.channel_id,
            offending_message_id=jailed_user.offending_message_id,
            start_time=jailed_user.start_time,
            end_time=jailed_user.end_time,
            has_been_humiliated=jailed_user.has_been_humiliated
        )
        
        # Remove from current
        jailed_user.delete_instance()
        return True

    def data_set_user_jailed(self, guild_id: int, user_id: int, channel_id: int, offending_message_id: int) -> bool:
        """
        Updates the data stores to mark a user as jailed.
        """
        config = self.get_config(guild_id)
        if config is None:
            return False

        if JailedUser.select().where((JailedUser.server_id == guild_id) & (JailedUser.user_id == user_id)).exists():
            return False

        JailedUser.create(
            server_id=guild_id,
            user_id=user_id,
            channel_id=channel_id,
            offending_message_id=offending_message_id,
            start_time=int(time.time() * 1000),
            end_time=int(time.time() * 1000) + config.jail_length_ms,
            has_been_humiliated=False
        )
        
        if not UsedMessage.select().where((UsedMessage.server_id == guild_id) & (UsedMessage.message_id == offending_message_id)).exists():
            UsedMessage.create(server_id=guild_id, message_id=offending_message_id)

        return True

    def get_is_jailed(self, guild_id: int, user_id: int) -> bool:
        return JailedUser.select().where((JailedUser.server_id == guild_id) & (JailedUser.user_id == user_id)).exists()

    async def try_jail_user(self, guild_id: int, user_id: int, offending_message: discord.Message) -> bool:
        """
        Called when a user should be jailed.
        """
        config = self.get_config(guild_id)
        if config is None:
            return False

        # 1. Check if message already used
        if UsedMessage.select().where((UsedMessage.server_id == guild_id) & (UsedMessage.message_id == offending_message.id)).exists():
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

        TomatoHistory.create(
            server_id=payload.guild_id,
            thrower_user_id=reacting_user_id,
            attacked_user_id=target_user_id,
            message_id=payload.message_id,
            timestamp=int(time.time() * 1000),
            is_innocent=not is_jailed
        )

        # 1. Increment counter
        self.logger.info(f"Tomato thrown at user {target_user_id} in guild {payload.guild_id}")
        counter, created = TomatoCounter.get_or_create(
            server_id=payload.guild_id, 
            user_id=target_user_id, 
            defaults={'count': 0}
        )
        counter.count += 1
        counter.save()
            
        # 2. If target is NOT in jail -> Warn the thrower
        if not is_jailed:
            self.logger.info(f"Tomato thrown at innocent user {target_user_id} in guild {payload.guild_id}")
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
        try:
            jail_data = JailedUser.get((JailedUser.server_id == message.guild.id) & (JailedUser.user_id == message.author.id))
        except JailedUser.DoesNotExist:
            return

        if not jail_data.has_been_humiliated:
            self.logger.info(f"User {message.author.id} is in jail and has not been humiliated. Humiliating...")
            jail_data.has_been_humiliated = True
            jail_data.save()
            
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


    @tasks.loop(seconds=60)
    async def check_jail_timers(self):
        """
        Periodically checks if users should be released from jail.
        Also checks for users who have the role but are not in the DB (desync).
        """
        current_time_ms = int(time.time() * 1000)

        # Iterate over known guilds to clean up DB entries
        guild_ids_in_db = [g.server_id for g in JailedUser.select(JailedUser.server_id).distinct()]

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
            for jailed_user in JailedUser.select().where(JailedUser.server_id == guild_id):
                if current_time_ms >= jailed_user.end_time:
                    users_to_release.append(jailed_user.user_id)

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
                        server_config = self.bot.config_manager.get_server_config(guild.id)
                        announce_channel = guild.get_channel(server_config["bot_channel_id"])
                        if announce_channel:
                            await announce_channel.send(script.format(jailed_user=member.mention))

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
            if not server_config or "bot_channel_id" not in server_config:
                continue
            
            announce_channel = guild.get_channel(server_config["bot_channel_id"])
            if not announce_channel:
                continue

            # Check all members with the jail role
            for member in role.members:
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

    @commands.group(name="jail", invoke_without_command=True)
    @commands.guild_only()
    async def jail_group(self, ctx: commands.Context):
        """Commands to see jail stats."""
        config_manager = getattr(self.bot, "config_manager", None)
        if not config_manager or not ctx.guild:
            return
        bot_channel_id = config_manager.get_bot_channel_id(ctx.guild.id)
        if bot_channel_id and ctx.channel.id != bot_channel_id:
            return
        await ctx.send_help(ctx.command)

    @jail_group.command(name="current", help="Shows who is currently in jail.")
    @commands.guild_only()
    async def jail_current(self, ctx: commands.Context):
        config_manager = getattr(self.bot, "config_manager", None)
        if not config_manager or not ctx.guild:
            return
        bot_channel_id = config_manager.get_bot_channel_id(ctx.guild.id)
        if bot_channel_id and ctx.channel.id != bot_channel_id:
            return

        jailed_users = list(JailedUser.select().where(JailedUser.server_id == ctx.guild.id))
        if not jailed_users:
            await ctx.send("No one is currently in jail!")
            return

        mentions = [f"<@{user.user_id}>" for user in jailed_users]
        await ctx.send(f"Currently in jail: {', '.join(mentions)}")

    @jail_group.command(name="info", help="Shows stats about a specific person's jail time.")
    @commands.guild_only()
    async def jail_info(self, ctx: commands.Context, *, member_query: str):
        config_manager = getattr(self.bot, "config_manager", None)
        if not config_manager or not ctx.guild:
            return
        bot_channel_id = config_manager.get_bot_channel_id(ctx.guild.id)
        if bot_channel_id and ctx.channel.id != bot_channel_id:
            return

        try:
            member = await commands.MemberConverter().convert(ctx, member_query)
        except commands.MemberNotFound:
            await ctx.send(f"Could not find anyone named '{member_query}'.")
            return


        current_jail = JailedUser.select().where((JailedUser.server_id == ctx.guild.id) & (JailedUser.user_id == member.id)).count()
        historical_jail = HistoricalJailedUser.select().where((HistoricalJailedUser.server_id == ctx.guild.id) & (HistoricalJailedUser.user_id == member.id)).count()
        total_jail_time = current_jail + historical_jail

        tomato_counter = TomatoCounter.get_or_none((TomatoCounter.server_id == ctx.guild.id) & (TomatoCounter.user_id == member.id))
        tomatoes_thrown = tomato_counter.count if tomato_counter else 0

        message = f"**Jail Stats for {member.display_name}**\n"
        message += f"- Times in jail: {total_jail_time}\n"
        message += f"- Tomatoes thrown at them: {tomatoes_thrown}"

        await ctx.send(message)

