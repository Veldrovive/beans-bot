from discord.ext import commands
import logging
from typing import TYPE_CHECKING
import re
import discord
from discord.ext import tasks
import datetime
import zoneinfo
import random
from peewee import *

import cogs.birthday.scrapers as scrapers

if TYPE_CHECKING:
    from main import Bot
    from config import ConfigManager

tz = zoneinfo.ZoneInfo("America/Detroit")
scheduled_time = datetime.time(hour=10, tzinfo=tz)

db_proxy = Proxy()

class SentActionHoliday(Model):
    server_id = IntegerField()
    sent_date = DateField()
    holiday_name = CharField()
    holiday_link = CharField()
    year_used = IntegerField()

    class Meta:
        database = db_proxy

class ActionHolidayCog(commands.Cog):
    bot: "Bot"
    config_manager: "ConfigManager"

    def __init__(self, bot: commands.Bot):
        self.cog_id = "ActionHolidayCog"
        self.logger = logging.getLogger(self.cog_id)

        self.bot = bot
        self.config_manager = self.bot.config_manager

        self.db = self.config_manager.open_peewee_store("action_holiday_cog.db")
        db_proxy.initialize(self.db)
        self.db.connect()
        self.db.create_tables([SentActionHoliday], safe=True)

        self.check_and_send_holiday_task.start()
        self.logger.info("Initialized ActionHolidayCog")

    async def cog_unload(self):
        self.check_and_send_holiday_task.cancel()

    async def get_action_holidays(self, day: int, month: int, year: int) -> list[scrapers.Holiday]:
        holidays = await scrapers.get_holidays(day, month, year)
        
        verbs = [
            "draw", "create", "make", "paint", "write", "build", "design", "invent",
            "craft", "color", "tell", "do", "start", "hug", "kiss", "thank", "plant",
            "adopt", "take", "bake", "cook", "read", "sing", "pet", "visit", "learn",
            "earn", "buy", "send", "call", "watch", "play", "give", "find", "catch",
            "wear", "eat", "drink", "walk", "clean", "wash"
        ]
        verb_group = "|".join(verbs)
        pattern = re.compile(rf'\b({verb_group})\s+an?\s+', re.IGNORECASE)

        action_holidays = [h for h in holidays if pattern.search(h.name)]
        return action_holidays

    async def check_and_send_holiday(self, guild_id: int):
        self.logger.info(f"Checking for action holidays to send in guild {guild_id}")
        
        bot_channel_id = self.config_manager.get_bot_channel_id(guild_id)
        if bot_channel_id is None:
            self.logger.error(f"Bot channel ID not found for guild {guild_id}")
            return
            
        bot_channel = self.bot.get_channel(bot_channel_id)
        if bot_channel is None:
            self.logger.error(f"Bot channel {bot_channel_id} not found for guild {guild_id}")
            return
        assert isinstance(bot_channel, discord.TextChannel), "Bot channel must be a text channel."

        now = datetime.datetime.now(tz)
        today = now.day
        this_month = now.month
        this_year = now.year

        # Check if we already sent one today
        already_sent_today = SentActionHoliday.select().where(
            SentActionHoliday.server_id == guild_id,
            SentActionHoliday.sent_date == now.date()
        ).exists()

        if already_sent_today:
            self.logger.info(f"Already sent an action holiday today for guild {guild_id}")
            return

        action_holidays = await self.get_action_holidays(today, this_month, this_year)
        if not action_holidays:
            self.logger.info("No action holidays found for today.")
            return

        # Filter out ones used this year
        used_holidays = SentActionHoliday.select().where(
            SentActionHoliday.server_id == guild_id,
            SentActionHoliday.year_used == this_year
        )
        used_holiday_names = {h.holiday_name for h in used_holidays}
        
        unused_holidays = [h for h in action_holidays if h.name not in used_holiday_names]
        if not unused_holidays:
            self.logger.info("All action holidays for today have already been used this year.")
            return

        chosen_holiday = random.choice(unused_holidays)

        potential_messages = [
            f"Ok everyone! It's **[{chosen_holiday.name}]({chosen_holiday.link})** so get one that.",
            f"Today is **[{chosen_holiday.name}]({chosen_holiday.link})**. Make sure that's done by tonight.",
            f"Hear ye, hear ye! 'Tis **[{chosen_holiday.name}]({chosen_holiday.link})**, verily! Fetch one, if thou wilt!",
        ]
            
        await bot_channel.send(random.choice(potential_messages))
        
        # Record it
        SentActionHoliday.create(
            server_id=guild_id,
            sent_date=now.date(),
            holiday_name=chosen_holiday.name,
            holiday_link=chosen_holiday.link,
            year_used=this_year
        )
        self.logger.info(f"Sent action holiday {chosen_holiday.name} to guild {guild_id}")

    @tasks.loop(time=scheduled_time)
    async def check_and_send_holiday_task(self):
        for guild in self.bot.guilds:
            await self.check_and_send_holiday(guild.id)

    @commands.Cog.listener()
    async def on_ready(self):
        now = datetime.datetime.now(tz)
        
        try:
            action_holidays = await self.get_action_holidays(now.day, now.month, now.year)
            print(f"Test scrapers (action_holidays): Found {len(action_holidays)} action holidays today.")
            if action_holidays:
                reason, action = scrapers.get_best_holiday(action_holidays, self.bot.gemini_harness)
                if action:
                    print(f"Test scrapers (action_holidays): Best action holiday: {action.name} because {reason}")
        except Exception as e:
            print(f"Test scrapers (action_holidays) failed: {e}")

        if now.timetz() >= scheduled_time:
            for guild in self.bot.guilds:
                await self.check_and_send_holiday(guild.id)

    @check_and_send_holiday_task.before_loop
    async def before_check_and_send_holiday_task(self):
        await self.bot.wait_until_ready()
