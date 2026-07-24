from discord.ext import commands
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
import re
import discord
from discord.ext import tasks
import dateparser
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

@dataclass
class Birthday:
    user_name: str
    month: int
    day: int
    year: int | None

@dataclass
class BirthdayCogConfig:
    # These specify the message that Miranda sent that has the list of birthdays
    birthdays_message_channel_id: int
    birthdays_message_id: int

    # These enable or disable sections of the birthday message
    enable_holidays: bool
    enable_on_this_day: bool
        
db_proxy = Proxy()
class SentBirthdayMessage(Model):
    """
    Represents a message that has already been sent for a specific date.
    """
    sent_date = DateField()
    server_id = IntegerField()
    channel_id = IntegerField()
    message_id = IntegerField()

    user_name = CharField()
    for_year = IntegerField()

    used_holiday = CharField(null=True)
    used_on_this_day = CharField(null=True)

    class Meta:
        database = db_proxy

class UsedHolidays(Model):
    server_id = IntegerField()

    sent_date = DateField()
    holiday_name = CharField()
    holiday_link = CharField()

    day = IntegerField()
    month = IntegerField()

    class Meta:
        database = db_proxy

class UsedOnThisDay(Model):
    server_id = IntegerField()

    sent_date = DateField()
    on_this_day_text = CharField()

    day = IntegerField()
    month = IntegerField()
    year = IntegerField()

    class Meta:
        database = db_proxy

class BirthdayCog(commands.Cog):
    bot: "Bot"
    config_manager: "ConfigManager"

    def __init__(self, bot: commands.Bot):
        self.cog_id = "BirthdayCog"
        self.logger = logging.getLogger(self.cog_id)

        # assert isinstance(bot, Bot), "Bot must be an instance of Bot."
        self.bot = bot

        self.config_manager = self.bot.config_manager

        self.db = self.config_manager.open_peewee_store("birthday_cog.db")
        db_proxy.initialize(self.db)
        self.db.connect()
        self.db.create_tables([SentBirthdayMessage, UsedHolidays, UsedOnThisDay])

        self.check_and_send_birthday_messages_task.start()
        self.logger.info("Initialized BirthdayCog")

    async def cog_unload(self):
        self.check_and_send_birthday_messages_task.cancel()

    def get_cog_config(self, guild_id: int) -> BirthdayCogConfig:
        server_config = self.bot.config_manager.get_birthday_config(guild_id)
        if server_config is None:
            raise ValueError(f"BirthdayCogConfig not found for guild {guild_id}")
        return BirthdayCogConfig(**server_config)

    async def read_birthdays(self, guild_id: int) -> list[Birthday]:
        cog_config = self.get_cog_config(guild_id)

        birthdays_message_channel_id = cog_config.birthdays_message_channel_id
        birthdays_message_id = cog_config.birthdays_message_id
        # Use discord to get the message content
        guild = self.bot.get_guild(guild_id)
        if guild is None:
            raise ValueError(f"Guild {guild_id} not found.")
        channel = guild.get_channel(birthdays_message_channel_id)
        if channel is None:
            raise ValueError(f"Channel {birthdays_message_channel_id} not found.")
        assert isinstance(channel, discord.TextChannel), "Channel must be a text channel."
        message = await channel.fetch_message(birthdays_message_id)
        if message is None:
            raise ValueError(f"Message {birthdays_message_id} not found.")
        
        # The birthdays message is expected to have a format like:
        """
        # The Group's Birthdays 
        ## Old 
        - Austen: October 22nd, 1997
        - Mel: November 30th, 1999
        - Aidan: April 2nd, 2000
        - Sophia: May 3rd, 2000
        - Jace: July 27th, 2000

        ## Normal 
        - Laurie: January 31st, 2001
        - Connor: February 25th, 2001
        - Ted: June 8th, 2001
        - Andy: June 18th, 2001
        - Miranda: July 26th, 2001

        ## Baby 
        - Tate: May 29th, 2003
        - Halina: October 19th, 2004
        - Cody: April 13th, 2003
        - Julianne: July 14th, 2007

        ## Cat 
        - Merlin: July 21st, 2021
        - Wrigley: April 8th, 2025
        - Harvey: unknown
        - Bean: Unknown
        """
        # We will just look for lines that match the regex for a name and a date.

        birthdays: list[Birthday] = []
        for line in message.content.split("\n"):
            # Regex for a name and a date.
            match = re.match(r"- (.*): (.*)", line)
            if match:
                name = match.group(1)
                date_str = match.group(2)
                
                # Parse the date string.
                # It will be in the format "Month Day, Year" or "Month Day"
                # Or "unknown"
                if date_str.lower() == "unknown":
                    # self.logger.warning(f"{name} has an unknown birthday.")
                    continue
                else:
                    parsed_datetime = dateparser.parse(date_str, settings={'REQUIRE_PARTS':['day','month'], "RELATIVE_BASE":datetime.datetime(9999,1,1)})
                    if parsed_datetime is None:
                        self.logger.error(f"Failed to parse birthday for {name}: {date_str}")
                        continue

                    day, month, year = parsed_datetime.day, parsed_datetime.month, parsed_datetime.year
                    if year == 9999:
                        year = None
                    
                    birthdays.append(Birthday(name, month, day, year))

        return birthdays

    def check_if_birthday_already_messaged(self, guild_id: int, birthday: Birthday, for_year: int) -> bool:
        """
        Check if a birthday has already been messaged for a specific date.
        """
        # Search the database for a message that was sent for this year and user
        user_name = birthday.user_name
        
        has_entry = SentBirthdayMessage.select().where(
            SentBirthdayMessage.server_id == guild_id,
            SentBirthdayMessage.user_name == user_name,
            SentBirthdayMessage.for_year == for_year
        ).exists()

        return has_entry

    def log_birthday_messaged(self, guild_id: int, channel_id: int, message_id: int, birthday: Birthday, for_year: int, holiday_used: scrapers.Holiday | None, on_this_day_used: scrapers.OnThisDayEvent | None):
        """
        Log that a birthday has been messaged.
        """
        user_name = birthday.user_name
        SentBirthdayMessage.create(
            sent_date = datetime.datetime.now(),
            server_id = guild_id,
            channel_id = channel_id,
            message_id = message_id,
            user_name = user_name,
            for_year = for_year,

            used_holiday = holiday_used.name if holiday_used is not None else None,
            used_on_this_day = on_this_day_used.description if on_this_day_used is not None else None,
        )

    async def get_unused_holidays(self, server_id: int, day: int, month: int, year: int) -> list[scrapers.Holiday]:
        """
        Gets all holidays on the specified day that are not in the UsedHolidays table.
        """
        holidays = await scrapers.get_holidays(day, month, year)
        single_day_holidays = [h for h in holidays if h.type == scrapers.HolidayType.SINGLE_DAY]

        # Get all holidays that have been used before that fall on this date
        used_holidays = UsedHolidays.select().where(
            UsedHolidays.server_id == server_id,
            UsedHolidays.day == day,
            UsedHolidays.month == month
        )
        used_holiday_names = [h.holiday_name for h in used_holidays]
        
        unused_holidays = [h for h in single_day_holidays if h.name not in used_holiday_names]
        return unused_holidays

    def log_holiday_used(self, server_id: int, holiday: scrapers.Holiday):
        UsedHolidays.create(
            server_id = server_id,

            sent_date = datetime.datetime.now(),
            holiday_name = holiday.name,
            holiday_link = holiday.link,

            day = holiday.day,
            month = holiday.month,
        )
    
    async def get_unused_on_this_day_events(self, server_id: int, day: int, month: int, year: int) -> list[scrapers.OnThisDayEvent]:
        events = await scrapers.get_on_this_day(day, month)

        # Get all events that have been used before that fall on this date
        used_events = UsedOnThisDay.select().where(
            UsedOnThisDay.server_id == server_id,
            UsedOnThisDay.day == day,
            UsedOnThisDay.month == month
        )
        # We remove all events that have the same year as an already used one
        used_event_years = [e.year for e in used_events]
        
        
        unused_events = [e for e in events if e.year not in used_event_years]
        return unused_events
    
    def log_on_this_day_event_used(self, server_id: int, event: scrapers.OnThisDayEvent):
        UsedOnThisDay.create(
            server_id = server_id,

            sent_date = datetime.datetime.now(),
            on_this_day_text = event.description[:200],

            day = event.day,
            month = event.month,
            year = event.year
        )

    async def get_birthday_message(self, server_id: int, birthday: Birthday, current_year: int, enable_holidays: bool, enable_on_this_day: bool) -> tuple[str, scrapers.Holiday | None, scrapers.OnThisDayEvent | None]:
        holiday = None
        if enable_holidays:
            unused_holidays = await self.get_unused_holidays(server_id, birthday.day, birthday.month, current_year)
            if unused_holidays:
                try:
                    reason, holiday = scrapers.get_best_holiday(unused_holidays, self.bot.gemini_harness)
                    if holiday is not None:
                        self.logger.info(f"Gemini chose the holiday {holiday.name} because {reason}")
                    
                    if holiday is None:
                        self.logger.warning("Gemini failed to choose a holiday")
                        holiday = random.choice(unused_holidays)
                        self.logger.info(f"Randomly chose the holiday {holiday.name}")
                except Exception as e:
                    self.logger.error(f"Failed to choose a holiday: {e}")
                    holiday = random.choice(unused_holidays)
                    self.logger.info(f"Randomly chose the holiday {holiday.name}")

        on_this_day = None
        if enable_on_this_day:
            unused_on_this_day_events = await self.get_unused_on_this_day_events(server_id, birthday.day, birthday.month, current_year)
            if unused_on_this_day_events:
                try:
                    reason, on_this_day = scrapers.get_best_on_this_day(unused_on_this_day_events, self.bot.gemini_harness)
                    if on_this_day is not None:
                        self.logger.info(f"Gemini chose the on this day event {on_this_day.description} because {reason}")
                    
                    if on_this_day is None:
                        self.logger.warning("Gemini failed to choose an on this day event")
                        on_this_day = random.choice(unused_on_this_day_events)
                        self.logger.info(f"Randomly chose the on this day event {on_this_day.description}")
                except Exception as e:
                    self.logger.error(f"Failed to choose an on this day event: {e}")
                    on_this_day = random.choice(unused_on_this_day_events)
                    self.logger.info(f"Randomly chose the on this day event {on_this_day.description}")

        message = ""
        if birthday.year is not None:
            age = datetime.datetime.now().year - birthday.year
            message = f"Happy {age}rd birthday, {birthday.user_name}!"
        else:
            message = f"Happy birthday, {birthday.user_name}!"

        if holiday is not None:
            message += f"\n\nIt just so happens that it is also [{holiday.name}]({holiday.link})! Maybe you should celebrate?"

        if on_this_day is not None:
            message += f"\n\nAlso, on this day way back in {on_this_day.year}, {on_this_day.description}\nYou can learn more [here]({on_this_day.link})."
        
        return message, holiday, on_this_day

    async def check_and_send_birthday_messages(self, guild_id: int):
        """
        Main function to send birthday messages to anybody who should receive messages in a guild
        """
        self.logger.info(f"Checking for birthdays in guild {guild_id}")
        try:
            cog_config = self.get_cog_config(guild_id)
        except ValueError:
            self.logger.warning(f"BirthdayCogConfig not found for guild {guild_id}")
            return

        try:
            birthdays = await self.read_birthdays(guild_id)
            self.logger.info(f"Found {len(birthdays)} birthdays. {birthdays}")
        except ValueError:
            self.logger.warning(f"Failed to read birthdays for guild {guild_id}")
            return
        self.logger.debug(f"Found {len(birthdays)} birthdays. {birthdays}")

        # Get the current day, month, year to compare to our birthdays list
        now = datetime.datetime.now(tz)
        today = now.day
        this_year = now.year
        this_month = now.month

        self.logger.info(f"Looking for birthdays with day {today} and month {this_month}")

        # Check if there are any birthdays today
        birthdays_today = [b for b in birthdays if b.day == today and b.month == this_month]
        self.logger.info(f"Found {len(birthdays_today)} birthdays today. {birthdays_today}")
        non_messaged_birthdays = []
        for birthday in birthdays_today:
            self.logger.info(f"{birthday.user_name} has a birthday today. Checking if message already sent")
            already_sent = self.check_if_birthday_already_messaged(guild_id, birthday, this_year)
            if already_sent:
                self.logger.info(f"{birthday.user_name}'s birthday has already been messaged")
                continue
            else:
                self.logger.info(f"{birthday.user_name}'s birthday has not been messaged")
                non_messaged_birthdays.append(birthday)

        if len(non_messaged_birthdays) == 0:
            return

        bot_channel_id = self.config_manager.get_bot_channel_id(guild_id)
        if bot_channel_id is None:
            self.logger.error(f"Bot channel ID not found for guild {guild_id}")
            return
        bot_channel = self.bot.get_channel(bot_channel_id)
        if bot_channel is None:
            self.logger.error(f"Bot channel {bot_channel_id} not found for guild {guild_id}")
            return
        assert isinstance(bot_channel, discord.TextChannel), "Bot channel must be a text channel."
        
        for birthday in non_messaged_birthdays:
            message, holiday, on_this_day = await self.get_birthday_message(guild_id, birthday, this_year, cog_config.enable_holidays, cog_config.enable_on_this_day)
            self.logger.info(f"Sending birthday message for {birthday.user_name}: {message}")
            sent_message = await bot_channel.send(message)
            self.log_birthday_messaged(guild_id, bot_channel_id, sent_message.id, birthday, this_year, holiday, on_this_day)

            if holiday is not None:
                self.log_holiday_used(guild_id, holiday)
            if on_this_day is not None:
                self.log_on_this_day_event_used(guild_id, on_this_day)

            self.logger.info(f"Logged birthday message for {birthday.user_name}")
            

    @tasks.loop(time=scheduled_time)
    async def check_and_send_birthday_messages_task(self):
        for guild in self.bot.guilds:
            await self.check_and_send_birthday_messages(guild.id)

    # We also run this on startup in case the bot was down on a birthday
    @commands.Cog.listener()
    async def on_ready(self):
        print("on_ready")
        now = datetime.datetime.now(tz)
        if now.timetz() >= scheduled_time:
            for guild in self.bot.guilds:
                await self.check_and_send_birthday_messages(guild.id)

    @check_and_send_birthday_messages_task.before_loop
    async def before_check_and_send_birthday_messages_task(self):
        await self.bot.wait_until_ready()