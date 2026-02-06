"""
The council of teds serves the mother of bean.

? Respond to message with !council_vote
You can add an parameter to put your argument in
Bot responds with a message with the offeding statement, the argument, and instructions.

A person with the mother of bean role voting immediately decides the vote.

Other people can react thumbs up or thumbs down, but that is not binding. We just keep the
message where we see the results up to date.
"""

"""
{
    channel_id: int,
    message_id: int,
    guild_id: int,
    start_time: int,
    end_time: Optional[int],
    state: str,
    vote_on: str, # The thing being voted on
    vote_info: str # The additional info provided by the user that instigated the vote

}
"""

from discord.ext import commands
from typing import Optional, List, Tuple
import discord
import asyncio
import logging
from dataclasses import dataclass

class VoteData(dataclass):
    yay_council_members: list[str]
    nay_council_members: list[str]
    yay_mother_of_bean: list[str]
    nay_mother_of_bean: list[str]
    yay_others: list[str]
    nay_others: list[str]

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

        self.votes_json = {}

    # async def _get_council_reactions(self, )

    def _start_vote(self, instagating_message_id: int, argument: Optional[str] = None, target_message_id: Optional[int] = None) -> int:
        """
        instagating_message_id: The message that starts with !vote
        argument: The argument to the !vote command
        target_message_id: If the !vote command is in a replay to another message, that message is the target message

        Sends a message in reply to the instagating message with the thing being voted on and the instructions.
        """
        # Step 1: Construct the initial vote message
        # Step 2: Send the message
        # Step 3: Get the message id
        # Step 4: Add the vote data to the json
        # Step 5: Return the message id

    def _get_votes_message(self, votes: VoteData):
        """
        States:
        1. No result - Has not reached majority and mother of bean has not voted
        2. Tie - All of council has voted and it is a tie
        3. Mother of bean tiebreaker - Mother of bean has voted and council has tied
        4. Mother of bean decision - Mother of bean has voted and council has not reached majority for one side
        5. Council majority - Majority of council has voted for one side
        6. Council approval - Majority of council has voted for one side and mother of bean has voted for the same side
        7. Council veto - Majority of council has voted for one side and mother of bean has voted for the opposite side
        8. Mother of Bean overrule - Mother of bean has voted in disagreement with the council, but council unanimously voted for the same side causing a veto

        Additional detail:
        If the "others" have voted in opposition of the current decision (accounting for council and mother of bean), then we note that the common peoples are being oppressed.

        Miscreants reacting with :fire: cause the vote to be thrown out cause it caught on fire. They get sent to jail.
        """
        total_council_members = 4  # TODO Get from config
        
    
    def _get_votes(self, server_id: int,channel_id: int, message_id: int) -> VoteData:
        """
        Segment the individuals who reacted to the message into the appropriate categories with priority
        Mother of bean > council > miscreant > others

        For each category, decide for each user whether they voted yay, nay, or if none of the above.
        If voted both yay and nay, then they are counted as nay.
        """
        mother_of_bean_users = set()
        council_users = set()
        miscreant_users = set()
        other_users = set()

        # Get the reactions
        guild = self.bot.get_guild(server_id)
        channel = guild.get_channel(channel_id)
        message = channel.get_message(message_id)

        for reaction in message.reactions:
            async for user in reaction.users():
                if user.id == self.bot.user.id:
                    continue
                
                user_type = "other"
                


        

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        if not self.active:
            self.logger.debug("Council of Teds is not active")
            return
        
        # Check if we are in the right server and get the config
        # TODO

        # Check if the message is a council vote


        # Get the votes
        votes = self._get_votes(payload.message_id)

        # Get the response message
        response_message = self._get_votes_message(**votes)

        # The full response is a 

        pass

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not self.active:
            self.logger.debug("Council of Teds is not active")
            return
        
        pass