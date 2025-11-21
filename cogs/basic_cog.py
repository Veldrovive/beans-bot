from discord.ext import commands

class BasicCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.command(
        name='ping',
        aliases=['p'],  # Users can type !ping or !p
        help="Responds with the bot's latency." # This text appears in the !help command
    )
    async def ping(self, ctx: commands.Context):
        """
        Usage: !ping
        A simple command to check if the bot is responsive.
        The docstring is also used by the default help command for more detail.
        """
        # self.latency is a built-in attribute of the bot client
        latency_ms = round(self.bot.latency * 1000)
        await ctx.send(f'Pong! Latency: {latency_ms}ms')