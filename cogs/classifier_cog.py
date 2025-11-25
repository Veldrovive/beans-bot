import discord
import logging
from discord.ext import commands
import io
import random
from PIL import Image
import numpy as np
from dino_classifier import DinoFeatureExtractor, Classifier
from enum import Enum


HIGH_CONFIDENCE_THRESHOLD = 0.68
MID_CONFIDENCE_THRESHOLD = 0.4
LOW_CONFIDENCE_THRESHOLD = 0.0
class PredictionConfidence(Enum):
    HIGH = 1
    MID = 2
    LOW = 3

ALLOW_NONE_LABEL_FOR_ANYONE = False

class ClassifierCog(commands.Cog):
    # scripts = {
    #     ("andy", PredictionConfidence.HIGH): "\"Oh wacky tobacky this is Andy!\" - Connor",
    #     ("andy", PredictionConfidence.HIGH): "Andy! Go to sleep!",
    #     ("andy", PredictionConfidence.HIGH): "Take your bets if Andy's leaving FRB before it's tomorrow",

    #     # ("bean", HIGH_CONFIDENCE_THRESHOLD): "BEAN!!!",
    #     ("bean", PredictionConfidence.HIGH): "Send bean pics",
    #     ("bean", PredictionConfidence.HIGH): "Hey, when are you doing playdate?",

    #     ("harvey", PredictionConfidence.HIGH): "Harvey! Outside of a couch! It's more likely than you think.",
    #     ("harvey", PredictionConfidence.HIGH): "Oh Harvey, please stop being a menace at night",

    #     ("levi", PredictionConfidence.HIGH): "It's the perfect princess Laverne!",
    #     # ("levi", HIGH_CONFIDENCE_THRESHOLD): "",

    #     ("merlin", PredictionConfidence.HIGH): "Watch your feet Merlin's here",
    #     ("merlin", PredictionConfidence.HIGH): "STINKY!",
    #     ("merlin", PredictionConfidence.HIGH): "Get put in air jail!",

    #     (None, PredictionConfidence.HIGH): "AWWWW it's {name_1}!",
    #     (None, PredictionConfidence.MID): "Could it be! Is it {name_1}?! Or wait is it {name_2}?",
    #     (None, PredictionConfidence.LOW): "Who is this man? {name_1}?"
    # }
    scripts = [
        ("andy", PredictionConfidence.HIGH, "\"Oh wacky tobacky this is Andy!\" - Connor"),
        ("andy", PredictionConfidence.HIGH, "Andy! Go to sleep!"),
        ("andy", PredictionConfidence.HIGH, "Take your bets if Andy's leaving FRB before it's tomorrow"),

        ("bean", PredictionConfidence.HIGH, "Send bean pics"),
        ("bean", PredictionConfidence.HIGH, "Hey, when are you doing playdate?"),

        ("harvey", PredictionConfidence.HIGH, "Harvey! Outside of a couch! It's more likely than you think."),
        ("harvey", PredictionConfidence.HIGH, "Oh Harvey, please stop being a menace at night"),

        ("levi", PredictionConfidence.HIGH, "It's the perfect princess Laverne!"),

        ("merlin", PredictionConfidence.HIGH, "Watch your feet Merlin's here"),
        ("merlin", PredictionConfidence.HIGH, "STINKY!"),
        ("merlin", PredictionConfidence.HIGH, "Get put in air jail!"),

        (None, PredictionConfidence.HIGH, "AWWWW it's {name_1}!"),
        (None, PredictionConfidence.MID, "Could it be! Is it {name_1}?! Or wait is it {name_2}?"),
        (None, PredictionConfidence.LOW, "Who is this man? {name_1}?"),
    ]

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.logger = logging.getLogger("ClassifierCog")
        self.logger.info("Initializing ClassifierCog...")
        self.feature_extractor = DinoFeatureExtractor()
        self.classifier = Classifier()
        try:
            self.classifier.load()
            self.logger.info("Classifier loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load classifier: {e}")

        self.processed_messages_table_name = "classifier_processed_messages"
        with self.bot.db.connect() as con:
            con.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.processed_messages_table_name} (
                    message_id INTEGER PRIMARY KEY,
                    guild_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def is_message_processed(self, message_id: int) -> bool:
        with self.bot.db.connect() as con:
            cur = con.cursor()
            cur.execute(f'''
                SELECT 1 FROM {self.processed_messages_table_name} WHERE message_id = ?
            ''', (message_id,))
            return cur.fetchone() is not None

    def mark_message_processed(self, message_id: int, guild_id: int, channel_id: int):
        with self.bot.db.connect() as con:
            con.execute(f'''
                INSERT OR IGNORE INTO {self.processed_messages_table_name} (
                    message_id, guild_id, channel_id
                ) VALUES (?, ?, ?)
            ''', (message_id, guild_id, channel_id))

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # Ignore messages from the bot itself
        if message.author == self.bot.user:
            return

        # Check if message is in the configured bot channel
        if message.guild:
            server_config = self.bot.config_manager.get_server_config(message.guild.id)
            if not server_config:
                self.logger.warning(f"No server config found for guild {message.guild.id}")
                return
            classifier_config = server_config.get("classifier_conf", {})
            enabled_channels = classifier_config.get("enabled_channels", [])
            if message.channel.id not in enabled_channels:
                return
        else:
            self.logger.debug(f"Message not in guild")
            return

        # Check for attachments
        await self.process_image_message(message)

    async def process_image_message(self, message: discord.Message):
        if not message.attachments:
            return

        # Check if message has already been processed
        if self.is_message_processed(message.id):
            self.logger.info(f"Message {message.id} already processed.")
            return

        # Get the server config
        server_config = self.bot.config_manager.get_server_config(message.guild.id)
        if not server_config:
            self.logger.warning(f"No server config found for guild {message.guild.id}")
            return
        classifier_config = server_config.get("classifier_conf", {})
        allow_none_label_for_anyone = classifier_config.get("allow_none_label_for_anyone", False)

        # Previously we sent one message for each image. Now instead we use an independence assumption to combine the probabilities for each image
        # and proceed as if it was a single recognition.
        indiv_probs = []
        for attachment in message.attachments:
            # Check if attachment is an image
            if not attachment.content_type or not attachment.content_type.startswith('image/'):
                continue

            try:
                # Download image
                image_bytes = await attachment.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Extract features
                embeddings = self.feature_extractor.extract_features(image)

                # Predict
                probs = self.classifier.predict_proba(embeddings)[0]

                indiv_probs.append(probs)
            except Exception as e:
                self.logger.error(f"Failed to process attachment: {e}")
                continue

        # Combine probabilities
        combined_probs = np.prod(indiv_probs, axis=0)
        normed_probs = combined_probs / np.sum(combined_probs)

        # We now have our single prediction
        sorted_indices = np.argsort(normed_probs)[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1] if len(sorted_indices) > 1 else None

        # Extract relevant information about our first and second predictions
        top_pred_label = self.classifier.index_to_label[top_idx]
        second_pred_label = self.classifier.index_to_label[second_idx] if second_idx is not None else None

        top_pred_name = top_pred_label.title()
        second_pred_name = second_pred_label.title() if second_pred_label is not None else "Unknown"

        top_pred_confidence = normed_probs[top_idx]
        
        # In order to find the candidate messages we need to figure out which confidence threshold it falls into
        if top_pred_confidence >= HIGH_CONFIDENCE_THRESHOLD:
            confidence = PredictionConfidence.HIGH
        elif top_pred_confidence >= MID_CONFIDENCE_THRESHOLD:
            confidence = PredictionConfidence.MID
        else:
            confidence = PredictionConfidence.LOW
        
        # Now we can find the candidate messages
        candidate_messages = []
        for name, script_conf, script_message in self.scripts:
            # First we check if it meets the name requirement
            meets_requirements = True
            if (name is None and not allow_none_label_for_anyone) or (name is not None and name.lower() != top_pred_name.lower()):
                meets_requirements = False
            if script_conf != confidence:
                meets_requirements = False
            if not meets_requirements:
                continue
            candidate_messages.append(script_message)

        if not candidate_messages:
            # Then no candidates satisfy both the name and confidence requirements.
            # We back up to only requiring the confidence requirement.
            candidate_messages = [script_message for name, script_conf, script_message in self.scripts if script_conf == confidence]
        self.logger.info(f"Candidate messages: {candidate_messages}")

        if not candidate_messages:
            self.logger.info(f"No candidate messages found for {top_pred_name} with confidence {confidence}")
            return
        
        # Pick a random message from the candidate messages
        selected_script = random.choice(candidate_messages)
        response = selected_script.format(name_1=top_pred_name, name_2=second_pred_name)

        # Send the response
        await message.reply(response)

        # Mark the message as processed
        self.mark_message_processed(message.id, message.guild.id, message.channel.id)

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """
        Handles reactions added to messages.
        If the reaction matches the configured manual trigger, it processes the message
        regardless of the channel it's in.
        """
        if payload.user_id == self.bot.user.id:
            return

        guild_id = payload.guild_id
        if not guild_id:
            return

        server_config = self.bot.config_manager.get_server_config(guild_id)
        if not server_config:
            return

        classifier_config = server_config.get("classifier_conf", {})
        manual_trigger_reaction = classifier_config.get("manual_trigger_reaction", None)

        if not manual_trigger_reaction:
            return

        if str(payload.emoji) != manual_trigger_reaction:
            return

        # Fetch the message
        channel = self.bot.get_channel(payload.channel_id)
        if not channel:
            return

        try:
            message = await channel.fetch_message(payload.message_id)
            await self.process_image_message(message)
        except discord.NotFound:
            self.logger.warning(f"Message {payload.message_id} not found.")
        except discord.Forbidden:
            self.logger.warning(f"Missing permissions to fetch message {payload.message_id}.")
        except Exception as e:
            self.logger.error(f"Error handling manual trigger: {e}")
