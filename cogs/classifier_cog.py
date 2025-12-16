import discord
import logging
from discord.ext import commands
import io
import random
from PIL import Image
import numpy as np
from classifier.dino_classifier import DinoFeatureExtractor, Classifier
from enum import Enum
import os
import torch
from pathlib import Path
from typing import Optional
import datetime
from collections import OrderedDict
import hashlib

MODEL_DIR = Path(__file__).parent.parent / "classifier" / "models"

class ClassifierCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config_manager = self.bot.config_manager

        self.logger = logging.getLogger("ClassifierCog")
        self.logger.info("Initializing ClassifierCog...")
        torch.set_num_threads(self.config_manager.get_torch_cpu_threads())
        self.logger.info(f"Setting torch cpu threads to {self.config_manager.get_torch_cpu_threads()}")
        
        self.feature_extractor = DinoFeatureExtractor(hf_token=os.getenv("HF_TOKEN"), device=self.config_manager.get_model_device())
        self.server_configs = { server_id: self.config_manager.get_classifier_config(server_id) for server_id in self.config_manager.get_all_server_ids() }
        
        # Load all unique models defined in config
        self._load_classifiers()

        self.embedding_cache = OrderedDict()
        self.max_cache_size = 100

        self.processed_messages_table_name = "classifier_v2_processed_messages"
        self.script_last_used_time_table_name = "classifier_v2_script_last_used_time"
        with self.bot.db.connect() as con:
            con.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.processed_messages_table_name} (
                    message_id INTEGER NOT NULL,
                    classifier_name TEXT NOT NULL,
                    guild_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (message_id, classifier_name)
                )
            ''')

            con.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.script_last_used_time_table_name} (
                    script_text TEXT NOT NULL,
                    guild_id INTEGER NOT NULL,
                    classifier_name TEXT NOT NULL,
                    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (script_text, guild_id, classifier_name)
                )
            ''')

    def _get_last_used_time(self, script_text: str, guild_id: int, classifier_name: str) -> Optional[datetime.datetime]:
        with self.bot.db.connect() as con:
            cur = con.execute(f"""
                SELECT last_used_at FROM {self.script_last_used_time_table_name} WHERE script_text = ? AND guild_id = ? AND classifier_name = ?
            """, (script_text, guild_id, classifier_name))
            row = cur.fetchone()
            if row:
                return row[0]
            else:
                return None

    def _mark_script_used(self, script_text: str, guild_id: int, classifier_name: str):
        """
        If the message has never been used, insert a new row.
        If it has been used update to the current time
        """
        with self.bot.db.connect() as con:
            con.execute(f"""
                INSERT OR IGNORE INTO {self.script_last_used_time_table_name} (
                    script_text, guild_id, classifier_name, last_used_at
                ) VALUES (?, ?, ?, ?)
            """, (script_text, guild_id, classifier_name, datetime.datetime.now()))
            con.execute(f"""
                UPDATE {self.script_last_used_time_table_name} SET last_used_at = ? WHERE script_text = ? AND guild_id = ? AND classifier_name = ?
            """, (datetime.datetime.now(), script_text, guild_id, classifier_name))

    def _load_classifiers(self):
        self.classifiers = {}  # Maps from model path to classifier
        models_to_load = set()
        for server_config in self.server_configs.values():
            for model in server_config.get("models", {}).values():
                models_to_load.add(model["model_path"])

        for model_path in models_to_load:
            try:
                self.logger.info(f"Loading classifier from {model_path}...")
                classifier = Classifier()
                full_path = MODEL_DIR / model_path
                classifier.load(full_path)
                self.classifiers[model_path] = classifier
                self.logger.info(f"Successfully loaded classifier: {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load classifier from {model_path}: {e}")
    
    def _is_message_processed(self, message_id: int, classifier_name: str) -> bool:
        with self.bot.db.connect() as con:
            cur = con.cursor()
            cur.execute(f"""
                SELECT 1 FROM {self.processed_messages_table_name} WHERE message_id = ? AND classifier_name = ?
            """, (message_id, classifier_name))
            row = cur.fetchone()
            return row is not None

    def _mark_message_processed(self, message_id: int, classifier_name: str, guild_id: int, channel_id: int):
        with self.bot.db.connect() as con:
            con.execute(f"""
                INSERT OR IGNORE INTO {self.processed_messages_table_name} (
                    message_id, classifier_name, guild_id, channel_id
                ) VALUES (?, ?, ?, ?)
            """, (message_id, classifier_name, guild_id, channel_id))

    def _get_server_config(self, guild_id: int) -> Optional[dict]:
        return self.server_configs.get(guild_id)

    async def _get_classifier_probs(self, message: discord.Message, classifier_path: str, classifier_name: str):
        classifier = self.classifiers[classifier_path]
        if not message.attachments:
            return None

        indiv_probs = []
        for attachment in message.attachments:
            # Check if attachment is an image
            if not attachment.content_type or not attachment.content_type.startswith('image/'):
                continue

            try:
                # Download image
                image_bytes = await attachment.read()

                # Get image hash for caching
                img_hash = hashlib.md5(image_bytes).hexdigest()

                if img_hash in self.embedding_cache:
                    embeddings = self.embedding_cache[img_hash]
                    self.embedding_cache.move_to_end(img_hash)
                    self.logger.info(f"Using cached embedding for image {img_hash[:8]}")
                else:
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    # Extract features
                    embeddings = self.feature_extractor.extract_features(image)
                    
                    # Update cache
                    self.embedding_cache[img_hash] = embeddings
                    if len(self.embedding_cache) > self.max_cache_size:
                        self.embedding_cache.popitem(last=False)

                # Predict
                probs = classifier.predict_proba(embeddings)[0]

                indiv_probs.append(probs)
            except Exception as e:
                self.logger.error(f"Failed to process attachment: {e}")
                continue

        if not indiv_probs:
            return None

        # Combine probabilities
        combined_probs = np.prod(indiv_probs, axis=0)
        normed_probs = combined_probs / np.sum(combined_probs)

        # Now we can make our predictions
        sorted_indices = np.argsort(normed_probs)[::-1]
        sorted_probs = normed_probs[sorted_indices]
        top_label = classifier.index_to_label[sorted_indices[0]]

        string_names = {
            f"{classifier_name}_label_{i+1}": classifier.index_to_label[label_index] for i, label_index in enumerate(sorted_indices)
        }

        label_probs = {
            classifier.index_to_label[i]: normed_probs[i] for i in sorted_indices
        }

        return string_names, label_probs, sorted_probs, top_label

    def _get_confidence_threshold(self, classifier_config: dict, max_prob: float) -> str:
        confidence_thresholds = classifier_config.get("confidence_thresholds", {})
        sorted_thresholds = sorted(confidence_thresholds.items(), key=lambda x: x[1], reverse=True)
        for label, threshold in sorted_thresholds:
            if max_prob >= threshold:
                return label
        return None

    async def _run_classifier(self, message: discord.Message, classifier_name: str, classifier_config: dict):
        if not message.attachments:
            return

        guild_id = message.guild.id
        model_path = classifier_config["model_path"]
        
        # Check if message was already processed by this classfier
        if self._is_message_processed(message.id, classifier_name):
            self.logger.info(f"Message {message.id} already processed by {classifier_name}.")
            return

        if model_path not in self.classifiers:
            self.logger.error(f"Model {model_path} not loaded.")
            return

        scripts = classifier_config["scripts"]
        classifier = self.classifiers[model_path]
        
        string_names, label_probs, sorted_probs, top_label = await self._get_classifier_probs(message, model_path, classifier_name)
        self.logger.info(f"Label Probabilities: {label_probs}")
        self.logger.info(f"String Names: {string_names}")
        self.logger.info(f"Top Label: {top_label}")
        confidence_threshold = self._get_confidence_threshold(classifier_config, sorted_probs[0])
        if not confidence_threshold:
            self.logger.error(f"No confidence threshold found for {model_path}.")
            return

        candidate_messages = {}  # Map from priority to list of messages
        for script in scripts:
            script_label = script["label"]
            script_confidence = script["confidence"]
            script_text = script["text"]
            script_priority = script["priority"]

            meets_requirements = True
            if (script_label != "ANY") and (script_label != top_label):
                meets_requirements = False
            if (script_confidence != confidence_threshold):
                meets_requirements = False
            if not meets_requirements:
                continue

            if script_priority not in candidate_messages:
                candidate_messages[script_priority] = []
            candidate_messages[script_priority].append(script_text)
        
        if not candidate_messages:
            self.logger.error(f"No candidate messages found for {classifier_name}.")
            return

        top_priority = max(candidate_messages.keys())
        top_messages = candidate_messages[top_priority]
        select_random_prob = classifier_config.get("select_random_prob", 0.0)
        selected_script = None

        if random.random() < select_random_prob:
            selected_script = random.choice(top_messages)
        else:
            # Sort by last used time (None = never used = oldest)
            script_times = []
            for script in top_messages:
                t = self._get_last_used_time(script, guild_id, classifier_name)
                script_times.append((script, t))
            
            # Sort with None being considered "smallest" (oldest)
            # Assuming timestamps are strings, we can use empty string for None
            script_times.sort(key=lambda x: x[1] if x[1] is not None else "")
            selected_script = script_times[0][0]

        try:
            formatted_message = selected_script.format(**string_names)
            # await message.channel.send(formatted_message)
            await message.reply(formatted_message)
            self._mark_script_used(selected_script, guild_id, classifier_name)
            self._mark_message_processed(message.id, classifier_name, guild_id, message.channel.id)
            self.logger.info(f"Replied to message {message.id} with script: {selected_script}")
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author == self.bot.user:
            return

        if not message.guild:
            return

        server_config = self.bot.config_manager.get_server_config(message.guild.id)
        if not server_config:
            return
        if not server_config["active"]:
            return

        if message.guild.id not in self.server_configs:
            return
        server_classifiers_config = self.server_configs[message.guild.id]

        for classifier_config_name, classifier_config in server_classifiers_config["models"].items():
            auto_respond_channels = classifier_config.get("auto_respond_channels", [])
            if message.channel.id not in auto_respond_channels:
                continue
            await self._run_classifier(message, classifier_config_name, classifier_config)

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """
        Handles reactions added to messages.
        If the reaction matches the configured manual trigger, it processes the message
        regardless of the channel it's in.
        """
        if payload.user_id == self.bot.user.id:
            return

        if not payload.guild_id:
            return

        server_config = self.bot.config_manager.get_server_config(payload.guild_id)
        if not server_config:
            return
        if not server_config["active"]:
            return

        if payload.guild_id not in self.server_configs:
            return
        server_classifiers_config = self.server_configs[payload.guild_id]

        for classifier_config_name, classifier_config in server_classifiers_config["models"].items():
            manual_trigger = classifier_config.get("manual_trigger_reaction", None)
            if manual_trigger and str(payload.emoji) == manual_trigger:
                channel = self.bot.get_channel(payload.channel_id)
                if not channel:
                    continue
                message = await channel.fetch_message(payload.message_id)
                await self._run_classifier(message, classifier_config_name, classifier_config)

        