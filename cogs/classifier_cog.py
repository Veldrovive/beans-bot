import discord
from discord.ext import commands
import io
import random
from PIL import Image
import numpy as np
from dino_classifier import DinoFeatureExtractor, Classifier


HIGH_CONFIDENCE_THRESHOLD = 0.68
MID_CONFIDENCE_THRESHOLD = 0.4
LOW_CONFIDENCE_THRESHOLD = 0.0

ALLOW_NONE_LABEL_FOR_ANYONE = False

class ClassifierCog(commands.Cog):
    scripts = {
        ("andy", HIGH_CONFIDENCE_THRESHOLD): "\"Oh wacky tobacky this is Andy!\" - Connor",
        ("andy", HIGH_CONFIDENCE_THRESHOLD): "Andy! Go to sleep!",
        ("andy", HIGH_CONFIDENCE_THRESHOLD): "Take your bets if Andy's leaving FRB before it's tomorrow",

        # ("bean", HIGH_CONFIDENCE_THRESHOLD): "BEAN!!!",
        ("bean", HIGH_CONFIDENCE_THRESHOLD): "Send bean pics",
        ("bean", HIGH_CONFIDENCE_THRESHOLD): "Hey, when are you doing playdate?",

        ("harvey", HIGH_CONFIDENCE_THRESHOLD): "Harvey! Outside of a couch! It's more likely than you think.",
        ("harvey", HIGH_CONFIDENCE_THRESHOLD): "Oh Harvey, please stop being a menace at night",

        ("levi", HIGH_CONFIDENCE_THRESHOLD): "It's the perfect princess Laverne!",
        # ("levi", HIGH_CONFIDENCE_THRESHOLD): "",

        ("merlin", HIGH_CONFIDENCE_THRESHOLD): "Watch your feet Merlin's here",
        ("merlin", HIGH_CONFIDENCE_THRESHOLD): "STINKY!",
        ("merlin", HIGH_CONFIDENCE_THRESHOLD): "Get put in air jail!",

        (None, HIGH_CONFIDENCE_THRESHOLD): "AWWWW it's {name_1}!",
        (None, MID_CONFIDENCE_THRESHOLD): "Could it be! Is it {name_1}?! Or wait is it {name_2}?",
        (None, LOW_CONFIDENCE_THRESHOLD): "Who is this man? {name_1}?"
    }

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        print("Initializing ClassifierCog...")
        self.feature_extractor = DinoFeatureExtractor()
        self.classifier = Classifier()
        try:
            self.classifier.load()
            print("Classifier loaded successfully.")
        except Exception as e:
            print(f"Failed to load classifier: {e}")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # Ignore messages from the bot itself
        if message.author == self.bot.user:
            return

        # Check for attachments
        if not message.attachments:
            return

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
                
                # Get top predictions
                sorted_indices = np.argsort(probs)[::-1]
                top_idx = sorted_indices[0]
                second_idx = sorted_indices[1] if len(sorted_indices) > 1 else None
                
                pred_label = self.classifier.index_to_label[top_idx]
                confidence = probs[top_idx]
                
                name_1 = pred_label.title()
                name_2 = self.classifier.index_to_label[second_idx].title() if second_idx is not None else "Unknown"

                # Filter valid scripts based on confidence
                candidates = []
                for (label, threshold), script_template in self.scripts.items():
                    if confidence >= threshold:
                        candidates.append(((label, threshold), script_template))

                if not candidates:
                     # Fallback if no script matches (shouldn't happen with LOW_CONFIDENCE_THRESHOLD=0.0)
                    response = f"I see {name_1} with {confidence:.2%} confidence, but I have nothing to say."
                else:
                    # Find the highest threshold among the candidates
                    max_threshold = max(c[0][1] for c in candidates)
                    
                    # Filter candidates to only those with the highest threshold
                    best_candidates = [c for c in candidates if c[0][1] == max_threshold]
                    
                    # Separate into specific and generic matches
                    specific_matches = []
                    generic_matches = []
                    
                    for (label, _), script_template in best_candidates:
                        if label is not None and label.lower() == pred_label.lower():
                            specific_matches.append(script_template)
                        elif label is None:
                            generic_matches.append(script_template)
                    
                    # Apply selection logic
                    final_options = []
                    if ALLOW_NONE_LABEL_FOR_ANYONE:
                        final_options = specific_matches + generic_matches
                    else:
                        if specific_matches:
                            final_options = specific_matches
                        else:
                            final_options = generic_matches
                    
                    if not final_options:
                         # Should ideally not be reached if candidates existed, unless logic flaw
                         response = f"I see {name_1} ({confidence:.2%}), but I'm confused about what to say."
                    else:
                        selected_script = random.choice(final_options)
                        response = selected_script.format(name_1=name_1, name_2=name_2)

                await message.reply(response)

            except Exception as e:
                print(f"Error processing image: {e}")
                await message.reply(f"Error processing image: {e}")
