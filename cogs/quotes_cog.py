"""
Quote Guess Who:
1. A quote is posted with the name in spoiler tags
2. A user reacts with a specific emoji
3. We send a DM to that person asking them who they think the quote is from
4. We respond to them with whether they were right or wrong.
5. We visually show somehow how many people have gotten it right versus wrong.
We also store all quotes in the database along with people's guesses.

CLIP Quote Similarity:
1. Anybody posts any message
2. We use CLIP to find the quote that has highest similarity
3. We reply forward the message with the quote

Quote Chain:
Produce markov model for individual's quotes.
"""