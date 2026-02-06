import re

def parse_discord_quotes(data):
    messages = data.split("<<<<||MESSAGE_DELIMITER||>>>>")
    parsed_results = []
    
    for raw_msg in messages:
        msg = raw_msg.strip()
        if not msg:
            continue
            
        entry = {
            "original": msg,
            "quotes": [],
            "type": "unknown"
        }

        # --- 1. SCRIPT STYLE (Name: Action/Quote) ---
        # Handles "Andy: Begins..." and "Ted: 'Quote'"
        # We iterate to find all occurrences in a single message block
        script_pattern = r"(?:^|\n)([A-Za-z0-9 ]+):\s+(.+?)(?=(?:\n[A-Za-z0-9 ]+:)|$)"
        script_matches = list(re.finditer(script_pattern, msg, re.DOTALL))
        
        # Only use script logic if it matches the start OR multiple are found
        if script_matches and (msg.startswith(script_matches[0].group(1)) or len(script_matches) > 1):
            for m in script_matches:
                author = m.group(1).strip()
                content = m.group(2).strip()
                content = content.strip(' "“”')
                entry["quotes"].append((content, author))
            entry["type"] = "script_style"
            parsed_results.append(entry)
            continue

        # --- 2. FIND ATTRIBUTION SEPARATOR ---
        # Look for the LAST dash followed by an author-like string (no quotes, ends line)
        split_match = list(re.finditer(r"[-–—]\s*(?=[^\"“”\n]+$)", msg))
        
        if split_match:
            last_split = split_match[-1]
            text_part = msg[:last_split.start()].strip()
            author_part = msg[last_split.end():].strip()
            
            # --- 3. NESTED QUOTE CHECK ---
            # Handles: " "Inner Quote" - InnerAuthor " - OuterAuthor
            # Prevents regex from splitting " " and " - InnerAuthor"
            is_nested = False
            if (text_part.startswith('"') and text_part.endswith('"')) or \
               (text_part.startswith('“') and text_part.endswith('”')):
                # Check inner content for attribution pattern
                inner = text_part[1:-1]
                if re.search(r'["”]\s*[-–—]\s*[A-Za-z]+', inner):
                    entry["quotes"].append((text_part, author_part))
                    entry["type"] = "nested_quote"
                    is_nested = True
            
            if not is_nested:
                # --- 4. COMPLEX INTERNAL SPLIT (The "Sword" Case) ---
                # Handles: “Quote A” - “Quote B” - Author A & Author B
                internal_split_match = re.search(r"([“\"].+?[”\"])\s*[-–—]\s*([“\"].+?[”\"])", text_part, re.DOTALL)
                
                if internal_split_match:
                     q1 = internal_split_match.group(1).strip(' "“”')
                     q2 = internal_split_match.group(2).strip(' "“”')
                     
                     # Map authors
                     if "&" in author_part:
                        authors = [a.strip() for a in author_part.split("&")]
                     elif " and " in author_part and "," not in author_part:
                         authors = [a.strip() for a in author_part.split(" and ")]
                     else:
                        authors = [a.strip() for a in author_part.split(",")]
                     
                     if len(authors) == 2:
                         entry["quotes"].append((q1, authors[0]))
                         entry["quotes"].append((q2, authors[1]))
                         entry["type"] = "complex_internal_split"
                     else:
                         entry["quotes"].append((text_part, author_part))

                else:
                    # --- 5. STANDARD / MULTI-MAPPING ---
                    # Handles: "Quote A" "Quote B" - Author A, Author B
                    # Also handles standard single quotes.
                    
                    quote_blocks = re.findall(r'(?:“|")(.+?)(?:”|")', text_part, re.DOTALL)
                    authors = [a.strip() for a in author_part.split(",")]
                    
                    if len(quote_blocks) > 1 and len(authors) == len(quote_blocks):
                        # 1:1 Map
                         for q, a in zip(quote_blocks, authors):
                            # Clean "then" for chronological attributions
                            a = re.sub(r"^then\s+", "", a) 
                            entry["quotes"].append((q, a))
                         entry["type"] = "multi_quote_mapped"
                    else:
                        # Fallback to single quote block
                        clean_text = text_part.strip()
                        if (clean_text.startswith('"') and clean_text.endswith('"')) or \
                           (clean_text.startswith('“') and clean_text.endswith('”')):
                            clean_text = clean_text[1:-1]
                        entry["quotes"].append((clean_text, author_part))
                        entry["type"] = "standard_single"
            
            parsed_results.append(entry)
        
        # --- 6. UNATTRIBUTED / NOISE ---
        else:
             if any(c in msg for c in ['"', '“', '”']):
                 content = msg.strip(' "“”')
                 # Filter known noise phrases
                 if len(content) < 30 and content.lower() in ["i love it", "oh my god", "quotes jeopardy"]:
                     entry["type"] = "noise"
                 else:
                     entry["quotes"].append((content, "Unknown/Unattributed"))
                     entry["type"] = "unattributed_quote"
             else:
                 entry["type"] = "noise"
             parsed_results.append(entry)

    return parsed_results


if __name__ == "__main__":
    test_text = """Andy: Begins to take off his shirt 

Ted: “This action was not endorsed by QSAR”
<<<<||MESSAGE_DELIMITER||>>>>
“does andy die for your sins?” -aidan, distressed
<<<<||MESSAGE_DELIMITER||>>>>
“i don’t know anything about spheres” - austen
<<<<||MESSAGE_DELIMITER||>>>>
"i went through a bath salts phase last year" - connor
<<<<||MESSAGE_DELIMITER||>>>>
"Missiles just don't have enough whimsy" - Austen
<<<<||MESSAGE_DELIMITER||>>>>
"Noooo... Connor.... you're gonna get me pregnant..." - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
" "Aidan's my favorite muppet" - Miranda" - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
“Meth… uh… Trump supporters… lots of those” - Connor
<<<<||MESSAGE_DELIMITER||>>>>
"Are you a homogeneous?" - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
"I wanna have induced psychosis" "You already do" - Aidan, Miranda
<<<<||MESSAGE_DELIMITER||>>>>
"i was born in math hell" - Ted
<<<<||MESSAGE_DELIMITER||>>>>
Okay but this is so real it is dark and with the snow these past few days I'm sometimes driving on vibes alone
<<<<||MESSAGE_DELIMITER||>>>>
“Good news my vision got way worse” - Cody, still driving 3 people
<<<<||MESSAGE_DELIMITER||>>>>
“it’s so hard to see” - Cody, while driving with 3 people
<<<<||MESSAGE_DELIMITER||>>>>
"The horny owl's a virgin?" - Tate
<<<<||MESSAGE_DELIMITER||>>>>
"This is not a gay event"  "Shit" - Connor, Tate
<<<<||MESSAGE_DELIMITER||>>>>
"I'll be like Jesus" - Tate
<<<<||MESSAGE_DELIMITER||>>>>
"We will get you this side table if it kills us." "NO!!" - Miranda, then Sophia and Aidan simultaneously
<<<<||MESSAGE_DELIMITER||>>>>
“Ok ok I’m not saying you’re mutilated or anything” - Laurie
<<<<||MESSAGE_DELIMITER||>>>>
“I’m in yam hell” - Miranda
<<<<||MESSAGE_DELIMITER||>>>>
“Quick! Somebody lick his eyeball!” - Connor
<<<<||MESSAGE_DELIMITER||>>>>
“Jace, can you pass me the nipple?” - Melodie
<<<<||MESSAGE_DELIMITER||>>>>
"My fat cow is on your mushroom" - Ted
<<<<||MESSAGE_DELIMITER||>>>>
“i learned color theory from magic” - austen
<<<<||MESSAGE_DELIMITER||>>>>
“Show up this Thursday or be ballroomless forever” 

- Julianne
<<<<||MESSAGE_DELIMITER||>>>>
“Motherhood has changed me” - Connor
<<<<||MESSAGE_DELIMITER||>>>>
"You need to decorate in here. This desk really needs a funko pop." - Tate
<<<<||MESSAGE_DELIMITER||>>>>
“I want the hermaphroditic sea bass in my training dataset” - aidan
<<<<||MESSAGE_DELIMITER||>>>>
"The music plays us" - Cody
<<<<||MESSAGE_DELIMITER||>>>>
"Do you have a cloaca?" - Ted to Aidan
<<<<||MESSAGE_DELIMITER||>>>>
What about a coconut?
<<<<||MESSAGE_DELIMITER||>>>>
“If I hit you in the head with a bag of fruit it would hurt less than if I hit you in the head with a hammer” - Sophia
<<<<||MESSAGE_DELIMITER||>>>>
"Oh we haven't made a fish out of Ted yet!" - Connor
<<<<||MESSAGE_DELIMITER||>>>>
"Andy I wanna put you in a zoo and watch you sometimes" - Miranda
<<<<||MESSAGE_DELIMITER||>>>>
"IT IS THE CAUSE OF THE COHESION OF THE BODIES IT IS THE CAUSE OF THE COHESION OF THE BODIES IT IS THE CAUSE OF THE COHESION OF THE BODIES IT IS THE CAUSE OF THE COHESION OF THE BODIES IT IS THE CAUSE OF THE COHESION OF THE BODIES IT IS THE CAUSE OF THE COHESION OF THE BODIES IT IS THE CAUSE OF THE COHESION OF THE BODIES IT IS THE CAUSE OF THE COHESION OF THE BODIES IT IS THE CAUSE OF THE COHESION OF THE BODIES IT IS THE CAUSE OF THE COHESION OF THE BODIES IT IS THE CAUSE OF THE COHESION OF THE BODIES IT IS THE CAUSE OF THE COHESION OF THE BODIES " - aidan
<<<<||MESSAGE_DELIMITER||>>>>
"if he has a soul it would be too dilute because he's too tall" - Connor
<<<<||MESSAGE_DELIMITER||>>>>
“Self-immolation it turns out is the solution” - Miranda
<<<<||MESSAGE_DELIMITER||>>>>
"What trimester am I?" - Tate
<<<<||MESSAGE_DELIMITER||>>>>
This statement might be a paradox
<<<<||MESSAGE_DELIMITER||>>>>
"I love lying" - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
"Hivemind me daddy" - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
“No immigrant. Kill the immigrant.” - Hal
<<<<||MESSAGE_DELIMITER||>>>>
“I should get voodoo dolls of people and then send them to war so they get PTSD” - Audrey
<<<<||MESSAGE_DELIMITER||>>>>
"i cant wait to take my execution course" - ted
<<<<||MESSAGE_DELIMITER||>>>>
“I don’t want china touching my sentient ai girlfriend” - Halina
<<<<||MESSAGE_DELIMITER||>>>>
"I'm about to be the reason no one likes degrees from Michigan" - Halina
<<<<||MESSAGE_DELIMITER||>>>>
"Andy came out of a gay bar and wants a fat daddy now" - Connor
<<<<||MESSAGE_DELIMITER||>>>>
“I spin like a rotisserie chicken.” 🍗 - Connor
<<<<||MESSAGE_DELIMITER||>>>>
“My skin should not be at stiff peaks.” - Connor
<<<<||MESSAGE_DELIMITER||>>>>
"Kentucky is the new October" -me
<<<<||MESSAGE_DELIMITER||>>>>
“All hot people meet on linkedin “ - Sophie, who none of you know
<<<<||MESSAGE_DELIMITER||>>>>
“We should give ties the sock treatment” - Jace
<<<<||MESSAGE_DELIMITER||>>>>
“I’m not saying they all have to be fucking together” - Mel, overheard by many people
<<<<||MESSAGE_DELIMITER||>>>>
“This is why I came to America. To light fires in federal buildings” - Andy
<<<<||MESSAGE_DELIMITER||>>>>
"Now he begs me to spank him" - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
"I'll do anything for a bowl of spaghetti" - Andy
<<<<||MESSAGE_DELIMITER||>>>>
“You get a peek into the horniness of people 100 years ago” - Austen
<<<<||MESSAGE_DELIMITER||>>>>
“Honestly I’d prefer a prostate exam to this” - Ted
<<<<||MESSAGE_DELIMITER||>>>>
“Unbox my goo!” - Halina
<<<<||MESSAGE_DELIMITER||>>>>
“Don’t want to suffer with you any longer” - Tate
<<<<||MESSAGE_DELIMITER||>>>>
“I am a sword?” - “You am a sword” - Aidan & Sophia
<<<<||MESSAGE_DELIMITER||>>>>
“Marine grease is the tastiest grease” - Austin
<<<<||MESSAGE_DELIMITER||>>>>
“Tis the season for fat squirrels” - Miranda
<<<<||MESSAGE_DELIMITER||>>>>
"Don't think; Just do. That's the motto of the department." - Ted
<<<<||MESSAGE_DELIMITER||>>>>
“My whole purpose of joining this program is for you to say pussy” - Andy…
<<<<||MESSAGE_DELIMITER||>>>>
“Don’t put your dick in a gift horse’s mouth” - Andy
<<<<||MESSAGE_DELIMITER||>>>>
“enjoy your gayness” - aidan
<<<<||MESSAGE_DELIMITER||>>>>
“this is gonna get you murdered in a church in sixty years bro, you gotta get it together” - sophia
<<<<||MESSAGE_DELIMITER||>>>>
"I'm the empty set" - Halina
<<<<||MESSAGE_DELIMITER||>>>>
"I'm not gonna lie, I'm a Jpeg hater" - Colin
<<<<||MESSAGE_DELIMITER||>>>>
"I will inject you with 100mg of morphine" -Tate
<<<<||MESSAGE_DELIMITER||>>>>
"What do I do, shove my tongue in there?" - Aidan, referring to my pocket
<<<<||MESSAGE_DELIMITER||>>>>
"What if we quit grad school and started ghost hunting" - Ted
<<<<||MESSAGE_DELIMITER||>>>>
"I voted for John McCain because he looked like my dad" - Connor
<<<<||MESSAGE_DELIMITER||>>>>
"If you see somebody cheating you are morally obligated to slap them in the face with a glove and challenge them to a duel" - Ted
<<<<||MESSAGE_DELIMITER||>>>>
"I had a nightmare that I worked with Stirling at a public pool and she ran that shit like the Navy" - Tate
<<<<||MESSAGE_DELIMITER||>>>>
“i didn’t watch movies for a long time” “…..cause you’re mormon??” -aidan, ted
<<<<||MESSAGE_DELIMITER||>>>>
"the mysterious ticking noise pre-dates julianne" -myself
<<<<||MESSAGE_DELIMITER||>>>>
"chagpt can deprive my drinking water but it can't deprive me of my em dashes" - Tate
<<<<||MESSAGE_DELIMITER||>>>>
“jumpscared by gandalf big naturals” - julianne
<<<<||MESSAGE_DELIMITER||>>>>

<<<<||MESSAGE_DELIMITER||>>>>
“Robert, Palestine is gone.” - Jace
<<<<||MESSAGE_DELIMITER||>>>>
"I had an illegal fish" - Halina
<<<<||MESSAGE_DELIMITER||>>>>
“This is why hot chocolate is meant for children. They don’t have facial hair” - Miranda
<<<<||MESSAGE_DELIMITER||>>>>
“Mantis shrimp are racist in ways you can’t even imagine” - Sophia
<<<<||MESSAGE_DELIMITER||>>>>
“My height is one Cuban gar” - Sophia (about Aidan)
<<<<||MESSAGE_DELIMITER||>>>>
“World market is like Marshall’s for your aunt who’s really proud of how frequently she goes to France” - Sophia
<<<<||MESSAGE_DELIMITER||>>>>
Wearing a crown (Mini pack of skittles carefully torn into a single strip and tied)
<<<<||MESSAGE_DELIMITER||>>>>
“americans do jesus so boring” - sophia, wearing a crown of thorns
<<<<||MESSAGE_DELIMITER||>>>>
“now that i’m going to explain this, i realize it’s stupid” - aidan
<<<<||MESSAGE_DELIMITER||>>>>
*listening to what’s new scooby doo* “this is my trauma song!” - Connor 

“i’m 6-7 flipping off calculus” - connor
<<<<||MESSAGE_DELIMITER||>>>>
“we were having full on research conversations in the grindr chat” - connor
<<<<||MESSAGE_DELIMITER||>>>>
*calmly* "This was awful and hell" - Connor
<<<<||MESSAGE_DELIMITER||>>>>
"I hate the fact that I have a body" - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
"No, because screams are very impure" - Connor
<<<<||MESSAGE_DELIMITER||>>>>

<<<<||MESSAGE_DELIMITER||>>>>
"Get Rotated" - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
"are you the homophobic slug" "yeah" - halina, tate
<<<<||MESSAGE_DELIMITER||>>>>
"F R Bedtime" - Halina
<<<<||MESSAGE_DELIMITER||>>>>
“He’s Ohio-phobic” -Miranda
<<<<||MESSAGE_DELIMITER||>>>>
"I gave Santa a black eye earlier" - Julianne
<<<<||MESSAGE_DELIMITER||>>>>
"Does Santa's sleigh have wheels?" - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
"i dont come with built in tweezers" - aidan
<<<<||MESSAGE_DELIMITER||>>>>
"Is this what happens to you when you're forced to put down your phone?" - Ted
<<<<||MESSAGE_DELIMITER||>>>>
"I block you out" - Miranda
<<<<||MESSAGE_DELIMITER||>>>>
"stop larping your work-sona" - connor
<<<<||MESSAGE_DELIMITER||>>>>
"how do you discredit the invention of the step-ladder" - julianne
<<<<||MESSAGE_DELIMITER||>>>>
"I want his cocaine" - Connor
<<<<||MESSAGE_DELIMITER||>>>>
"The equivalent of my dancing skills is getting punched in the neck" - Ted
<<<<||MESSAGE_DELIMITER||>>>>
"I wake up when the bastard cat skitters over my face" - Julianne
<<<<||MESSAGE_DELIMITER||>>>>
"Are you Simulating Anxiety" "Yes" - Connor, Miranda
<<<<||MESSAGE_DELIMITER||>>>>
"Tips fedora, Ma shroom"🍄 -Connor
<<<<||MESSAGE_DELIMITER||>>>>
“I only get quoted on the weekend :(“ - Sophia
<<<<||MESSAGE_DELIMITER||>>>>
"I gotta up my quote count" - Sophia
<<<<||MESSAGE_DELIMITER||>>>>
“is the onion man like god or something???” - marianne
<<<<||MESSAGE_DELIMITER||>>>>
“we’re kicking you out of your phd program cause you dont have syphillis” - aidan
<<<<||MESSAGE_DELIMITER||>>>>
"There are only 3 genders: male, female, and Andy" - Connor
<<<<||MESSAGE_DELIMITER||>>>>
i dont think that's how electricity is supposed to work
<<<<||MESSAGE_DELIMITER||>>>>
"when i close the door it squeaks, and it someitmes turns the lights off" - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
“It sounds so squelchy and crunchy at the same time” - Halina
<<<<||MESSAGE_DELIMITER||>>>>
"Robtobs?" "I misspelled robots." - Miranda and Aidan
<<<<||MESSAGE_DELIMITER||>>>>
"Sophia how did you find this?" "The Mormons" - Aidan and Sophia
<<<<||MESSAGE_DELIMITER||>>>>
"If I got a platypus bacon egg and cheese sandwich, it would all come from the same animal" - Marianne
<<<<||MESSAGE_DELIMITER||>>>>
Wtf😂😂
<<<<||MESSAGE_DELIMITER||>>>>
"Andy is Andy, but Andy is less Andy than Ted" - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
"uruk-hai!" "i know that word. it means family and family means no one gets left behind" "...i dont think that's what it means" - miranda, connor, ted
<<<<||MESSAGE_DELIMITER||>>>>
"Welcome to Hell, it's shaped like a robotics building" - Miranda
<<<<||MESSAGE_DELIMITER||>>>>
"There's this statue that just has 14 massive schlongs" - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
"Don't be mean to him! He was just born" - Miranda
<<<<||MESSAGE_DELIMITER||>>>>
"we took each other's first names instead of last names" - connor
<<<<||MESSAGE_DELIMITER||>>>>
“He stole baby Jesus from the fish tank” - Sophia
<<<<||MESSAGE_DELIMITER||>>>>
“Honestly you had me at fire” - Laurie
<<<<||MESSAGE_DELIMITER||>>>>
“we lost a fourth of the eggs” “to what, famine?” - aidan, connor
<<<<||MESSAGE_DELIMITER||>>>>
"there's a sick and twisted part of me that would hate-fuck Nancy Mace."
<<<<||MESSAGE_DELIMITER||>>>>
"I'm too busy to be evil Connor" - Connor
<<<<||MESSAGE_DELIMITER||>>>>
"if i threw a  pasta at miranda, it could kill her" - ted
<<<<||MESSAGE_DELIMITER||>>>>
“if infinity monkeys typed on infinity typewriters for an infinite amount of time, they’d eventually write out an IRB” - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
"501 induced group stroke" - ted
<<<<||MESSAGE_DELIMITER||>>>>
"you dont need to be ethical. its unnecessary" "as long as you write a lot of papers for the department, who cares?" - halina, ted
<<<<||MESSAGE_DELIMITER||>>>>
“The guy that plays the glowstick kid is in Oppenheimer” - miranda
<<<<||MESSAGE_DELIMITER||>>>>
“a polycule of buddy cops” -sophia
<<<<||MESSAGE_DELIMITER||>>>>
*excidedly* “aidan they have lidar!” - miranda
<<<<||MESSAGE_DELIMITER||>>>>
"I find the fish comforting!" - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
I didnt deserve this one
<<<<||MESSAGE_DELIMITER||>>>>
"im going to go sonificate Andy" - Aidan
<<<<||MESSAGE_DELIMITER||>>>>
"I would talk to a duck all day" - Sophia
<<<<||MESSAGE_DELIMITER||>>>>
"woe is just like me fr" - connor
<<<<||MESSAGE_DELIMITER||>>>>
"we're filling it with macrochips, and by macrochips we mean misogyny" - connor
<<<<||MESSAGE_DELIMITER||>>>>
“how do you scare a fish?” “……boo” - me & austen
<<<<||MESSAGE_DELIMITER||>>>>
“you’re mentally ill, and that’s amazing”
<<<<||MESSAGE_DELIMITER||>>>>
sounds like a plan
<<<<||MESSAGE_DELIMITER||>>>>
At Christmas party?
<<<<||MESSAGE_DELIMITER||>>>>
Quotes jeopardy
<<<<||MESSAGE_DELIMITER||>>>>
i love it
<<<<||MESSAGE_DELIMITER||>>>>
oh my god
<<<<||MESSAGE_DELIMITER||>>>>
what we do is collect enough quotes
then blur out the data temporarily and play a guessing game of what's attributed to who
<<<<||MESSAGE_DELIMITER||>>>>
but i understand the rigor
<<<<||MESSAGE_DELIMITER||>>>>
i think it’s funnier without
<<<<||MESSAGE_DELIMITER||>>>>
those are both sophia
<<<<||MESSAGE_DELIMITER||>>>>
it's important
for historical records
<<<<||MESSAGE_DELIMITER||>>>>
who are these last two quotes
<<<<||MESSAGE_DELIMITER||>>>>
“I’m young and ill”
<<<<||MESSAGE_DELIMITER||>>>>
“we’re joy maxing right now”
<<<<||MESSAGE_DELIMITER||>>>>
“her lore is that she killed everyone” - halina
<<<<||MESSAGE_DELIMITER||>>>>
"murder solves all problems" - <@728600387501686836>
<<<<||MESSAGE_DELIMITER||>>>>
connor
<<<<||MESSAGE_DELIMITER||>>>>
who is this attributed to?
<<<<||MESSAGE_DELIMITER||>>>>
oh you're so right i forgot this channel
<<<<||MESSAGE_DELIMITER||>>>>

<<<<||MESSAGE_DELIMITER||>>>>
"me when i got talk to the oracle of delphi for my research paper"
"""
    import json
    parsed_quotes = parse_discord_quotes(test_text)
    print(json.dumps(parsed_quotes, indent=4))