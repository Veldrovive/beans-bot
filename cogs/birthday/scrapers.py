"""
For fun we scrape data from a couple web pages.
"""

import httpx
import asyncio
from bs4 import BeautifulSoup
from enum import Enum
from dataclasses import dataclass
import datetime
import json

from google import genai

class HolidayType(Enum):
    SINGLE_DAY = "Single Day Holiday"
    MULTI_DAY = "Multi-day Event"

@dataclass
class Holiday:
    type: HolidayType
    name: str
    description: str
    link: str

    day: int
    month: int

    def __str__(self):
        return f"{self.name} ({self.type.value}) - {self.description} ({self.link})"

def scrape_holidays(html_content: str, day: int, month: int) -> list[Holiday]:
    soup = BeautifulSoup(html_content, 'html.parser')
    all_holidays = []

    # 1. Scrape Single Day Holidays
    # These are stored in div elements with the class 'holiday'
    holiday_cards = soup.find_all('div', class_='holiday')
    for card in holiday_cards:
        title_tag = card.find('h2', class_='mdl-card__title-text')
        if not title_tag or not title_tag.a:
            continue
            
        name = title_tag.a.text.strip()
        if name == "On This Day in History":
            continue
        link = title_tag.a['href']
        
        # The description usually contains alternative names and the observed date
        desc_tag = card.find('div', class_='mdl-card__supporting-text')
        description = desc_tag.text.strip() if desc_tag else "No description available"
        
        all_holidays.append(Holiday(
            type=HolidayType.SINGLE_DAY,
            name=name,
            description=description,
            link=link,
            day=day,
            month=month,
        ))

    # 2. Scrape Multi-day Events Continuing Today
    # These are stored in list items under the checkiday-list class
    multi_day_items = soup.find_all('li', class_='mdl-list__item')
    for item in multi_day_items:
        primary_content = item.find('span', class_='mdl-list__item-primary-content')
        if not primary_content or not primary_content.a:
            continue
            
        name = primary_content.a.text.strip()
        link = primary_content.a['href']
        
        # The sub-title contains the observance duration
        sub_title_tag = primary_content.find('span', class_='mdl-list__item-sub-title')
        description = sub_title_tag.text.strip() if sub_title_tag else "No description available"
        
        all_holidays.append(Holiday(
            type=HolidayType.MULTI_DAY,
            name=name,
            description=description,
            link=link,
            day=day,
            month=month,
        ))

    return all_holidays

async def get_holidays(day: int, month: int, year: int) -> list[Holiday]:
    url = f"https://www.checkiday.com/{month}/{day}/{year}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to get holidays from {url}: {response.status_code}")
        return scrape_holidays(response.text, day, month)

def get_best_holiday(holidays: list[Holiday], gemini_harness) -> tuple[str | None, Holiday | None]:
    response_schema = genai.types.Schema(
        type = genai.types.Type.OBJECT,
        required = ["reason", "event_number"],
        properties = {
            "reason": genai.types.Schema(
                type = genai.types.Type.STRING,
            ),
            "event_number": genai.types.Schema(
                type = genai.types.Type.INTEGER,
            ),
        },
    )

    holidays_list = "\n".join([f"{i+1}: {holiday.name}" for i, holiday in enumerate(holidays)])

    prompt = f"""
{holidays_list}

You are given a list of holidays that occur on a given day.
Your job is to pick the best holiday to celebrate.
We prefer lesser known and odd holidays.
    """

    response = gemini_harness.generate(
        prompt=prompt,
        response_schema=response_schema,
    )
    if response.text is None:
        print("No response from Gemini")
        return None, None

    response_json = json.loads(response.text)
    reason = response_json["reason"]
    event_number = response_json["event_number"]

    chosen_event = holidays[event_number - 1]
    return reason, chosen_event

def get_month_str(month: int) -> str:
    if month == 1:
        return "January"
    elif month == 2:
        return "February"
    elif month == 3:
        return "March"
    elif month == 4:
        return "April"
    elif month == 5:
        return "May"
    elif month == 6:
        return "June"
    elif month == 7:
        return "July"
    elif month == 8:
        return "August"
    elif month == 9:
        return "September"
    elif month == 10:
        return "October"
    elif month == 11:
        return "November"
    elif month == 12:
        return "December"
    else:
        raise ValueError(f"Invalid month: {month}")

@dataclass
class OnThisDayEvent:
    day: int
    month: int
    year: int
    description: str
    link: str

    def __str__(self):
        return f"**{self.year}**: {self.description} ({self.link})"

def scrape_on_this_day_events(html_content: str, day: int, month: int) -> list[OnThisDayEvent]:
    soup = BeautifulSoup(html_content, 'html.parser')
    events = []

    # Find all highlighted sections (ignoring standard <ul class="event-list">)
    highlight_sections = soup.select('.section--highlight')

    for section in highlight_sections:
        # 1. Extract the title (e.g., "Battle of Nocera")
        title_tag = section.select_one('.poi__heading-txt')
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)

        # 2. Locate the paragraph containing the year and description
        # Usually, this is the first `<p>` tag inside the section.
        p_tag = section.select_one('p')
        if not p_tag:
            continue

        # 3. Extract the year 
        date_tag = p_tag.select_one('.date')
        if not date_tag:
            continue
            
        try:
            year = int(date_tag.get_text(strip=True))
        except ValueError:
            # Fallback if the date isn't a cleanly parsed year 
            year = datetime.datetime.now().year
            
        # Remove the date tag from the DOM temporarily so it doesn't pollute our description text
        date_tag.extract()
        
        # Clean up the description and merge it with the highlighted title
        # Strip trailing colons or spaces that might have been left next to the date
        raw_desc = p_tag.get_text(separator=' ', strip=True).lstrip(': ')
        description = f"{title}: {raw_desc}"

        # 4. Extract the most relevant link
        # Priority 1: An article link in the header (e.g., `<a class="section__link">`)
        link_tag = section.select_one('.poi__heading a.section__link')
        if not link_tag:
            # Priority 2: The first embedded link in the description (often the primary subject)
            link_tag = p_tag.select_one('a')

        link = link_tag.get('href', '') if link_tag else ''
        
        # Resolve relative URLs to absolute URLs
        if link.startswith('/'):
            link = f"https://www.onthisday.com{link}"

        events.append(OnThisDayEvent(
            day=day,
            month=month,
            year=year,
            description=description,
            link=link
        ))

    return events

async def get_on_this_day(day: int, month: int) -> list[OnThisDayEvent]:
    # onthisday.com uses full month names in its URL paths
    month_str = datetime.date(2000, month, 1).strftime('%B').lower()
    
    url = f"https://www.onthisday.com/events/{month_str}/{day}"
    
    async with httpx.AsyncClient() as client:
        # Sites like this often block default Python User-Agents, so we supply a standard one.
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = await client.get(url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to get on this day from {url}: {response.status_code}")
            
        return scrape_on_this_day_events(response.text, day, month)

def get_best_on_this_day(events: list[OnThisDayEvent], gemini_harness) -> tuple[str | None, OnThisDayEvent | None]:
    response_schema = genai.types.Schema(
        type = genai.types.Type.OBJECT,
        required = ["reason", "on_this_day_number"],
        properties = {
            "reason": genai.types.Schema(
                type = genai.types.Type.STRING,
            ),
            "on_this_day_number": genai.types.Schema(
                type = genai.types.Type.INTEGER,
            ),
        },
    )

    events_list = "\n".join([f"{i+1}: {event.year} - {event.description}" for i, event in enumerate(events)])

    prompt = f"""
{events_list}

You are given a list of events that happened on a given day.
Your job is to pick the best event to celebrate.
We prefer lesser known and odd events that are still important to history.
    """

    response = gemini_harness.generate(
        prompt=prompt,
        response_schema=response_schema,
    )
    if response.text is None:
        print("No response from Gemini")
        return None, None

    response_json = json.loads(response.text)
    reason = response_json["reason"]
    on_this_day_number = response_json["on_this_day_number"]

    chosen_event = events[on_this_day_number - 1]
    return reason, chosen_event
        

if __name__ == "__main__":
    async def test():
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

        from gemini_harness import GeminiHarness
        from dotenv import load_dotenv
        load_dotenv()
        gemini_harness = GeminiHarness()

        # test_day, test_month, test_year = 24, 12, 2025
        test_day, test_month, test_year = 18, 7, 2026

        holidays = await get_holidays(test_day, test_month, test_year)
        single_day_holidays = [h for h in holidays if h.type == HolidayType.SINGLE_DAY]
        print("Holidays:")
        for i, holiday in enumerate(single_day_holidays):
            print(f"{i+1}: {holiday}")
        
        reason, chosen_holiday = get_best_holiday(single_day_holidays, gemini_harness)
        if chosen_holiday is None:
            print("No holiday chosen")
        else:
            print(f"\nBest Holiday: {chosen_holiday}")
            print(f"Reason: {reason}")

        print("\n\n\nOn This Day:")
        on_this_day = await get_on_this_day(test_day, test_month)
        for i, event in enumerate(on_this_day):
            print(f"{i+1}: {event}")

        reason, chosen_on_this_day = get_best_on_this_day(on_this_day, gemini_harness)
        if chosen_on_this_day is None:
            print("No on this day chosen")
        else:
            print(f"\nBest On This Day: {chosen_on_this_day}")
            print(f"Reason: {reason}")

    async def test_find_action_holidays():
        import re
        import datetime
        import asyncio
        
        # We look for actionable, creative holidays following the "Verb a [Noun] Day" pattern
        verbs = [
            "draw", "create", "make", "paint", "write", "build", "design", "invent",
            "craft", "color", "tell", "do", "start", "hug", "kiss", "thank", "plant",
            "adopt", "take", "bake", "cook", "read", "sing", "pet", "visit", "learn",
            "earn", "buy", "send", "call", "watch", "play", "give", "find", "catch",
            "wear", "eat", "drink", "walk", "clean", "wash"
        ]
        
        # Matches "Draw a", "Write an", etc.
        verb_group = "|".join(verbs)
        pattern = re.compile(rf'\b({verb_group})\s+an?\s+', re.IGNORECASE)
        
        year = datetime.datetime.now().year
        start_date = datetime.date(year, 1, 1)
        end_date = datetime.date(year, 12, 31)
        
        dates_to_check = []
        current_date = start_date
        while current_date <= end_date:
            dates_to_check.append((current_date.day, current_date.month, current_date.year))
            current_date += datetime.timedelta(days=1)
            
        print(f"Searching for action holidays across {year}...")
        semaphore = asyncio.Semaphore(5) # Conservative concurrency to avoid rate-limiting
        
        async def fetch_and_check(day, month, year):
            async with semaphore:
                try:
                    holidays = await get_holidays(day, month, year)
                    matches = []
                    for h in holidays:
                        if pattern.search(h.name):
                            matches.append(h)
                    
                    if matches:
                        print(f"[{month}/{day}/{year}] Matches found:")
                        for m in matches:
                            print(f"  - {m.name} ({m.link})")
                    return matches
                except Exception as e:
                    print(f"Error on {month}/{day}/{year}: {e}")
                    return []
                finally:
                    await asyncio.sleep(0.5) # Polite delay
                    
        # Concurrently fetch and check all days in the year
        tasks = [fetch_and_check(d, m, y) for d, m, y in dates_to_check]
        results = await asyncio.gather(*tasks)
        
        all_matches = [h for day_matches in results for h in day_matches]
        print(f"\nFinished! Found {len(all_matches)} action holidays in {year}.")

    # Run the standard test:
    asyncio.run(test())

    # Uncomment to run the full year scan for action holidays:
    # asyncio.run(test_find_action_holidays())