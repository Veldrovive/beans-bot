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

    # 1. Extract the Featured Event
    featured_card = soup.select_one('.featured-event-card')
    if featured_card:
        year_tag = featured_card.select_one('.date-label')
        desc_tag = featured_card.select_one('.description')
        link_tag = featured_card.select_one('.title a')
        
        if year_tag and desc_tag and link_tag:
            events.append(OnThisDayEvent(
                day=day,
                month=month,
                year=int(year_tag.get_text(strip=True)),
                description=desc_tag.get_text(strip=True),
                link=link_tag.get('href', '')
            ))

    # 2. Extract the remaining "More Events"
    for card in soup.select('.md-history-event'):
        year_tag = card.select_one('.date-label')
        body_tag = card.select_one('.card-body')
        
        if year_tag and body_tag:
            year_tag = year_tag.get_text(strip=True)
            if year_tag.lower() == "today":
                year = datetime.datetime.now().year
            else:
                year = int(year_tag)
            
            # The first anchor tag in the body usually represents the primary subject link
            link_tag = body_tag.find('a')
            link = link_tag.get('href', '') if link_tag else ''
            
            # Remove "Test your knowledge" quiz links and image credits to get a clean description
            for unwanted in body_tag.select('.otd-he-link, .credit'):
                unwanted.decompose()
                
            # Extract the remaining text
            description = body_tag.get_text(separator=' ', strip=True)
            
            events.append(OnThisDayEvent(
                day=day,
                month=month,
                year=year,
                description=description,
                link=link
            ))

    return events

async def get_on_this_day(day: int, month: int) -> list[OnThisDayEvent]:
    month_str = get_month_str(month)
    url = f"https://www.britannica.com/on-this-day/{month_str}-{day}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
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
We prefer lesser known and odd events.
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

        test_day, test_month, test_year = 24, 12, 2025

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

    asyncio.run(test())