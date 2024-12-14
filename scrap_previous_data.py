from datetime import date, timedelta
from asyncio import run

from async_scraping.scraping import scrap_day


if __name__ == "__main__":
    start_date = date(2024, 1, 1)
    today = date.today()
    current_date = start_date

    while current_date < today:
        print(f"Scraping {current_date}")
        try:
            run(scrap_day(current_date))
        except BaseException as e:
            print(f"\033[91m{e}\033[0m")

        current_date += timedelta(days=1)
