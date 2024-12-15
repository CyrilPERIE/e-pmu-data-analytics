from datetime import date, timedelta
from asyncio import run

from async_scraping.scraping import scrap_day


if __name__ == "__main__":
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    data_folder = "data/2023"

    current_date = start_date
    while current_date < end_date:
        print(f"Scraping {current_date}")
        try:
            run(scrap_day(current_date, data_folder))
        except BaseException as e:
            print(f"\033[91m{e}\033[0m")

        current_date += timedelta(days=1)
