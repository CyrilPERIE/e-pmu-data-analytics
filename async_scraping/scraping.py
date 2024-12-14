import os
import json
from typing import Any
from datetime import date, datetime

from asyncio import gather, run
from httpx import AsyncClient


BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/61/programme"
SUFFIX = "?specialisation=INTERNET"
COMBINATION_JOIN = "-"
DATA_FOLDER = "async_scraping/data"


def sub_dict(d: dict[str, Any], sub_keys: list[str]) -> dict[str, Any]:
    return {x: d[x] for x in sub_keys}


def get_race_key(race_input: dict[str, Any]) -> str:
    return f"R{race_input['numReunion']}C{race_input['numOrdre']}"


async def fetch_planned_race(
    client: AsyncClient, program_url: str, race_input: dict[str, Any], race_output: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """
    Fetch a race that is not finished yet
    """
    race_url = f"{program_url}/R{race_input['numReunion']}/C{race_input['numOrdre']}"
    response = await client.get(race_url + "/participants" + SUFFIX)
    participants = response.json()["participants"]

    # Odds
    race_output["rapports"] = race_output.get("rapports", {})
    for participant in participants:
        if "dernierRapportDirect" not in participant:
            continue

        p = str(participant["numPmu"])
        if p not in race_output["rapports"]:
            race_output["rapports"][p] = {}

        odds_date, odds = (
            str(participant["dernierRapportDirect"]["dateRapport"]),
            participant["dernierRapportDirect"]["rapport"],
        )
        race_output["rapports"][p][odds_date] = odds

    # Betting amounts
    response = await client.get(race_url + "/combinaisons" + SUFFIX)
    combinations = response.json()["combinaisons"]
    betting_amounts = race_output.get("enjeux", {})
    for bet in combinations:
        bet_kind = bet["pariType"]
        betting_amounts[bet_kind] = betting_amounts.get(bet_kind, {})
        for combination in bet["listeCombinaisons"]:
            key = COMBINATION_JOIN.join(map(str, combination["combinaison"]))
            if key not in betting_amounts[bet_kind]:
                betting_amounts[bet_kind][key] = {}

            betting_amount_date, betting_amount = str(bet["updateTime"]), combination["totalEnjeu"]
            betting_amounts[bet_kind][key][betting_amount_date] = betting_amount

    race_output["enjeux"] = betting_amounts

    return get_race_key(race_input), race_output


async def fetch_finished_race(
    client: AsyncClient, program_url: str, race_input: dict[str, Any], race_output: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """
    Fetch a race that is finished
    """
    # Final ordering
    race_output["ordreArrivee"] = race_input["ordreArrivee"]

    # Final odds
    race_url = f"{program_url}/R{race_input['numReunion']}/C{race_input['numOrdre']}"
    response = await client.get(race_url + "/rapports-definitifs" + SUFFIX)
    final_odds = response.json()
    race_output["rapportsDefinitifs"] = {}
    for bet in final_odds:
        bet_kind = bet["typePari"]
        race_output["rapportsDefinitifs"][bet_kind] = {
            odds["combinaison"]: odds["dividendePourUnEuro"] / 100 for odds in bet["rapports"]
        }

    return get_race_key(race_input), race_output


async def scrap_day(_date: date, data_folder: str = DATA_FOLDER) -> None:
    program_url = f"{BASE_URL}/{_date.strftime('%d%m%Y')}"

    # Fetch the program of the day
    async with AsyncClient() as client:
        response = await client.get(program_url + SUFFIX + "&meteo=true")
        meetings = response.json()["programme"]["reunions"]
        races = [race for meeting in meetings for race in meeting["courses"]]

    # Init data or load current version
    filepath = os.path.join(data_folder, _date.strftime("%d_%m_%Y") + ".json")
    if os.path.exists(filepath):
        with open(filepath, "r+") as f:
            data = json.load(f)

        if all("ordreArrivee" in v for k, v in data.items()):
            # We already have all that we need
            return
    else:
        race_metadata = [
            "heureDepart",
            "montantPrix",
            "distance",
            "discipline",
            "specialite",
            "nombreDeclaresPartants",
            "conditionSexe",
            "grandPrixNationalTrot",
            "montantOffert1er",
            "montantOffert2eme",
            "montantOffert3eme",
        ]
        meeting_metadata = ["nature", "hippodrome", "meteo"]
        data = {
            get_race_key(race): (sub_dict(race, race_metadata) | sub_dict(meeting, meeting_metadata))
            for meeting in meetings
            for race in meeting["courses"]
        }

    # Fetch races data
    async with AsyncClient() as client:
        planned_races = await gather(
            *[
                fetch_planned_race(client, program_url, race, data[get_race_key(race)])
                for race in races
                if race["categorieStatut"] == "A_PARTIR"
            ]
        )

        finished_races = await gather(
            *[
                fetch_finished_race(client, program_url, race, data[get_race_key(race)])
                for race in races
                if (
                    race.get("rapportsDefinitifsDisponibles", False)
                    and ("ordreArrivee" not in data[get_race_key(race)])
                    and ("ordreArrivee" in race)
                )
            ]
        )

    # Update data
    for race_key, race_output in planned_races + finished_races:
        data[race_key] = race_output

    with open(filepath, "w+") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    try:
        run(scrap_day(date.today()))
    except Exception as e:
        print(e)
        with open(DATA_FOLDER + date.today().strftime("%d_%m_%Y") + ".txt", "a+") as f:
            f.write(f"{datetime.now().strftime('%H:%M %Ss')} - {e}\n")
