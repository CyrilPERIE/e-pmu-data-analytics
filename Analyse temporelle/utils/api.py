import requests

def fetch_url(endpoint):
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Erreur lors de l'accès à {endpoint}: {e}")
        return None

def get_participants(programme_date, reunion_num, course_num):
    endpoint = (
        f"https://online.turfinfo.api.pmu.fr/rest/client/61/programme/{programme_date}/R{reunion_num}/C{course_num}/participants?specialisation=INTERNET"
    )
    return fetch_url(endpoint)

def get_programme(programme_date):
    endpoint = f"https://online.turfinfo.api.pmu.fr/rest/client/61/programme/{programme_date}?meteo=true&specialisation=INTERNET"
    return fetch_url(endpoint)