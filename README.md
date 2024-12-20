# Modèles de paris sur les courses de chevaux

## Install dependencies

### Install poetry

```
curl -sSL https://install.python-poetry.org | python3 -
```

### Install dependencies

```
poetry install
```

## Lancer le scraping sur les jours passés

Edit `epmu/scraping/scrap_previous_data.py` to set the date range you want to scrape and the data folder name. Then run:
```
python epmu/scraping/scrap_previous_data.py
```

## Run the whole process from model training to betting simulation

```
python epmu/model/optimize.py
python simulations/betting.py
```
