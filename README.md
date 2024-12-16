# Modèles de paris sur les courses de chevaux

## Lancer le scraping sur les jours passés

Edit `epmu/scraping/scrap_previous_data.py` to set the date range you want to scrape and the data folder name. Then run:
```
pip install -r requirements.txt
python epmu/scraping/scrap_previous_data.py
```

## Run the whole process from model training to betting simulation

```
python epmu/model/prepare.py
python epmu/model/evaluate_features.py
python epmu/model/train.py
python simulations/betting.py
```