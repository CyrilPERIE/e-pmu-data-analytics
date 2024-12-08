import os
import csv
import pandas as pd
import threading
from .worker import Worker
from .api import get_programme
import time
from .log import info

path = "data"
path_workers = "tmp"

class SuperWorker(threading.Thread):


    def __init__(self, programme_date):
        
        super().__init__()
        info(f"Lancement du SuperWorker pour le programme [{programme_date}]")
        self.programme_date = programme_date
        self.workers = []

        self.file_path = os.path.join(path, f"{self.programme_date}_results.csv")
        self.header = [
            "programme_date", 
            "reunion_num", 
            "course_num",
            "resultats"
        ]

    def run(self):

        while True:

            info(f"A la recherche de nouvelles courses à observer sur le programme [{self.programme_date}]...")
            self._start_workers_for_races_not_observed()

            if self._all_workers_sleeping():
                info(f"Tous les Worker dorment du programme [{self.programme_date}].")
                self._stop()
                break

            time.sleep(600)


    def _all_workers_sleeping(self):

        return all(i.status == "sleeping" for i in self.workers)


    def _start_workers_for_races_not_observed(self):

        programme = get_programme(self.programme_date)
        reunions = programme["programme"]["reunions"]
        for reunion in reunions:
            reunion_num = reunion["numExterne"]
            courses = reunion["courses"]
            for course in courses:
                if "isArriveeDefinitive" in course and course["isArriveeDefinitive"] == "true":
                    pass
                self._add_worker(reunion_num, course)


    def _stop(self):

        self._merge_csv_files()
        info(f"Les fichiers csv des côtes ont été fusionnés pour le programme [{self.programme_date}]")
        self._delete_workers_files()
        info(f"Les fichiers csv temporaires des côtes ont été supprimés pour le programme [{self.programme_date}]")
        self._retrieve_race_results()
        info(f"Les résultats des courses pour le programme [{self.programme_date}] ont été récupérées et ajoutées au fichier {self.file_path}.")


    def _retrieve_race_results(self):
        
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.header)

        programme = get_programme(self.programme_date)
        reunions = programme["programme"]["reunions"]
        for reunion in reunions:
            reunion_num = reunion["numExterne"]
            courses = reunion["courses"]
            for course in courses:
                course_num = course["numExterne"]
                resultat = course["ordreArrivee"]
                data = [
                    self.programme_date,
                    reunion_num,
                    course_num,
                    resultat
                ]
                with open(self.file_path, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerows(data)



    def _delete_workers_files(self):
        
        csv_files = [file for file in os.listdir(path_workers) if file.endswith('.csv') and file.startswith(self.programme_date)]
        for csv_file in csv_files:
            file_path = os.path.join(path_workers, csv_file)
            os.remove(file_path)


    def _add_worker(self, reunion_num, course):

        worker = Worker(self.programme_date, reunion_num, course)
        self.workers.append(worker)
        worker.start()


    def _merge_csv_files(self):

        merged_file_path = os.path.join(path, f"{self.programme_date}_merged.csv")
        all_data = []
        for worker in self.workers:
            if os.path.exists(worker.file_path):
                df = pd.read_csv(worker.file_path)
                all_data.append(df)
        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)
            merged_df.to_csv(merged_file_path, index=False)
