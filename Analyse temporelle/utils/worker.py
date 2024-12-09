import os
import csv
import threading
import pandas as pd
from .api import get_participants, get_course
import time
from .log import info


path = "tmp"

class Worker(threading.Thread):


    def __init__(self, programme_date, reunion_num, course):

        super().__init__()
        self.programme_date = programme_date
        self.reunion_num = reunion_num
        self.course_num = course["numExterne"]
        self.course_heureDepart = course["heureDepart"]
        self.course_specialite = course["specialite"]

        self.file_path = os.path.join(path, f"{self.programme_date}_R{self.reunion_num}_C{self.course_num}.csv")
        self.header = [
            "programme_date", 
            "reunion_num", 
            "course_num",
            "course_heureDepart", 
            "course_specialite",
            "participant_num", 
            "participant_nom", 
            "participant_dateRapport",
            "participant_rapport", 
            "participant_indicateurTendance", 
            "participant_nombreIndicateurTendance"
        ]

        self.status = "running"
        info(f"Worker {self.programme_date}_R{self.reunion_num}_C{self.course_num} est au status [{self.status}]")


    def run(self):

        while True:

            participants = get_participants(self.programme_date, self.reunion_num, self.course_num)
            if participants:
                self._create_file_if_not_exist()
                data, dateRapport = self._extract_useful_datas(participants)
                if self._is_new_datas(dateRapport) and data:
                    self._write_data(data)
                if self._is_course_over():
                    info(f"La course {self.programme_date}_R{self.reunion_num}_C{self.course_num} est terminée.")
                    break
            time.sleep(60)
        
        self.status = "sleeping" 
        info(f"Worker {self.programme_date}_R{self.reunion_num}_C{self.course_num} est au status [{self.status}].")

    
    def _is_course_over(self):
        course = get_course(self.programme_date, self.reunion_num, self.course_num)
        return "isArriveeDefinitive" in course and course["isArriveeDefinitive"]

    
    def _create_file_if_not_exist(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.header)
            info(f"Un fichier a été créé pour suivre les côtes de la course {self.programme_date}_R{self.reunion_num}_C{self.course_num}.")


    def _extract_useful_datas(self, participants):
        result, dateRapport = [], ""
        for participant in participants["participants"]:
            try:
                data = [
                    self.programme_date, 
                    self.reunion_num, 
                    self.course_num,
                    self.course_heureDepart, 
                    self.course_specialite,
                    participant["numPmu"], participant["nom"],
                    participant["dernierRapportDirect"]["dateRapport"],
                    participant["dernierRapportDirect"]["rapport"],
                    participant["dernierRapportDirect"]["indicateurTendance"],
                    participant["dernierRapportDirect"]["nombreIndicateurTendance"]
                ]
                result.append(data)
                dateRapport = participant["dernierRapportDirect"]["dateRapport"]
            except KeyError:
                continue
        return result, dateRapport
    

    def _is_new_datas(self, dateRapport):
        if not os.path.exists(self.file_path):
            return True
        df = pd.read_csv(self.file_path)
        if df.empty:
            return True
        last_date = df['participant_dateRapport'].iloc[-1]
        return dateRapport > last_date
    

    def _write_data(self, data):
        info(f"Nouvelles données de côtes ajoutées à la course {self.programme_date}_R{self.reunion_num}_C{self.course_num}.")
        with open(self.file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)
