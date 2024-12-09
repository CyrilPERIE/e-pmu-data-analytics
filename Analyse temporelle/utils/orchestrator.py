import time
from .api import get_programme
from .super_worker import SuperWorker
from .log import info

class Orchestrator:


    def __init__(self):

        info("Lancement de l'orchestrateur.")
        self.programmes_running = {}
        self.programmes_to_start = []
        self.super_workers = []

        while True:
            info("Vérification de l'existence de nouveaux programmes disponibles...")
            programme = get_programme("01012021")
            dates_programmes_disponibles = programme["programme"]["datesProgrammesDisponibles"]
            self._update_programmes_running(dates_programmes_disponibles)
            for programme_to_start in self.programmes_to_start:
                info(f"Programme [{programme_to_start}] à démarrer")
                sw = SuperWorker(programme_to_start)
                sw.start()
                self.super_workers.append(sw)
            self.programmes_to_start = []
            time.sleep(3600)

    
    def _update_programmes_running(self, dates_programmes_disponibles):
        
        self.programmes_to_start = [i for i in dates_programmes_disponibles if i not in self.programmes_running]
        self.programmes_running = set([i for i in self.programmes_running if i in dates_programmes_disponibles])
        self.programmes_running.update(dates_programmes_disponibles)

        for i in range(len(self.super_workers) -1, -1, -1):
            super_worker = self.super_workers[i]
            if super_worker.status == "sleeping":
                super_worker.join()
                info(f"Le Super Worker du programme {[super_worker.programme_date]} a été arrêté.")
                del self.super_workers[i]
        


if __name__ == "__main__":

    Orchestrator()