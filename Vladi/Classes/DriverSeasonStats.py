class DriverSeasonStats:
    def __init__(self, driver_name, year):
        self.driver = driver_name
        self.year = year
        self.results = []  # Lista di dict per ogni gara
    
    def add_race_result(self, race_name, position, points, fastest_lap):
        """Adding results of a race."""
        self.results.append({
            'race': race_name,
            'position': position,
            'points': points,
            'fastest_lap': fastest_lap
        })

    
    def get_avg_finish(self):
        """Calculate the average finish position."""
        positions = [r['position'] for r in self.results if r['position']]
        return sum(positions) / len(positions) if positions else None