class SessionAnalyzer:
    def __init__(self, session):
        self.session = session
        self.laps = session.laps
        self.drivers = session.drivers

    def get_fastest_laps(self, driver=None):
        """Returns the fastest lap of a driver or all drivers"""
        if driver:
            return self.laps.pick_driver(driver).fastest()
        return self.laps.fastest()
    
    def get_telemetry_comparison(self,drivers):
        """Returns the telemetry comparison of two drivers"""
        telemtry_data = {}
        for driver in drivers:
            lap = self.laps.pick_driver(driver).fastest()
            telemetry_data[driver] = lap.get_telemetry()
        return telemetry_data
    
    def get_tire_strategy(self):
        """Analizza le strategie di gomme."""
        return self.laps[['Driver', 'Stint', 'Compound', 'LapNumber']]