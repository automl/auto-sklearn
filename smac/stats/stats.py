"""Mock smac.stats.stats module"""
class Stats:
    def __init__(self, scenario):
        self.scenario = scenario
        self.n_ta_runs = 0
        self.n_success = 0
        self.n_crashed = 0
        self.n_timeout = 0
        self.n_memout = 0
        self.n_aborted = 0
        self._start_time = None
        self._time_used = 0
        
    def start_timer(self):
        pass
    
    def stop_timer(self):
        pass
    
    def get_used_wallclock_time(self):
        return 0
    
    def get_reported_wallclock_time(self):
        return 0
    
    def reset(self):
        pass
    
    def save(self, fn=""):
        pass
    
    def load(self, fn):
        pass
    
    def update_from_incumbent(self, incumbent):
        pass
    
    def print_stats(self):
        pass
