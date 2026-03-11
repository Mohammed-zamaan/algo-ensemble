class BacktestEngine:

    def __init__(self, config):
        self.config = config

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def run(self):
        return {
            "status": "ok",
            "start": self.config.start_date,
            "end": self.config.end_date,
            "mode": self.config.backtest_mode,
        }