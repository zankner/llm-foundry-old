from composer import Callback, State, Logger

class SeedStopper(Callback):

    def __init__(self, stop_time: float):
        self.stop_time = stop_time
    
    def batch_end(self, state: State, logger: Logger) -> None:
        if state.timestamp >= self.stop_time:
            state.stop_training()