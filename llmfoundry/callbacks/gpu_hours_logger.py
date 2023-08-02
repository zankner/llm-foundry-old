from composer import Callback, State, Logger
from composer.utils import dist


class GpuHourLogger(Callback):

    def batch_end(self, state: State, logger: Logger):
        train_wct = state.timestamp.total_wct.total_seconds()
        world_size = dist.get_world_size()
        logger.log_metrics({
            "gpu-hours/train": world_size * train_wct / self.divider,
        })
