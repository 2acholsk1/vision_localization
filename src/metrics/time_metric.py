import time

import numpy as np

from src.logger import log
from src.metrics.metric import Metric


class TimeMetric(Metric):
    def __init__(self):
        self.total_time = 0.0
        self.timestamps = np.array([])
        self.delta_t = np.array([])
        self.time_started = None
        self.time_last = None

    def compute(self):
        if self.timestamps.size == 0:
            self.time_started = time.time()
            self.timestamps = np.append(self.timestamps, 0.0)
        else:
            self.timestamps = np.append(self.timestamps, time.time() - self.time_started)

        if self.delta_t.size == 0:
            self.delta_t = np.append(self.delta_t, 0.0)
        elif self.delta_t.size == 1:
            self.delta_t = np.append(self.delta_t, time.time() - self.time_started)
            self.time_last = time.time()
        else:
            self.delta_t = np.append(self.delta_t, time.time() - self.time_last)
            self.time_last = time.time()

    def evaluate(self):
        mean_delta_t = np.sum(self.delta_t)/len(self.delta_t)
        self.total_time = self.timestamps[-1]
        log.info(
            "Total time: %s\n"
            "Mean delta t: %s\n",
            self.total_time,
            mean_delta_t
        )
