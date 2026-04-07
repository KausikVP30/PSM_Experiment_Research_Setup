import csv
import os
from datetime import datetime

class Logger:
    def __init__(self, log_file='logs/experiment_log.csv'):
        self.log_file = log_file

        # Create file with header if not exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "query",
                    "confidence",
                    "sim_query",
                    "sim_answer",
                    "sim_docs",
                    "source",
                    "latency",
                    "memory_size",
                    "retrieval_count"
                ])

    def log(self, query, confidence, sim_q, sim_a, sim_d,
            source, latency, memory_size, retrieval_count):

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(),
                query,
                confidence,
                sim_q,
                sim_a,
                sim_d,
                source,
                latency,
                memory_size,
                retrieval_count
            ])