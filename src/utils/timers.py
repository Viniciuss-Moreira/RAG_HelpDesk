from time import time

class Timer:
    def __init__(self, label="Elapsed"):
        self.label = label

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        print(f"{self.label}: {time() - self.start:.2f}s")