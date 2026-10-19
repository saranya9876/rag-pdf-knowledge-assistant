import time

def log_time(label, start):
    end = time.time()
    print(f"[{label}] Time: {end - start:.2f}s")