import time

def measure_latency(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Latency: {end - start:.2f}s")
        return result
    return wrapper