# bench_oni.py  
import time  
from core import EntropicIdentity  
import psutil

with open("stress_log.csv", "a") as f:
    f.write(f"{time.time()},{identity.μ},{identity.λ},{psutil.cpu_percent()}\n")


def stress_test():  
    identity = EntropicIdentity()  
    for _ in range(10_000):  
        identity.update(f"Stress input {_}")  
        print(f"CPU: {psutil.cpu_percent()}% | μ: {identity.μ:.2f}")  