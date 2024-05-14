import time
from joblib import Parallel, delayed

def countdown(n):
    while n>0:
        n -= 1
    return n


t = time.time()
for _ in range(20):
    print(countdown(10**7), end=" ")
print(time.time() - t)  
# takes ~10.5 seconds on medium sized Macbook Pro


t = time.time()
results = Parallel(n_jobs=8)(delayed(countdown)(10**7) for _ in range(20))
print(results)
print(time.time() - t)