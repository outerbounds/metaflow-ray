# Source: https://docs.ray.io/en/latest/ray-core/walkthrough.html

import ray
import time

ray.init()

"""
In recent versions of Ray (>=1.5), ray.init() is automatically called on the first use of a Ray remote API.
"""


# Define the Counter actor.
@ray.remote
class Counter:
    def __init__(self):
        self.i = 0

    def get(self):
        return self.i

    def incr(self, value):
        self.i += value


# Create a Counter actor.
c = Counter.remote()

# Submit calls to the actor. These calls run asynchronously but in
# submission order on the remote actor process.
for _ in range(int(25)):
    time.sleep(1)
    c.incr.remote(1)

# Retrieve final actor state.
print(ray.get(c.get.remote()))
# -> 10
