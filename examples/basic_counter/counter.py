import ray
import time


@ray.remote
class Counter:
    def __init__(self):
        self.i = 0

    def get(self):
        return self.i

    def incr(self, value):
        print(f"incrementing i={self.i} from remote...")
        self.i += value


if __name__ == "__main__":
    ray.init()

    c = Counter.remote()

    for _ in range(10):
        time.sleep(1)
        c.incr.remote(1)

    print(ray.get(c.get.remote()))
