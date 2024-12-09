# Introduction

The following two files introduce basic usage of the `@metaflow_ray` decorator with `@kubernetes`.

1. `counter.py` contains an actor (class decorated with `@ray.remote`) called `Counter`
- The usage (/user code) of this counter is present in `__main__` and can be verified by running `python examples/basic_counter/counter.py`

2. `flow.py` shows how the usage (/user code) for this counter (after importing it from `counter.py`) can now be moved within the `execute` step of the `RayCounterFlow`.
- This flow can now be run with `python examples/basic_counter/flow.py --no-pylint --environment=pypi run`
- If you are on the [Outerbounds](https://outerbounds.com/) platform, you can leverage `fast-bakery` for blazingly fast docker image builds. This can be used by `python examples/basic_counter/flow.py --no-pylint --environment=fast-bakery run`
