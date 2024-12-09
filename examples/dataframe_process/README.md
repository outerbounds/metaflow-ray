# Introduction

The following two files showcase how to process large dataframes using the `@metaflow_ray` decorator with `@kubernetes`.

1. `utils.py` contains a remote function called `process_dataframe_chunk` which is used in `process_dataframe`.
- The usage (/user code) of this is present in `__main__` and can be verified by running `python examples/dataframe_process/utils.py`

2. `flow.py` shows how the usage (/user code) for `process_dataframe` (after importing it from `utils.py`) can now be moved within the `execute` step of the `RayDFProcessFlow`.
- This flow can now be run with `python examples/dataframe_process/flow.py --no-pylint --environment=pypi run`
- If you are on the [Outerbounds](https://outerbounds.com/) platform, you can leverage `fast-bakery` for blazingly fast docker image builds. This can be used by `python examples/dataframe_process/flow.py --no-pylint --environment=fast-bakery run`
