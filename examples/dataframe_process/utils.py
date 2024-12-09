import ray
import numpy as np
import pandas as pd
from typing import Callable


@ray.remote
def process_dataframe_chunk(df_chunk: pd.DataFrame, processing_function: Callable):
    df_chunk["new_column"] = df_chunk["existing_column"].apply(processing_function)
    return df_chunk


def process_dataframe(df: pd.DataFrame, processing_function: Callable, num_chunks: int = 10):
    chunks = np.array_split(df, num_chunks)
    result_futs = [process_dataframe_chunk.remote(each_chunk, processing_function) for each_chunk in chunks]
    processed_chunks = ray.get(result_futs)
    processed_df = pd.concat(processed_chunks)
    return processed_df


if __name__ == "__main__":
    sample_df = pd.DataFrame({"existing_column": np.random.randint(1, 100, size=1000000)})
    custom_function = lambda x: x**2
    processed_df = process_dataframe(sample_df, custom_function, num_chunks=10)
    print(processed_df.head())
