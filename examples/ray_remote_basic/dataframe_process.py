import ray
import pandas as pd
import numpy as np

# Initialize Ray if it's not already initialized
if not ray.is_initialized():
    ray.init()


@ray.remote
def process_dataframe_chunk(df_chunk, processing_function):
    """
    Processes a chunk of the DataFrame.

    Args:
        df_chunk (pd.DataFrame): A chunk of the DataFrame to process.
        processing_function (function): A function to apply to each chunk of the DataFrame.
                                         The function should take a pandas Series as input.
                                         Example: lambda x: x**2
    Returns:
        pd.DataFrame: Processed chunk of the DataFrame.
    """
    df_chunk["new_column"] = df_chunk["existing_column"].apply(processing_function)
    return df_chunk


def process_large_dataframe(df: pd.DataFrame, processing_function, num_chunks=10):
    """
    This function processes a large DataFrame in parallel using Ray.

    Args:
        df (pd.DataFrame): The input DataFrame to process.
        processing_function (function): A function that defines how to process the DataFrame.
                                         The function should take a pandas Series as input.
                                         Example: lambda x: x**2
        num_chunks (int): The number of chunks to split the DataFrame into for parallel processing.

    Returns:
        pd.DataFrame: The processed DataFrame with modifications.
    """
    # Split the DataFrame into chunks
    chunks = np.array_split(df, num_chunks)

    # Process each chunk in parallel using Ray
    result_ids = [
        process_dataframe_chunk.remote(chunk, processing_function) for chunk in chunks
    ]

    # Collect the processed chunks
    processed_chunks = ray.get(result_ids)

    # Concatenate the processed chunks back into a single DataFrame
    processed_df = pd.concat(processed_chunks)

    return processed_df


def do_something(num_chunks=10):

    # Create a large DataFrame (1 million rows as an example)
    df = pd.DataFrame({"existing_column": np.random.randint(1, 100, size=1000000)})

    # Define a custom processing function
    custom_function = lambda x: x**2

    # Process the DataFrame with the custom function
    processed_df = process_large_dataframe(df, custom_function, num_chunks=num_chunks)

    # Display the first few rows of the processed DataFrame
    print(processed_df.head())
