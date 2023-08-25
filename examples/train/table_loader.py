from metaflow import S3, profile
from concurrent.futures import ThreadPoolExecutor


def _print_throughput(caption, stats, total_size):
    gbps = (total_size * 8) / (next(iter(stats.values())) / 1000.0)
    print("%s: %2.1f Gbit/s" % (caption, gbps))


def _load_s3(url, s3, num_files):
    files = list(s3.list_recursive([url]))[:num_files]
    total_size = sum(f.size for f in files) / 1024**3
    print("Loading %2.1dGB of data" % total_size)
    stats = {}

    with profile("download", stats_dict=stats):
        loaded = s3.get_many([f.url for f in files])
    _print_throughput("S3->EC2 download", stats, total_size)

    return [f.path for f in loaded], total_size


def load_table(
    url,
    num_files,
    num_threads=8,
    only_download=False,
    reduce_precision=True,
    drop_cols=["row_id"],
):
    import pyarrow.parquet as pq
    import pyarrow as pa

    stats = {}
    with profile("read", stats_dict=stats):
        with S3() as s3:
            fls, total_size = _load_s3(url, s3, num_files)
            if only_download:
                return
            with ThreadPoolExecutor(max_workers=num_threads) as exe:
                tables = exe.map(lambda f: pq.read_table(f, use_threads=False), fls)
                table = pa.concat_tables(tables)

    _print_throughput(
        "Decoding throughput from S3 to PyArrow tables", stats, total_size
    )

    if reduce_precision:
        for i, (col_name, type_) in enumerate(
            zip(table.schema.names, table.schema.types)
        ):
            if pa.types.is_decimal(type_):
                table = table.set_column(
                    i, col_name, pac.cast(table.column(col_name), pa.float32())
                )

    if len(drop_cols) > 0:
        table = table.drop(drop_cols)

    return table
