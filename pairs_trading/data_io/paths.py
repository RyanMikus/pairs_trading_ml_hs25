from pathlib import Path


def as_path(data_dir):
    return data_dir if isinstance(data_dir, Path) else Path(data_dir)


def _strip_parquet_suffix(basename: str) -> str:
    return basename[:-8] if basename.endswith(".parquet") else basename


def unprocessed_files(data_dir, pattern: str):
    data_dir_p = as_path(data_dir)
    return sorted(data_dir_p.glob(pattern))


def parquet_path(data_dir, basename: str) -> Path:
    data_dir_p = as_path(data_dir)
    base = _strip_parquet_suffix(basename)
    return data_dir_p / f"{base}.parquet"


def metadata_path(data_dir, basename: str) -> Path:
    data_dir_p = as_path(data_dir)
    base = _strip_parquet_suffix(basename)
    if base.endswith("_metadata"):
        base = base[:-9]
    return data_dir_p / f"{base}_metadata.json"
