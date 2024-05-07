import json
import pickle as pkl
from pathlib import Path
from typing import Union

import jsonlines
import zstandard as zstd

LINE_CHANGE = "\n"


def read_lines(path, rm_empty_lines=False, strip=False):
    with open(path, "r") as f:
        lines = f.readlines()
    if strip:
        lines = [line.strip() for line in lines]
    if rm_empty_lines:
        lines = [line for line in lines if len(line.strip()) > 0]
    return lines


def write_lines(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        lines = [
            line if line.endswith(LINE_CHANGE) else f"{line}{LINE_CHANGE}"
            for line in lines
        ]
        f.writelines(lines)


def read_jsonlines(path):
    with jsonlines.open(path) as reader:
        samples = list(reader)
    return samples


def write_jsonlines(path, samples):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        writer.write_all(samples)


def read_json(path, not_found_ok=False):
    if not_found_ok and not Path(path).exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_text(path):
    with open(path, "r") as fd:
        data = fd.read()
    return data


def write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fd:
        fd.write(text)


def read_pickle(path):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return data


def write_pickle(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fd:
        pkl.dump(obj=obj, file=fd)


def compress(
    content: Union[str, bytes],
    compression: str = "zstd",
    compress_level: int = 3,
):

    if type(content) == str:
        content = content.encode()
    if compression == "zstd":
        compressed_content = zstd.ZstdCompressor(level=compress_level).compress(
            content
        )
    else:
        compressed_content = content  # No compression
    return compressed_content
