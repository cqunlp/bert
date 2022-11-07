# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
    some utils about dataset, such as download.
"""
import os
import re
import shutil
import tempfile
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import Union, Dict, List, Tuple, Optional

import requests
from requests import HTTPError

from pkg_resources import parse_version
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.ERROR)

DATASET_URL = {'SST-2': 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
               'CoLA': 'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
               'RTE': 'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
               'MNLI': 'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
               'STS-B': 'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
               'QQP': 'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
               'QNLI': 'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
               'AFQMC': "https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip",
               'WNLI': 'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',
               'DBpedia': 'https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz'}


def _get_dataset_url(name: str) -> str:
    """
    return the url of dataset for downloading.

    Args:
        name (str): Dataset name.

    Returns:
        str: The url of dataset for downloading.
    """
    url = DATASET_URL.get(name, None)
    if not url:
        raise KeyError(f"There is no {name}.")
    return url


def split_filename_suffix(filepath: str) -> Tuple[str, str]:
    """
    Return the corresponding name and suffix for a given filepath. If the suffix is multiple points,
    only.tar.gz is supported.

    Args:
        filepath (str): File path.

    Returns:
        Tuple[str, str]: The file name and suffix.
    """
    filename = os.path.basename(filepath)
    if filename.endswith('.tar.gz'):
        return filename[:-7], '.tar.gz'
    return os.path.splitext(filename)


def get_filepath(filepath: str) -> Path:
    """
    If filepath is directory:
        filepath will be returned if multiple files are included.
        filepath+filename will be returned if single file is included.
    If filepath is file:
        return filepath.

    Args:
        filepath (str): File path.

    Returns:
        Path: The file path or directory path.
    """
    if os.path.isdir(filepath):
        files = os.listdir(filepath)
        if len(files) == 1:
            filepath = os.path.join(filepath, files[0])
    else:
        raise FileNotFoundError(f"{filepath} is not a valid file or directory.")
    return filepath


def get_cache_path() -> str:
    """
    get the default cache path.

    Returns:
        str: Dataset cache path.
    """
    if 'DATASET_CACHE_DIR' in os.environ:
        dataset_cache_dir = os.environ.get('DATASET_CACHE_DIR')
        if not os.path.isdir(dataset_cache_dir):
            raise NotADirectoryError(f"{os.environ['DATASET_CACHE_DIR']} is not a directory.")
    else:
        dataset_cache_dir = os.path.expanduser(os.path.join("~", ".mindtext"))
    return dataset_cache_dir


def get_from_cache(url: List[str], cache_dir: Optional[Path] = None) -> Path:
    """
    Find the file defined by the url in cache.
    If not found,downloading from the url and placed under cache.

    Args:
        url (str): Dataset download url.
        cache_dir (Path, Optional): Cache directory path,default None.

    Returns:
        Path: The dataset path in cache directory.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dir_name = url[0]
    url = url[1]
    filename = re.sub(r".+/", "", url)
    _, suffix = split_filename_suffix(filename)
    cache_path = cache_dir / dir_name

    # Get cache path to put the file.
    if cache_path.exists():
        filepath = get_filepath(cache_path)

    if not cache_path.exists():
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        # GET file object.
        req = requests.get(url, stream=True, headers={"User-Agent": "fastNLP"})
        if req.status_code != 200:
            raise HTTPError(f"Status code:{req.status_code}. Fail to download from {url}.")
        success = False
        fd, temp_filename = tempfile.mkstemp()
        try:
            # Download.
            content_length = req.headers.get("Content-Length")
            progress = tqdm(unit="B", total=int(content_length) if content_length != 0 else None, unit_scale=1)
            logging.info("%s not found in cache, downloading to %s", url, temp_filename)

            with open(temp_filename, "wb") as temp_file:
                for chunk in req.iter_content(chunk_size=1024 * 16):
                    # filter out keep-alive new chunks.
                    if chunk:
                        progress.update(len(chunk))
                        temp_file.write(chunk)
            progress.close()
            logging.info("Finish download from %s", url)

            # Uncompress.
            if suffix in ('.zip', '.tar.gz', '.gz'):
                logging.debug("Start to uncompress file to %s", cache_dir)
                if suffix == '.zip':
                    unzip_file(Path(temp_filename), Path(cache_dir))
                elif suffix == '.gz':
                    ungzip_file(temp_filename, cache_dir, dir_name)
                else:
                    untar_gz_file(Path(temp_filename), Path(cache_dir))

                cache_path.mkdir(parents=True, exist_ok=True)
                logging.debug("Finish un-compressing file.")
            else:
                cache_path = str(cache_path) + suffix
            success = True
        except Exception as e:
            logging.error(e)
            raise e
        finally:
            if not success:
                if cache_path.exists():
                    if cache_path.is_file():
                        os.remove(cache_path)
                    else:
                        shutil.rmtree(cache_path)
            os.close(fd)
            os.remove(temp_filename)
        filepath = Path(get_filepath(cache_path))
    return filepath


def cached_path(url_or_filename: List[str], cache_dir: str = '', name: Optional[str] = None) -> Path:
    """
    Given a url,try to find the file under {cache_dir}/{name}/{filename} by parsing the filename from the url.
    1. If cache_dir=None, then cache_dir=~/.mindtext /; Otherwise the cache_dir = cache_dir.
    2. If name=None, there is no intermediate {name} layer; If not, the intermediate structure is {name}.

    If the file is available, the path is returned directly.
    If the file is not available, try to download it using the incoming URL.

    Args:
        url_or_filename (str): Dataset download url or dataset name.
        cache_dir (str, Optional): Cache directory path, default ``.
        name (str, Optional): The name of directory in cache path, default None.

    Returns:
        Path: The dataset path.
    """
    if cache_dir == '':
        data_cache = Path(get_cache_path())
    else:
        data_cache = cache_dir

    if name:
        data_cache = os.path.join(data_cache, name)

    parsed = urlparse(url_or_filename[1])

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary).
        filepath = get_from_cache(url_or_filename, Path(data_cache))
    elif parsed.scheme == "" and Path(os.path.join(data_cache, url_or_filename)).exists():
        # File, and it exists.
        filepath = Path(os.path.join(data_cache, url_or_filename))
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError(f'file {url_or_filename} not found in {data_cache}.')
    else:
        # Something unknown.
        raise ValueError(
            f"unable to parse {url_or_filename} as a URL or as a local path"
        )
    return filepath


def unzip_file(file: Path, to: Path):
    """
    Uncompress zip file.

    Args:
        file (Path): The zip file path.
        to (Path): The path of zip file will be compressed to.
    """
    # Unpack and write out in CoNLL column-like format.
    from zipfile import ZipFile

    with ZipFile(file, "r") as zip_obj:
        # Extract all the contents of zip file in current directory.
        zip_obj.extractall(to)


def ungzip_file(file: str, to: str, filename: str):
    """
    Uncompress gz file.

    Args:
        file (str): The gz file path.
        to (str): The path of gz file will be compressed to.
        filename (str): The name of gz file will be compressed to.
    """
    import gzip

    g_file = gzip.GzipFile(file)
    with open(os.path.join(to, filename), 'wb+') as f:
        f.write(g_file.read())
    g_file.close()


def untar_gz_file(file: Path, to: Path):
    """
    Uncompress tar file.

    Args:
        file (Path): The tar file path.
        to (Path): The path of tar file will be compressed to.
    """
    import tarfile

    with tarfile.open(file, 'r:gz') as tar:
        tar.extractall(to)


def check_loader_paths(paths: Union[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Check the validity of the files passed into the Dataset. If it is a legal path,
    a dict containing at least the key 'train' will be returned. Something like the following::
        {
            'train': '/some/path/to/'
            'test': 'xxx'
            ...
        }

    Args:
        paths (Union[str, Dict[str, str]]): Dataset file path or a dictionary of dataset path.

    Returns:
        Dict[str, str]: The dictionary of dataset path.
    """
    files = {}
    if isinstance(paths, (str, Path)):
        paths = os.path.abspath(os.path.expanduser(paths))
        if os.path.isfile(paths):
            files['train'] = paths
        elif os.path.isdir(paths):
            filenames = os.listdir(paths)
            filenames.sort()

            for filename in filenames:
                path_pair = None
                if 'train' in filename:
                    path_pair = ('train', filename)
                if 'dev' in filename:
                    if path_pair:
                        raise Exception(
                            f"Directory:{filename} in {paths} contains both `{path_pair[0]}` and `dev`.")
                    path_pair = ('dev', filename)
                if 'test' in filename:
                    if path_pair:
                        raise Exception(
                            f"Directory:{filename} in {paths} contains both `{path_pair[0]}` and `test`.")
                    path_pair = ('test', filename)
                if path_pair:
                    if path_pair[0] in files:
                        raise FileExistsError(f"Two files contain `{path_pair[0]}` were found, please specify the "
                                              f"filepath for `{path_pair[0]}`.")
                    files[path_pair[0]] = os.path.join(paths, path_pair[1])
            if 'train' not in files:
                raise KeyError(f"There is no train file in {paths}.")
        else:
            raise FileNotFoundError(f"{paths} is not a valid file path.")

    elif isinstance(paths, dict):
        if paths:
            if 'train' not in paths:
                raise KeyError("You have to include `train` in your dict.")
            for key, value in paths.items():
                if isinstance(key, str) and isinstance(value, str):
                    value = os.path.abspath(os.path.expanduser(value))
                    if not os.path.exists(value):
                        raise TypeError(f"{value} is not a valid path.")
                    files[key] = value
                else:
                    raise TypeError("All keys and values in paths should be str.")
        else:
            raise ValueError("Empty paths is not allowed.")
    else:
        raise TypeError(f"paths only supports str and dict. not {type(paths)}.")
    return files


def _split(text: str) -> List[str]:
    """
    Raw text tokenizer.

    Args:
        text (str): Raw text.

    Returns:
        List[str]: The tokens list.
    """
    text = text.strip()
    if not text:
        tokens = []
    else:
        tokens = text.split()
    return tokens


def _cn_split(text: str) -> List[str]:
    """
    Raw chinese text tokenizer.

    Args:
        text (str): Raw text.

    Returns:
        List[str]: The tokens list.
    """
    return [i for i in text]


def get_tokenizer(tokenize_method: str, lang: str = 'en'):
    """
    Get a tokenizer.

    Args:
        tokenize_method (str): Select a tokenizer method.
        lang (str, Optional): Tokenizer language(when using `spacy` tokenizer), default English.

    Returns:
        function: A tokenizer function.
    """
    tokenizer_dict = {
        'spacy': None,
        'raw': _split,
        'cn-char': _cn_split,
    }
    if tokenize_method == 'spacy':
        import spacy
        spacy.prefer_gpu()
        if lang != 'en':
            raise RuntimeError("Spacy only supports english")
        if parse_version(spacy.__version__) >= parse_version('3.0'):
            en = spacy.load('en_core_web_sm')
        else:
            en = spacy.load(lang)

        def _spacy_tokenizer(text):
            return [w.text for w in en.tokenizer(text)]

        tokenizer = _spacy_tokenizer
    elif tokenize_method in tokenizer_dict:
        tokenizer = tokenizer_dict[tokenize_method]
    else:
        try:
            # tokenizer = AutoTokenizer.from_pretrained(tokenize_method)
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        except BaseException:
            raise RuntimeError(
                f"'{tokenize_method}' should be a {tokenizer_dict.keys()} tokenizer or a pretrained model tokenizer.")
    return tokenizer


def _preprocess_sequentially(dataset_file_name: List[str]) -> List[str]:
    """
    Preprocess the dataset sequentially (train_dataset->dev_dataset->test_dataset).

    Args:
        dataset_file_name (List[str]): Dataset file list.

    Returns:
        List[str]: A sorted dataset file list.
    """
    dataset_type = ['train', 'dev', 'test']
    dataset_sort_file = []
    while dataset_file_name:
        for i in dataset_type:
            for j in dataset_file_name:
                if i in j:
                    dataset_sort_file.append(j)
                    dataset_file_name.remove(j)
    return dataset_sort_file


def _get_dataset_type(dataset_file_name: str) -> str:
    """
    Get the dataset type (train,dev,test).

    Args:
        dataset_file_name (str): Dataset file name.

    Returns:
        str: The type of dataset file.
    """
    dataset_type = ['train', 'dev', 'test']
    d_t = None
    for i in dataset_type:
        if i in dataset_file_name:
            d_t = i
            break
    return d_t


def get_split_func(data: pd.DataFrame, sep: str) -> callable:
    col_name = data.columns.values[0]

    def _split_func(row):
        row_data = row[col_name].strip().split(sep)
        return row_data

    return _split_func
