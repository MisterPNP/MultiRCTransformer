import io
import os
import requests
import tarfile
import zipfile


cached_path = "./cached/"

mutlirc_url = "https://cogcomp.seas.upenn.edu/multirc/data/mutlirc-v2.zip"
multirc_path = cached_path + "multirc/"
multirc_dev_path = multirc_path + "splitv2/dev_83-fixedIds.json"
multirc_train_path = multirc_path + "splitv2/train_456-fixedIds.json"

coref_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
coref_path = cached_path + "coref-spanbert-large/"
dep_url = "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
dep_path = cached_path + "biaffine-dependency-parser-ptb/"

preprocessed_path = "./preprocessed/"
preprocessed_dev_path = preprocessed_path + "dev/"
preprocessed_train_path = preprocessed_path + "train/"


def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def download_and_extract(method, url, dest):
    if not os.path.exists(dest):
        print(f"downloading {url}")
        response = requests.get(url, stream=True)
        print(f"extracting to {dest}")
        file = method(response)
        file.extractall(path=dest)


def unzip(response):
    return zipfile.ZipFile(file=io.BytesIO(response.content))


def untar(response):
    return tarfile.open(fileobj=response.raw, mode="r|gz")


def setup():
    mkdir_if_not_exists(cached_path)
    mkdir_if_not_exists(preprocessed_path)
    mkdir_if_not_exists(preprocessed_dev_path)
    mkdir_if_not_exists(preprocessed_train_path)
    download_and_extract(unzip, mutlirc_url, multirc_path)
    download_and_extract(untar, coref_url, coref_path)
    download_and_extract(untar, dep_url, dep_path)
