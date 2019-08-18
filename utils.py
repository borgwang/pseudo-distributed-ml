import gzip
import os
import pickle
from urllib.error import URLError
from urllib.request import urlretrieve


def prepare_dataset(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    path = os.path.join(data_dir, url.split("/")[-1])

    # download
    try:
        if os.path.exists(path):
            print("{} already exists.".format(path))
        else:
            print("Downloading {}.".format(url))
            try:
                urlretrieve(url, path)
            except URLError:
                raise RuntimeError("Error downloading resource!")
            finally:
                print()
    except KeyboardInterrupt:
        print("Interrupted")

    # load
    print("Loading MNIST dataset.")
    with gzip.open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")
