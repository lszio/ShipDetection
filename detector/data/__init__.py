from detectron2.data import DatasetCatalog
from .hrsc import register_hrsc_datasets
from .opensar import register_opensar_datasets


def register_datasets():
    DatasetCatalog.clear()
    register_hrsc_datasets()
    register_opensar_datasets()


if __name__ == "__main__":
    register_datasets()
