from detectron2.data import DatasetCatalog, MetadataCatalog


class DataVisualizer():
    def __init__(self, datasets):
        if isinstance(datasets, str):
            datasets = [datasets]
        self.datasets = datasets
        self.thing_classes = MetadataCatalog.get(datasets[0]).thing_classes
        for name in datasets[1:]:
            assert self.thing_classes == MetadataCatalog.get(
                name).thing_classes
        self.result = self._count()

    def _count(self):
        count = {}
        for dataset in self.datasets:
            part = dataset.split('_')[-1]
            imgs = []
            for record in DatasetCatalog.get(dataset):
                anns = record['annotations']
                imgs.append(len(anns))
            count[part] = {
                'img_num': len(imgs),
                'item_num': sum(imgs)
            }
        return count

    def draw_plot(self):
        pass


if __name__ == '__main__':
    from hrsc import register_hrsc_datasets
    register_hrsc_datasets()
    name = 'hrsc_1_p'
    dv = DataVisualizer(
        [name + "_" + part for part in ['train', 'test']])
    print(dv.result)
