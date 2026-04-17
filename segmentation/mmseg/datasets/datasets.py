from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DDHRSKDataset(BaseSegDataset):
    METAINFO = dict(classes=('Building', 'Road', 'Greenery', 'Water',
                             'Farmland'),
                    palette=[[79, 100, 127], [148, 160, 190], [163, 193, 165],
                             [20, 99, 191], [200, 197, 164]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs):
        super().__init__(img_suffix=img_suffix,
                         seg_map_suffix=seg_map_suffix,
                         reduce_zero_label=reduce_zero_label,
                         **kwargs)