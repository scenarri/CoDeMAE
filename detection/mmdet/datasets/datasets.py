from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset

@DATASETS.register_module()
class DIORDataset(CocoDataset):
    METAINFO = {
        'classes':
            ('airplane', 'airport', 'baseballfield', 'basketballcourt',
               'bridge', 'chimney', 'dam', 'Expressway-Service-area',
               'Expressway-toll-station', 'golffield', 'groundtrackfield',
               'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
               'tenniscourt', 'trainstation', 'vehicle', 'windmill'),
        'palette':
            [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157)]
    }

@DATASETS.register_module()
class SARDet100K(CocoDataset):
    METAINFO = {
        'classes':
            ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'),
        'palette':
            [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100)]
    }
    
@DATASETS.register_module()
class SARAirCraftDataset(CocoDataset):
    METAINFO = {
        'classes':
            ('A220', 'A320/321', 'A330', 'ARJ21',
               'Boeing737', 'Boeing787', 'other'),
        'palette':
            [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100)]
    }

@DATASETS.register_module()
class SSDDDataset(CocoDataset):
    METAINFO = {
        'classes':
            ('ship'),
        'palette':
            [(220, 20, 60)]
    }
    
@DATASETS.register_module()
class HRSIDDataset(CocoDataset):
    METAINFO = {
        'classes':
            ('ship'),
        'palette':
            [(220, 20, 60)]
    }