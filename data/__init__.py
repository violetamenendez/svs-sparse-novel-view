from .dtu import MVSDatasetDTU
from .llff import LLFFDataset

dataset_dict = {'dtu': MVSDatasetDTU,
                'llff': LLFFDataset
                }