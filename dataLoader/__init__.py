from .llff import LLFFDataset
from .dtu import DTUDataset
from .your_own_data import YourOwnDataset
from .realestate10k import RealEstate10KDataset


dataset_dict = {'llff':LLFFDataset,
               'dtu':DTUDataset,
                'realestate10k':RealEstate10KDataset,
                'own_data':YourOwnDataset,}