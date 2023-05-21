from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, Dict


@dataclass
class ComboDLTransform:
    crop: Optional[Union[int, List[int], Tuple[int]]] = None # if not specified, no crop will occur
    mean: Optional[Union[float, List[float], Tuple[float]]] = 0.0
    std: Optional[Union[float, List[float], Tuple[float]]] = 1.0
    short_side_scale: Optional[int] = None # if not specified, no resize will occur
    pytorch_transforms: Optional[Dict] = None
    dali_transforms: Optional[Dict] = None

