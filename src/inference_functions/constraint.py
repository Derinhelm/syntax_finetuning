from abc import ABC, abstractmethod
import copy
import re
import string
from typing import List, Tuple
from collections import Counter


import torch

class Constraint(ABC):
    """Базовый класс для всех ограничений"""
    
    @abstractmethod
    def __call__(self, logits: torch.Tensor, processor, context) -> torch.Tensor:
        """
        Применяет ограничение к логитам
        
        Returns:
            измененные логиты
        """
        pass
       
    @abstractmethod
    def check(self, context):
        pass
         