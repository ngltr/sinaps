from enum import Enum
from enum import auto

class Species(Enum):
    Ca = auto()
    K = auto()
    Na = auto()
    Buffer = auto()
    BufferBinded = auto()


CHARGE = {      Species.Ca:2,
                Species.K:1,
                Species.Na:2,
                Species.Buffer:-2,
                Species.BufferBinded:0
                }

DIFFUSION_COEF = {
                Species.Ca:0.2,
                Species.K:0.2,
                Species.Na:0.2,
                Species.Buffer:0.2,
                Species.BufferBinded:0.2
                        }

INITIAL_CONCENTRATION = { #mMol/L
                Species.Ca:0.2,
                Species.K:0,
                Species.Na:0,
                Species.Buffer:0.2,
                Species.BufferBinded:0
                        }
