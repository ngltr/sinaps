from enum import Enum

class Species(Enum):
    Ca = 2
    K = 2
    Na = 2
    Buffer = -2
    BufferBinded = 0


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
