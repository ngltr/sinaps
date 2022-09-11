from enum import Enum
from enum import auto


class Species(Enum):
    Ca = auto()
    K = auto()
    Na = auto()
    Buffer = auto()
    Buffer_Ca = auto()
    Cl = auto()
    Mg = auto()
    Anion = auto()

    def __repr__(self):
        return "<Species {}>".format(self.__str__())

    def __str__(self):
        return "{}{}{}".format(
            self.name,
            abs(CHARGE[self]) if CHARGE[self] > 1 else "",
            "+" if CHARGE[self] > 0 else "-",
        )


CHARGE = {
    Species.Ca: 2,
    Species.K: 1,
    Species.Na: 1,
    Species.Buffer: -2,
    Species.Buffer_Ca: 0,
    Species.Cl: -1,
    Species.Mg: 2,
    Species.Anion: -1,
}

DIFFUSION_COEF = {
    Species.Ca: 0.2,
    Species.K: 1.15,
    Species.Na: 1.15,
    Species.Buffer: 0.2,
    Species.Buffer_Ca: 0.2,
    Species.Cl: 0.2,
    Species.Mg: 0.2,
    Species.Anion: 0.2,
}

INITIAL_CONCENTRATION = {  # mMol/L
    Species.Ca: 1e-4,
    Species.K: 140,
    Species.Na: 12,
    Species.Buffer: 2e-3,
    Species.Buffer_Ca: 0,
    Species.Cl: 150,
    Species.Mg: 0.5,
    Species.Anion: 0,
}
