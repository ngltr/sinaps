from enum import Enum
from enum import auto


class Species(Enum):
    Ca = auto()
    K = auto()
    Na = auto()
    Buffer = auto()
    Buffer_Ca = auto()
    Anion = auto()
    Cation = auto()

    def __repr__(self):
        return "<Species {}>".format(self.__str__())

    def __str__(self):
        return "{}{}{}".format(
            self.name,
            abs(CHARGE[self]) if CHARGE[self] > 1 else "",
            "+" if CHARGE[self] > 0 else "-",
        )

    def __lt__(self, other):
        return self.name < other.name and CHARGE[self] < CHARGE[other] and DIFFUSION_COEF[self] < DIFFUSION_COEF[other] and INITIAL_CONCENTRATION[self] < INITIAL_CONCENTRATION[other]

    def __gt__(self, other):
        return self.name > other.name and CHARGE[self] > CHARGE[other] and DIFFUSION_COEF[self] > DIFFUSION_COEF[other] and INITIAL_CONCENTRATION[self] > INITIAL_CONCENTRATION[other]

    def __le__(self, other):
        return self.name <= other.name and CHARGE[self] <= CHARGE[other] and DIFFUSION_COEF[self] <= DIFFUSION_COEF[other] and INITIAL_CONCENTRATION[self] <= INITIAL_CONCENTRATION[other]

    def __ge__(self, other):
        return self.name >= other.name and CHARGE[self] >= CHARGE[other] and DIFFUSION_COEF[self] >= DIFFUSION_COEF[other] and INITIAL_CONCENTRATION[self] >= INITIAL_CONCENTRATION[other]

    def __ne__(self, other):
        return self.name != other.name and CHARGE[self] != CHARGE[other] and DIFFUSION_COEF[self] != DIFFUSION_COEF[other] and INITIAL_CONCENTRATION[self] != INITIAL_CONCENTRATION[other]


CHARGE = {
    Species.Ca: 2,
    Species.K: 1,
    Species.Na: 1,
    Species.Buffer: -2,
    Species.Buffer_Ca: 0,
}

DIFFUSION_COEF = {
    Species.Ca: 0.2,
    Species.K: 1.15,
    Species.Na: 1.15,
    Species.Buffer: 0.2,
    Species.Buffer_Ca: 0.2,
    Species.Anion: 0.2,
    Species.Cation: 0.2,
}

INITIAL_CONCENTRATION = {  # mMol/L
    Species.Ca: 0.2,
    Species.K: 0,
    Species.Na: 0,
    Species.Buffer: 0.2,
    Species.Buffer_Ca: 0,
    Species.Anion: 0.2,
    Species.Cation: 0.2,
}
