class Hook:
    def callback(self, optimizer):
        raise NotImplementedError()


class NoNewSpeciesHook(Hook):
    def __init__(self, n_no_new_species):
        self.n_no_new_species = n_no_new_species

        self.count = n_no_new_species
        self.discovered_species = set()

    def callback(self, optimizer):
        ancestors = [s.ancestor for s in optimizer.species]

        before = len(self.discovered_species)
        self.discovered_species.update(ancestors)
        after = len(self.discovered_species)

        if after > before:
            self.count = self.n_no_new_species
        else:
            self.count -= 1
            if self.count < 0:
                return ['end']

        return []
