from spinup.algos.vpg.vpg import VPGBuffer


class Buffer(VPGBuffer):

    @property
    def full(self):
        return self.ptr >= self.max_size
