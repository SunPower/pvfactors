from pvfactors.base import BaseSide


class PVGround(BaseSide):

    def __init__(self, list_segments=[]):
        super(PVGround, self).__init__(list_segments)
