from composer import Algorithm, Event


class RestrictedHoldOut(Algorithm):

    def __init__(self, num_subsample: int):
        self.num_subsample = num_subsample
    
    def match(self, event: Event):
