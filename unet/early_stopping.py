class EarlyStopping:
    def __init__(self, patience: int=5, minDelta: float=0):
        """
        Args:
            patience: Number of epochs without improvement after which training will stop
            min_delta: Minimum change that is considered an improvement
        """
        self.patience = patience
        self.minDelta = minDelta
        self.counter = 0
        self.bestLoss = None
        self.earlyStop = False

    def __call__(self, valLoss):
        if self.bestLoss is None:
            self.bestLoss = valLoss
        elif valLoss < self.bestLoss - self.minDelta:
            self.bestLoss = valLoss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.earlyStop = True

