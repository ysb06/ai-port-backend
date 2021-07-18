from torch import nn

class TRADE(nn.Module):
    def __init__(self):
        super(TRADE, self).__init__()
        self.encoder = EncoderRNN()
        self.decoder = Generator()

class EncoderRNN(nn.Module):
    pass

class Generator(nn.Module):
    pass