



class RetinaPreprocessingBlock(nn.Module):

    def __init__(self, channels: int, temportal_alpha: float = 0.2) -> None:
        super().__init__()