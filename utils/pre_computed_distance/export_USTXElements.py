import torch
import numpy as np

class SaveTensor(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = torch.nn.Parameter(tensor)

if __name__=="__main__":
    UseTXElements   = torch.tensor([4,17,30,43,53,89,107,118,129,140,169,178,182,190,198,221,227,233,241,246], dtype=torch.float32) - 1
    UseTXElements   = torch.ones(10) * (246 - 1)
    module = SaveTensor(UseTXElements)
    torch.jit.save(torch.jit.script(module),f"UseTXElements_test_jit.pt")