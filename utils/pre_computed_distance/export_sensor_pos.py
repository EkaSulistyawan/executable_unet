import torch
import numpy as np

class SaveTensor(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = torch.nn.Parameter(tensor)

if __name__=="__main__":
    sensor_pos = torch.load("../../data/sensor_pos_interp.pt")
    module = SaveTensor(sensor_pos)
    torch.jit.save(torch.jit.script(module),f"sensor_pos_interp_jit.pt")