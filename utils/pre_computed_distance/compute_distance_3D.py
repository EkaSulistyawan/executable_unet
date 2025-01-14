
import torch
import numpy as np
###################################################### Parameters
# soon to be loaded by 
device      = 'cpu'
start_time  = 1144 # PA
end_time    = 1400 # PA
Fc          = 15.625e6     # in hertz                       
Fs          = 4*Fc 
imsz        = 128
r           = 1.5  # in mm 
cPA         = 1475 # in m #1475


class SaveTensor(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = torch.nn.Parameter(tensor)

def save_time_dist_PA(zidx,whichdim):
    rix, riy, riz = torch.meshgrid(
        torch.linspace(-r, r, imsz),
        torch.linspace(-r, r, imsz),
        torch.linspace(-r, 0, 10),
        indexing='ij'
    )

    ri  = torch.stack([rix.flatten(),riy.flatten(),riz.flatten()],dim=-1)

    phys_dist_rcv = torch.cdist(ri,sensor_pos.T)
    time_points_distPA = ((phys_dist_rcv) * 1e-3 / cPA) * Fs - start_time

    module = SaveTensor(time_points_distPA)
    torch.jit.save(torch.jit.script(module),f"c{cPA}_3D.pt")

if __name__=="__main__":
    sensor_pos = torch.load("../../data/sensor_pos_interp.pt")
    for i in np.linspace(-1.0, 1.0, 11):
        for j in ['x','y','z']:
            save_time_dist_PA(i,j)