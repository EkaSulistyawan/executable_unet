
import torch
import numpy as np
###################################################### Parameters
# soon to be loaded by 
device      = 'cpu'
start_time  = None # PA
Fc          = 15.625e6     # in hertz                       
Fs          = 4*Fc 
imsz        = 72
r           = 1.0  # in mm 
cUS         = 1475 # in m #1475
#UseTXElements   = torch.tensor([4,17,30,43,53,65,77,89,107,118,129,140,151,160,169,178,182,190,198,206,215,221,227,233,241,246,251,256], dtype=torch.float32) - 1
UseTXElements   = torch.tensor([4,17,30,43,53,89,107,118,129,140,169,178,182,190,198,221,227,233,241,246], dtype=torch.float32) - 1
# UseTXElements   = torch.ones(10) * (246 - 1)


class SaveTensor(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = torch.nn.Parameter(tensor)

def save_time_dist_US():
    rix, riy, riz = np.meshgrid(
            np.linspace(-r, r, imsz),
            np.linspace(-r, r, imsz),
            np.linspace(-r, r, imsz),
            indexing='ij'
        )

    # with rotation
    theta = -np.pi / 4  # 45 degrees
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Perform 45-degree rotation in the xy-plane
    rix_rot = cos_theta * rix - sin_theta * riy
    riy_rot = sin_theta * rix + cos_theta * riy

    # Convert the rotated coordinates back to tensors
    rix = torch.tensor(rix_rot, dtype=torch.float32).to(device)
    riy = torch.tensor(riy_rot, dtype=torch.float32).to(device)

    # The z-coordinates remain unchanged
    riz = torch.tensor(riz,dtype=torch.float).to(device)
  
    # get only mid slices
    ri  = torch.stack([rix[:,imsz//2,:].flatten(),riy[:,imsz//2,:].flatten(),riz[:,imsz//2,:].flatten()],dim=-1)
    time_points_distUS_all = torch.zeros(256,len(UseTXElements),imsz*imsz)
    
    for i in range(len(UseTXElements)):
        tx = sensor_pos.T[i,:].unsqueeze(0)
        phys_dist_rcv = torch.cdist(ri,sensor_pos.T) + torch.cdist(ri,tx.repeat(256,1))
        time_points_distUS_all[:,i,:] = ((phys_dist_rcv.T) * 1e-3 / cUS) * Fs

    module = SaveTensor(time_points_distUS_all)
    torch.jit.save(torch.jit.script(module),f"USYslice_c{cUS}_{imsz}.pt")

if __name__=="__main__":
    d = (3*(r**2))**0.5
    sensor_pos = torch.load("../../data/sensor_pos.pt")
    start_time  = int(np.floor((((sensor_pos.cpu()[:,0]**2).sum())**0.5 - d) * 1e-3 * Fs / cUS)) - 1
    end_time    = int(np.ceil((((sensor_pos.cpu()[:,0]**2).sum())**0.5 + d) * 1e-3 * Fs / cUS))+ 1
    save_time_dist_US()