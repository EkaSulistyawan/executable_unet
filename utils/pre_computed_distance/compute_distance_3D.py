
import torch
import numpy as np
###################################################### Parameters
# soon to be loaded by 
device      = 'cpu'
start_time  = 1144 # PA
end_time    = 1400 # PA
Fc          = 15.625e6     # in hertz                       
Fs          = 4*Fc 
imsz        = 72
r           = 1.0  # in mm 
cPA         = 1475 # in m #1475

# # depth calculation
# d = (3*(r**2))**0.5
# # PA
# start_time_PA  = np.floor((((sensor_pos.cpu().numpy()[:,0]**2).sum())**0.5 - d) * 1e-3 * Fs / cPA).astype(int) 
# end_time_PA    = np.ceil((((sensor_pos.cpu().numpy()[:,0]**2).sum())**0.5 + d) * 1e-3 * Fs / cPA).astype(int)
# # print(start_time_PA,end_time_PA)

# residue = (256 - (end_time_PA - start_time_PA))

# start_time_PA -= residue //2 
# end_time_PA += residue//2 + np.mod(residue,2)


class SaveTensor(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = torch.nn.Parameter(tensor)

def save_time_dist_PA():
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
    ri  = torch.stack([rix.flatten(),riy.flatten(),riz.flatten()],dim=-1)

    phys_dist_rcv = torch.cdist(ri,sensor_pos.T)
    time_points_distPA = ((phys_dist_rcv) * 1e-3 / cPA) * Fs - start_time

    module = SaveTensor(time_points_distPA)
    torch.jit.save(torch.jit.script(module),f"c{cPA}_{imsz}_{int(r)}mm.pt")

if __name__=="__main__":
    sensor_pos = torch.load("../../data/sensor_pos_interp.pt")
    save_time_dist_PA()