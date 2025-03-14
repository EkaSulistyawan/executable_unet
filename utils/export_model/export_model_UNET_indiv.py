import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import os

from torch.nn.functional import interpolate

import torch
import torch.nn as nn
import torch.nn.functional as F

import struct
import argparse

import time

import cv2

import torch.onnx


# soon to be loaded by 
device      = 'cpu'
start_time  = 1144 # PA
end_time    = 1400 # PA
Fc          = 15.625e6     # in hertz                       
Fs          = 4*Fc 
imsz        = 128
r           = 1.5  # in mm 
cPA         = 1475 # in m #1475


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate    

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)#
    
def sine_init(m, omega=30):
    if isinstance(m, nn.Conv2d):
        with torch.no_grad():
            m.weight.uniform_(-1/omega, 1/omega)

def split_into_patches(image: torch.Tensor, patch_size: tuple) -> torch.Tensor:
    """
    Splits an image into non-overlapping patches and treats each patch as a batch.
    
    Args:
        image (torch.Tensor): Input tensor of shape (Batch, Channel, Height, Width).
        patch_size (tuple): Tuple indicating the size of each patch (Height, Width).
        
    Returns:
        torch.Tensor: Patches of shape (NumPatches, Channels, PatchHeight, PatchWidth).
    """
    # Calculate padding
    _, _, height, width = image.shape
    pad_h = (patch_size[0] - height % patch_size[0]) % patch_size[0]
    pad_w = (patch_size[1] - width % patch_size[1]) % patch_size[1]

    # Pad the image
    padded_image = F.pad(image, (0, pad_w, 0, pad_h))  # Padding: (Left, Right, Top, Bottom)

    # Use nn.Unfold to extract patches
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    patches = unfold(padded_image)  # Shape: (Batch, PatchSize*PatchSize*Channels, NumPatches)

    # Reshape to make each patch a batch
    patches = patches.permute(0, 2, 1)  # Move patches to the second dimension
    patches = patches.view(-1, image.shape[1], patch_size[0], patch_size[1])  # (NumPatches, Channels, PatchHeight, PatchWidth)

    return patches

def reconstruct_from_patches(patches: torch.Tensor, original_size: tuple, patch_size: tuple) -> torch.Tensor:
    """
    Reconstructs the original image from non-overlapping patches.
    
    Args:
        patches (torch.Tensor): Patches of shape (NumPatches, Channels, PatchHeight, PatchWidth).
        original_size (tuple): Original size of the image (Height, Width).
        patch_size (tuple): Tuple indicating the size of each patch (PatchHeight, PatchWidth).
        
    Returns:
        torch.Tensor: Reconstructed image of shape (Batch, Channels, Height, Width).
    """
    # Get the number of patches per dimension
    batch_size = 1  # Assuming patches came from one image
    channels = patches.shape[1]
    padded_height = (original_size[0] + patch_size[0] - 1) // patch_size[0] * patch_size[0]
    padded_width = (original_size[1] + patch_size[1] - 1) // patch_size[1] * patch_size[1]
    
    # Reshape patches back to the unfolded shape
    num_patches_h = padded_height // patch_size[0]
    num_patches_w = padded_width // patch_size[1]
    patches = patches.reshape(batch_size, num_patches_h * num_patches_w, -1).permute(0, 2, 1)

    # Reconstruct the padded image using Fold
    fold = torch.nn.Fold(output_size=(padded_height, padded_width), kernel_size=patch_size, stride=patch_size)
    reconstructed_image = fold(patches)

    # Remove padding to restore original dimensions
    reconstructed_image = reconstructed_image[:, :, :original_size[0], :original_size[1]]
    return reconstructed_image
    
class DNNSinogramSR(nn.Module):

    def __init__(self, activ_func1, activ_func2, f = 32):

        super(DNNSinogramSR, self).__init__()
        # print('Old')
        # print('\n')
        self.f = f
        activ_func = activ_func1
        ksize = 3
        padsize = 1
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.f,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f, out_channels=self.f,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        

        # Conv block 2 - Down 2
        self.conv2_block = nn.Sequential(
            nn.Conv2d(in_channels=f, out_channels=self.f*2,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*2, out_channels=self.f*2,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        

        # Conv block 3 - Down 3
        self.conv3_block = nn.Sequential(
            nn.Conv2d(in_channels=self.f*2, out_channels=self.f*4,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*4, out_channels=self.f*4,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        

        # Conv block 4 - Down 4
        self.conv4_block = nn.Sequential(
            nn.Conv2d(in_channels=self.f*4, out_channels=self.f*8,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*8, out_channels=self.f*8,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        

        # Conv block 5 - Down 5
        self.conv5_block = nn.Sequential(
            nn.Conv2d(in_channels=self.f*8, out_channels=self.f*16,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*16, out_channels=self.f*16,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )
        
        

        # Up 1
        self.up_1 = nn.ConvTranspose2d(in_channels=self.f*16, out_channels=self.f*8, kernel_size=2, stride=2)

        # Up Conv block 1
        self.conv_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.f*16, out_channels=self.f*8,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*8, out_channels=self.f*8,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )

        # Up 2
        self.up_2 = nn.ConvTranspose2d(in_channels=self.f*8, out_channels=self.f*4, kernel_size=2, stride=2)

        # Up Conv block 2
        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.f*8, out_channels=self.f*4,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*4, out_channels=self.f*4,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )

        # Up 3
        self.up_3 = nn.ConvTranspose2d(in_channels=self.f*4, out_channels=self.f*2, kernel_size=2, stride=2)

        # Up Conv block 3
        self.conv_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.f*4, out_channels=self.f*2,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*2, out_channels=self.f*2,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )

        # Up 4
        self.up_4 = nn.ConvTranspose2d(in_channels=self.f*2, out_channels=self.f, kernel_size=2, stride=2)

        # Up Conv block 4
        self.conv_up_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.f*2, out_channels=self.f,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f, out_channels=self.f,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )

        # Final output
        # self.conv_final = nn.Conv2d(in_channels=32, out_channels=2,
        #                             kernel_size=1, padding=0, stride=1)

        self.conv_final = nn.Sequential(
            nn.Conv2d(in_channels=self.f, out_channels=1, # use channels 2 for positive negative output
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func2,
            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func2,
        )

    def forward(self, x):
        # print('input', x.shape)

        # Down 1
        x = self.conv1_block(x)
        # print('after conv1', x.shape)
        conv1_out = x  # Save out1
        conv1_dim = x.shape[2]
        x = self.max1(x)
        # print('before conv2', x.shape)

        # Down 2
        x = self.conv2_block(x)
        # print('after conv2', x.shape)
        conv2_out = x
        conv2_dim = x.shape[2]
        x = self.max2(x)
        # print('before conv3', x.shape)

        # Down 3
        x = self.conv3_block(x)
        # print('after conv3', x.shape)
        conv3_out = x
        conv3_dim = x.shape[2]
        x = self.max3(x)
        # print('before conv4', x.shape)

        # Down 4
        x = self.conv4_block(x)
        # print('after conv5', x.shape)
        conv4_out = x
        conv4_dim = x.shape[2]
        x = self.max4(x)
        # print('after conv4', x.shape)

        # Midpoint
        x = self.conv5_block(x)
        # print('mid', x.shape)

        # Up 1
        x = self.up_1(x)
        # print('up_1', x.shape)
        lower = int((conv4_dim - x.shape[2]) / 2)
        upper = int(conv4_dim - lower)
        conv4_out_modified = conv4_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv4_out_modified], dim=1)
        # print('after cat_1', x.shape)
        x = self.conv_up_1(x)
        # print('after conv_1', x.shape)

        # Up 2
        x = self.up_2(x)
        # print('up_2', x.shape)
        lower = int((conv3_dim - x.shape[2]) / 2)
        upper = int(conv3_dim - lower)
        conv3_out_modified = conv3_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv3_out_modified], dim=1)
        # print('after cat_2', x.shape)
        x = self.conv_up_2(x)
        # print('after conv_2', x.shape)

        # Up 3
        x = self.up_3(x)
        # print('up_3', x.shape)
        lower = int((conv2_dim - x.shape[2]) / 2)
        upper = int(conv2_dim - lower)
        conv2_out_modified = conv2_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv2_out_modified], dim=1)
        # print('after cat_3', x.shape)
        x = self.conv_up_3(x)
        # print('after conv_3', x.shape)

        # Up 4
        x = self.up_4(x)
        # print('up_4', x.shape)
        lower = int((conv1_dim - x.shape[2]) / 2)
        upper = int(conv1_dim - lower)
        conv1_out_modified = conv1_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv1_out_modified], dim=1)
        # print('after cat_4', x.shape)
        x = self.conv_up_4(x)
        # print('after conv_4', x.shape)

        # Final output
        x = self.conv_final(x)

        # if positive 2 channels
        # if (x.shape[1] == 2):
            # x = -x[:,0,:,:] + x[:,1,:,:]
            # x = x.unsqueeze(1)

        return x

def load_vrs(fileName,nt,ntx=28,verbose=False):
    headerInfo = {}
    with open(fileName, 'rb') as file_obj:
        # Read Version (uint16)
        version = struct.unpack('3B', file_obj.read(3))[0]  # Read 2 bytes for uint16
        headerInfo['version'] = version
        # Compression (uint8)
        compression = struct.unpack('B', file_obj.read(1))[0]
        headerInfo['compression'] = compression

        timetagflag = struct.unpack('B', file_obj.read(1))[0]
        headerInfo['timetagflag'] = timetagflag

    #     so no timetagflag
        if timetagflag == 1:
            time_tag = struct.unpack('6B', file_obj.read(6))  # 6 bytes: Sec, Min, Hour, Day, Month, Year
            headerInfo['time_tag'] = time_tag

        studyidlength = struct.unpack('<Q', file_obj.read(8))[0]
        headerInfo['studyidlength'] = studyidlength
        studyid = file_obj.read(studyidlength).decode('utf-8')
        headerInfo['studyid'] = studyid

        sampleidlength = struct.unpack('<Q', file_obj.read(8))[0]
        headerInfo['sampleidlength'] = sampleidlength
        sampleid = file_obj.read(sampleidlength).decode('utf-8')
        headerInfo['sampleid'] = sampleid

        commentlength = struct.unpack('<Q', file_obj.read(8))[0]
        headerInfo['commentlength'] = commentlength
        comment = file_obj.read(commentlength).decode('utf-8')
        headerInfo['comment'] = comment

        dim = struct.unpack('4Q', file_obj.read(32))  # 4 uint64 values
        headerInfo['dim'] = dim

        numdatapoints = struct.unpack('Q', file_obj.read(8))[0]
        headerInfo['numdatapoints'] = numdatapoints

        #print(file_obj.read(1))
        datatypes = struct.unpack('B', file_obj.read(1))[0]
        headerInfo['datatypes'] = datatypes

        # not sure what is this dtypes version
        # in the actual extraction, it shows S16, probably signed 16

        data = np.frombuffer(file_obj.read(numdatapoints * 2), dtype='<h')  # 2 bytes per int16
        # plt.imshow(np.reshape(data[0:(3072*58)],(3072,58)))
        # plt.gca().set_aspect("auto")
        # plt.show()
        data2 = np.reshape(data ,(dim[1],dim[0])) #  the initial data is flattened time with the amount of TX, for each sensor, they store the data in 4096 format
        data2 = np.reshape(data2[:,:ntx*nt],(dim[1],ntx,nt))

    if verbose:
        for key in headerInfo.keys():
            print(key,headerInfo[key])
    return data2

def load_vrs_data(pathFileName):
    data = load_vrs(pathFileName,3072,verbose=False)
    data = data.transpose(1,2,0)
    data = np.expand_dims(data,-1)
    # data should be (nsensor, ntime)

    data = data[0,start_time:end_time,:].squeeze().T

    # play around here 
    return data

def recon_z(zidx):
    rix, riy = np.meshgrid(
        np.linspace(-r, r, imsz),
        np.linspace(-r, r, imsz),
        indexing='ij'
    )
    rix = torch.tensor(rix, dtype=torch.float32).to(device)
    riy = torch.tensor(riy, dtype=torch.float32).to(device)
    riz = torch.ones_like(rix) * zidx
    ri  = torch.stack([rix.flatten(),riy.flatten(),riz.flatten()],dim=-1)

    phys_dist_rcv = torch.cdist(ri,sensor_pos.T)
    time_points_distPA = ((phys_dist_rcv) * 1e-3 / cPA) * Fs - start_time
    linear_factor = (time_points_distPA -  torch.floor(time_points_distPA))

    rows = torch.arange(sensor_pos.shape[1]).unsqueeze(1)  # Shape: (256, 1)
    resultlower = outp[rows, torch.floor(time_points_distPA.T).to(torch.int32)]
    resultupper = outp[rows, torch.ceil(time_points_distPA.T).to(torch.int32)]

    result = resultlower + linear_factor.T * (resultupper - resultlower)
    result = result.sum(0).reshape(imsz,imsz).abs()

    return result

class SaveTensor(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = torch.nn.Parameter(tensor)

def save_time_dist_PA(zidx,whichdim):
    ria, rib = np.meshgrid(
        np.linspace(-r, r, imsz),
        np.linspace(-r, r, imsz),
        indexing='ij'
    )
    all_ones = torch.ones_like(torch.tensor(ria, dtype=torch.float32).to(device)) * zidx
    if whichdim == 'x':
        rix = all_ones 
        riy = torch.tensor(ria, dtype=torch.float32).to(device)
        riz = torch.tensor(rib, dtype=torch.float32).to(device)
    elif whichdim == 'y':
        rix = torch.tensor(ria, dtype=torch.float32).to(device) 
        riy = all_ones
        riz = torch.tensor(rib, dtype=torch.float32).to(device)
    elif whichdim == 'z':
        rix = torch.tensor(ria, dtype=torch.float32).to(device) 
        riy = torch.tensor(rib, dtype=torch.float32).to(device)
        riz = all_ones

    ri  = torch.stack([rix.flatten(),riy.flatten(),riz.flatten()],dim=-1)

    phys_dist_rcv = torch.cdist(ri,sensor_pos.T)
    time_points_distPA = ((phys_dist_rcv) * 1e-3 / cPA) * Fs - start_time

    module = SaveTensor(time_points_distPA)
    torch.jit.save(torch.jit.script(module),f".\dist_dict\c{cPA}_{whichdim}{zidx:.2f}.pt")



import os
if __name__=="__main__":
    net = DNNSinogramSR(Sine(),Sine(),f=128).to(device)
    net.load_state_dict(torch.load('modelNoFK_test_4_f128_mean50um.pth', weights_only=True,map_location=device))
    net.eval()

    traced_model = torch.jit.script(net)
    traced_model.save('modelNoFK_test_4_f128_mean50um.pt')

