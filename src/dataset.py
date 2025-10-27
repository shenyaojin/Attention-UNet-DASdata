import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

class DASDataset(Dataset):
    def __init__(self, imagepath, labelpath, IDlist, chann, dim):
        super(DASDataset, self).__init__()
        self.imagepath = imagepath
        self.labelpath = labelpath
        self.IDlist = IDlist
        self.chann = chann
        self.dim = dim
    
    def __getitem__(self, index):
        # Load image
        image = np.fromfile(self.imagepath + self.IDlist[index], dtype=np.single)
        
        # Determine original dimensions and reshape
        orig_dim = int(np.sqrt(image.shape[0] / self.chann))
        image = image.reshape(self.chann, orig_dim, orig_dim)
        
        # Convert to tensor for resizing
        image = torch.from_numpy(image).unsqueeze(0)  # Add batch dim: [1, chann, H, W]
        
        # Resize using interpolation
        image = F.interpolate(image, size=self.dim, mode='bilinear', align_corners=False)
        
        # Remove batch dim and normalize
        image = image.squeeze(0)
        image_max = torch.max(image)
        image_min = torch.min(image)
        if image_max > image_min:
            image -= image_min
            image /= (image_max - image_min)

        # Load label and perform the same resizing
        label = np.fromfile(self.labelpath + self.IDlist[index], dtype=np.single)
        label = label.reshape(self.chann, orig_dim, orig_dim)
        label = torch.from_numpy(label).unsqueeze(0)
        label = F.interpolate(label, size=self.dim, mode='nearest') # Use nearest-neighbor for labels
        label = label.squeeze(0)
        
        label = 2 * torch.clamp(label, 0, 1)

        return image, label

    def __len__(self):
        return len(self.IDlist)