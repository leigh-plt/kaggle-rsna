import numpy as np
import pydicom, os, cv2

import torch
import torch.utils.data as D

class Identity():
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    """
    def __init__(self, *args, **kwargs):
        pass
    
    def __repr__(self):
        format_string = self.__class__.__name__ 
        return format_string
    
    def __call__(self, input):
        return input

class RSNADataset(D.Dataset):
    def __init__(self, patient, label=None, path='input', transform=None,
                                size=(512,512), hist_norm=True, bins=256):
                    
        self.path = path    
        self.transform = transform if transform else Identity()
        self.patient = patient
        self.label = label
        self._hist_norm = hist_norm
        self.quantiles = np.linspace(5e-3, 1., bins, endpoint=False)
        self.len = len(self.patient)
        self.size = size

    def hist_norm(self, image):
        """ Apply histogram hormalization to dicom image """

        bins = np.unique(np.quantile(image, self.quantiles))
        ys = np.linspace(0, 256, len(bins), endpoint=False)
        return np.interp(image, bins, ys).astype(np.uint8)

    def image(self, index):
        
        """ Load and extend single dimention to three """
        try:
            image = pydicom.read_file(os.path.join(
                                self.path,'ID_'+self.patient[index]+ '.dcm'))
            image = image.pixel_array 
        except:
            print("Oops! Something wrong with this file: {}".format(self.patient[index]))
            raise ValueError

        if self._hist_norm: image = self.hist_norm(image)

        ## apply resize for standartize image resolution
        if image.shape != (512,512):
            image = cv2.resize(image, self.size, interpolation = cv2.INTER_AREA)

        ## extend image to 3 dimention    
        return image[:,:,np.newaxis] * np.ones((1,1,3), dtype=np.uint8)
        
    def __getitem__(self, index):
        """ Return Image and Label or Image and ParientID if label is None """
        
        image = self.image(index)
        image = self.transform(image)

        if self.label is not None:
            return image, self.label[index].astype(np.float32)
        else:
            return image, self.patient[index]
            
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len