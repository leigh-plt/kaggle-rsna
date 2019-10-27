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
                                size=(512,512), window='Mixed', bins=256):
                    
        assert window.lower() in ['histogram', 'windows', 'mixed'], 'Unknown window type'
        self.path = path    
        self.transform = transform if transform else Identity()
        self.patient = patient
        self.label = label
        self.window = window.lower()
        self.quantiles = np.linspace(5e-3, 1., bins, endpoint=False)
        self.len = len(self.patient)
        self.size = size
        self.diapasons = {
            'B' : (40, 80),
            'S' : (80, 200),
            'F' : (40, 380)
        }

    def historgam(self, image, extend=True):
        """ Apply histogram hormalization to dicom image """

        bins = np.unique(np.quantile(image, self.quantiles))
        ys = np.linspace(0, 256, len(bins), endpoint=False)
        image = np.interp(image, bins, ys).astype(np.uint8)
        
        return image[:,:,np.newaxis] * np.ones((1,1,3), dtype=np.uint8) if extend else image

    @staticmethod
    def slice(data, center, width):

        image = data.pixel_array * data.RescaleSlope + data.RescaleIntercept
        image_min = center - width // 2
        image_max = center + width // 2
        image = np.clip(image, image_min, image_max)

        # rescale to 0..255
        image = ((image - image_min) * 255 / width).clip(0, 255).astype(np.uint8)
        return image

    def windows(self, data, interval='BSF'):
        r""" Apply windows selecten to dicom data.

        Argue:
            data: dicom data
            interval, string: returned windows, example: 'BS',
                              where (type - (center, width)):
                                    B - brain (40, 80)
                                    S - subdural (80, 200)
                                    F - soft (40, 380)
        Return:
            Stacked windows

        """

        image = np.stack([
                self.slice(data, *self.diapasons[iv]) for iv in interval
            ], axis = -1)

        return image

    def mixed(self, data):
        """ Return mixed layer to dicom data """
        image = np.concatenate([
                self.historgam(data.pixel_array, False)[:,:,None],
                self.windows(data, 'BS')
            ], axis = 2)
        return image

    @staticmethod
    def correct_dcm(data):
        """
            Rescale dicom data for typical format
            https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
        """
        if data.PixelRepresentation != 0 or data.RescaleIntercept<-100:
            return data
            
        scan = data.pixel_array + 1000
        px_mode = 4096
        scan = np.where(scan >= px_mode, scan - px_mode, scan)
        data.PixelData = scan.tobytes()
        data.RescaleIntercept = -1000

        return data 

    def image(self, index):
        
        """ Load and extend single dimention to three """
        try:
            data = pydicom.read_file(os.path.join(
                                self.path,'ID_'+self.patient[index]+ '.dcm'))
            image = data.pixel_array 
        except:
            print("Oops! Something wrong with this file: {}".format(self.patient[index]))
            raise ValueError
        data = self.correct_dcm(data)
        
        if self.window == 'histogram':
            image = self.historgam(data.pixel_array)
        elif self.window == 'windows':
            image = self.windows(data)
        else:
            image = self.mixed(data)

        ## apply resize for standartize image resolution
        if image.shape != (512,512):
            image = cv2.resize(image, self.size, interpolation = cv2.INTER_AREA)
 
        return image
        
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