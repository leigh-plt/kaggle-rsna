from multiprocessing import Pool
import numpy as np
import pandas as pd
import cv2, os, glob
from dataset import RSNADataset

def extract_patient(csv_file_path, return_label=True):
    df = pd.read_csv(csv_file_path)

    df['PatientID'] = df['ID'].str.slice(start=3, stop=12)

    if return_label:
        df.drop_duplicates('ID', keep='last', inplace=True)
        df = df[df['PatientID'] != '6431af929'] # corrupted file in train set
        
    patient = df['PatientID'].values.reshape((-1,6), order='C')[:,0]
    
    if return_label:
        label = df.Label.values.reshape((-1,6), order='C')
        return patient, label
    else:
        return patient
    
patient, label = extract_patient('data/stage_1_train.csv', return_label=True)

###
# continue proccess if break - skip files if converted
# r = glob.glob('data/train_images/*.jpg')
# r = np.array([f.split('/')[-1].split('.')[0] for f in r])
#
###
ds = RSNADataset(patient, label, path='data/stage_1_train_images', transform=None)

def saver(index):
    if ds.patient[index] not in r:
        cv2.imwrite(os.path.join('data/train_images', ds.patient[index] + '.jpg'), ds.image(index))
    
if __name__ == '__main__':
    p = Pool(processes=18)
    print('Start convert files dicom to jpg')
    p.map(saver, range(len(ds)))