import numpy as np
import pandas as pd
from collections import deque

import torchvision, torch

class MovingAverage():

    def __init__(self, ma_type='simple', maxlen=20, alpha=0.7):

        assert ma_type in ['simple', 'weighted', 'exponent']
        assert 0 < alpha < 1, 'alpha in range (0..1)'

        self.ma_type = ma_type
        self.values = deque(maxlen=maxlen)

    def mean(self):
        """ Return moving average mean """

        if len(self.values) == 0:
            # raise ValueError
            return 'Empty'

        if self.ma_type == 'simple':
            return np.array(self.values).mean()

        elif self.ma_type == 'weighted':
            weights = np.arange(1, len(self.values)+1)
            return sum(np.array(self.values) * weights ) / weights.sum()

        else:
            weights = self.alpha**np.arange(len(self.values), 0, -1)
            return sum(np.array(self.values) * weights ) / weights.sum()

    def __call__(self, value):
        """ Append value to memorized sequence """

        self.values.append(value)

def extract_patient(csv_file_path, return_label=True):
    df = pd.read_csv(csv_file_path)

    df['PatientID'] = df['ID'].str.slice(start=3, stop=12)

    if not return_label:
        df.drop_duplicates('ID', keep='last', inplace=True)
        df = df[~(df['PatientID'] == '6431af929')] # corrupted file in train set
    
    patient = df['PatientID'].values.reshape((-1,6), order='C')[:,0]
    
    if return_label:
        label = df.Label.values.reshape((-1,6), order='C')
        return patient, label
    else:
        return patient
    
def model_from_name(model_name, pretrained=True): 
    try:
        model = eval(f'torchvision.models.{model_name}(pretrained={pretrained}, progress=False)')
    except AttributeError as error:
        error.args = ('Unknown Model, available models: https://pytorch.org/docs/stable/torchvision/models.html',)
        raise error
    try:
        model.classifier = torch.nn.Linear(model.classifier.in_features, 6)
    except:
        try:
            model.fc = torch.nn.Linear(model.fc.in_features, 6)
        except AttributeError as error:
            error.args = ('Unknown Model classifier name',)
            raise error
    return model

def prediction_to_df(predictions, patients, submission_file):

    subtype = np.array(['epidural', 'intraparenchymal',
                        'intraventricular', 'subarachnoid', 'subdural', 'any'])[None]

    rows = np.core.defchararray.add(
                np.core.defchararray.add(
                    np.core.defchararray.add('ID_', np.array(patients)), '_')[:, None], subtype)

    df = pd.DataFrame(predictions, index=rows.flatten(), columns=['Label'])
    df.index.name = 'ID'
    df = df.reset_index()
    df.drop_duplicates('ID', keep='first', inplace=True)
    df.to_csv(submission_file, index=False)
    
     