import os
import h5py
import scipy.io
from scipy.signal import resample 
from meldataset import mel_spectrogram

def read_simulated_data(file_name, data_dir, verbose=False):
    vprint = print if verbose else lambda *args, **kwargs: None
    name = file_name[:-4]
    vprint('\nname',name)
    
    file = scipy.io.loadmat(os.path.join(data_dir, file_name))
    # Access specific variables or datasets
    data_variable = file['Ia']  
    vprint('type(data_variable)', type(data_variable))
    vprint('data_variable.shape',data_variable.shape)
    data_variable = data_variable[0] 
    vprint('type(data_variable)', type(data_variable))
    vprint('data_variable.shape',data_variable.shape)

    time_variable = file['Time']
    vprint('type(time_variable)', type(time_variable))
    vprint('time_variable.shape',time_variable.shape)
    time_variable = time_variable[0] # divided by 20 - To Numpy Array
    vprint('type(time_variable)', type(time_variable))
    vprint('time_variable.shape',time_variable.shape)
    
    return name, data_variable, time_variable

def read_experimental_data(file_name, data_dir, verbose=False):
    vprint = print if verbose else lambda *args, **kwargs: None
    name = file_name[:-4]
    vprint('\nname',name)
    
    with h5py.File(os.path.join(data_dir, file_name), 'r') as file:
        # Access specific variables or datasets
        data_variable = file['ch_AI_6_3']  
        vprint('type(data_variable)', type(data_variable))
        vprint('data_variable.shape',data_variable.shape)
        data_variable = data_variable[0]
        vprint('type(data_variable)', type(data_variable))
        vprint('data_variable.shape',data_variable.shape)
        
        time_variable = file['ch_AI_6_3_TIME']
        vprint('type(time_variable)', type(time_variable))
        vprint('time_variable.shape',time_variable.shape)
        time_variable = time_variable[0] # divided by 20 - To Numpy Array
        vprint('type(time_variable)', type(time_variable))
        vprint('time_variable.shape',time_variable.shape)

    return name, data_variable, time_variable

def get_mel(x, h=None): # this is not the librosa function
    if h is None:
        return mel_spectrogram(x, 1024, 80, 20000, 256, 1024, 0, 8000)
    else:
        return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
    
def resample_signal(wav, desired_sample_rate=22050, original_sample_rate=20000):
    wav = resample(wav, int(len(wav) * (desired_sample_rate / original_sample_rate)))
    return wav