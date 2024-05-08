from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator

h = None
device = None

import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x): # this is not the librosa function
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax) # all of these variables come from the config.json of the pre-trained model weight


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    # list of all the files in the input_wavs_dir directories; they should all be .wav files
    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            print("filname",filname)
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname)) # uses scipy.io.wavfile underneath
            print('wav.shape', wav.shape)
            print('type(wav)', type(wav))
            wav = wav / MAX_WAV_VALUE # -> divide by 2**15 for int16 max range format
            print('wav.max()', wav.max())
            print('wav.min()', wav.min())
            print('wav.mean()', wav.mean())
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0)) # add batch dimension in front before computing the mel-spec
            # (batch size, num_mels, signal length) = (1, 80, ?)
            print('x.shape', x.shape)
            print('x.max()', x.max())
            print('x.min()', x.min())
            print('x.mean()', x.mean())
            y_g_hat = generator(x)
            print('y_g_hat.shape', y_g_hat.shape)
            print('y_g_hat.max()', y_g_hat.max())
            print('y_g_hat.min()', y_g_hat.min())
            print('y_g_hat.mean()', y_g_hat.mean())
            audio = y_g_hat.squeeze() # remove the batch dim
            audio = audio * MAX_WAV_VALUE # re-scale
            audio = audio.cpu().numpy().astype('int16')
            print("audio.shape", audio.shape)
            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True) # weights/g_02500000
    a = parser.parse_args() # stores arguments from command line

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    # this is just a fancy way of passing all the arguments inside config.json to a variable h
    h = AttrDict(json_config) 

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

