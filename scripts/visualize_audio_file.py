#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.transforms as T
import torch
import librosa

torchaudio.set_audio_backend("sox_io")

# See also:
# https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
def visualize_audio_file(input_path):
    print("Details:")
    print(torchaudio.info(input_path))
    print()

    waveform, sample_rate = torchaudio.load(input_path)

    plot_waveform(waveform, sample_rate)
    plot_specgram(waveform, sample_rate)

    # See https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html#mfcc
    n_fft = 2048
    win_length = None
    hop_length = 512
    n_mels = 256
    n_mfcc = 256

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': n_fft,
            'n_mels': n_mels,
            'hop_length': hop_length,
            'mel_scale': 'htk',
        }
    )

    mfcc = mfcc_transform(waveform)

    plot_spectrogram(mfcc[0])

# Copied from https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)

# Copied from https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'MFCC Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show()

def main():
    ap = argparse.ArgumentParser("Visualize Audio File")
    ap.add_argument("path", type=str, help="Path to .mp3 file")
    args = ap.parse_args()

    visualize_audio_file(args.path)

if __name__ == "__main__":
    main()
