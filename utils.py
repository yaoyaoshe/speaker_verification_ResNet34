#utils.py
import random
import os
import torch
import math
import torchaudio
import torchaudio.functional as F

def load_scp_data(scp_path):
    '''
    从.scp文件中加载Vox1数据集
    '''
    data=[]
    with open(scp_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt_id, wav_path = parts[0], parts[1]
                    if os.path.exists(wav_path):
                        data.append((utt_id, wav_path))
    
    return data

def load_wave(path,sample_frequency=16000):
    '''
    从path加载.wav文件为tensor向量
    '''
    waveform, sr = torchaudio.load(path)
    waveform = waveform.squeeze(0)
    if sr != sample_frequency:
        waveform = F.resample(waveform, orig_freq=sr, new_freq=sample_frequency)
    return waveform

def add_noise(wave, noise, snr_db=10): 
    '''
    使用noise对wave加噪，输入输出均为tensor向量
    '''
    # 1. 长度对齐处理
    if len(noise) > len(wave):
        max_start = len(noise) - len(wave)
        start_idx = random.randint(0, max_start)
        noise = noise[start_idx : start_idx + len(wave)]
        
    elif len(noise) < len(wave):
        repeat_times = math.ceil(len(wave) / len(noise))
        noise = noise.repeat(repeat_times)[:len(wave)]

    wave_power = wave.pow(2).mean()
    noise_power = noise.pow(2).mean()
    
    if noise_power == 0:
        return wave

    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(wave_power / (snr_linear * noise_power))
    noisy_wave = wave + scale * noise
    
    return noisy_wave

def add_reverb(wave, rir):
    '''
    使用 rir 对 wave 进行混响 (卷积)，输入输出均为 tensor 向量
    '''
    rir = rir / (torch.norm(rir, p=2) + 1e-8)
    wave_input = wave.unsqueeze(0)  # [1, T_wave]
    rir_input = rir.unsqueeze(0)    # [1, T_rir]
    augmented = F.fftconvolve(wave_input, rir_input, mode="full")
    augmented = augmented.squeeze(0)[:wave.shape[0]]

    return augmented

