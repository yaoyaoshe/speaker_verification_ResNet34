import torch
import torchaudio
import random
from torch.utils.data import Dataset, DataLoader
from utils import load_scp_data, load_wave, add_noise, add_reverb

class Vox1Dataset(Dataset):
    def __init__(self, 
                 scp_path, 
                 noise_scp=None, 
                 speech_scp=None, 
                 music_scp=None,
                 rir_scp=None, 
                 aug_prob=0.6,
                 sample_frequency=16000, 
                 n_mels=80, 
                 n_min=300, 
                 n_max=800):
        super(Vox1Dataset, self).__init__()
        self.sample_frequency = sample_frequency
        self.n_mels = n_mels 
        self.n_min = n_min
        self.n_max = n_max
        self.aug_prob = aug_prob

        self.data = load_scp_data(scp_path)
        
        self.spk2idx = {}
        unique_spks = sorted(list(set([self._get_spk_id_from_utt(utt) for utt, _ in self.data])))
        
        for i, spk_id in enumerate(unique_spks):
            self.spk2idx[spk_id] = i
            
        self.num_spk = len(unique_spks)
        print(f"Dataset 统计完毕: 发现 {self.num_spk} 个说话人。")
        
        self.noise_data = load_scp_data(noise_scp) if noise_scp else []
        self.speech_data = load_scp_data(speech_scp) if speech_scp else []
        self.music_data = load_scp_data(music_scp) if music_scp else []
        self.rir_data = load_scp_data(rir_scp) if rir_scp else []
        self.do_augment = (len(self.noise_data) + len(self.speech_data) + 
                           len(self.music_data) + len(self.rir_data)) > 0

    def _get_spk_id_from_utt(self, utt_id):
        try:
            return utt_id.split('-')[0]
        except:
            raise ValueError(f"无法解析ID: {utt_id}")

    def __len__(self):
        return len(self.data)

    def _get_random_snr(self, noise_type):
        if noise_type == 'noise':
            return random.uniform(0, 15)
        elif noise_type == 'speech':
            return random.uniform(10, 30)
        elif noise_type == 'music':
            return random.uniform(5, 15)
        else:
            return random.uniform(0, 15)

    def _augment(self, waveform):
        if random.random() > self.aug_prob:
            return waveform

        available_types = []
        if self.noise_data: available_types.append('noise')
        if self.speech_data: available_types.append('speech')
        if self.music_data: available_types.append('music')
        if self.rir_data: available_types.append('rir')

        if not available_types:
            return waveform

        aug_type = random.choice(available_types)

        try:
            if aug_type == 'rir':
                _, rir_path = random.choice(self.rir_data)
                rir_wav = load_wave(rir_path, self.sample_frequency)
                return add_reverb(waveform, rir_wav)
            else:
                noise_path = None
                if aug_type == 'noise':
                    _, noise_path = random.choice(self.noise_data)
                elif aug_type == 'speech':
                    _, noise_path = random.choice(self.speech_data)
                elif aug_type == 'music':
                    _, noise_path = random.choice(self.music_data)
                
                if noise_path:
                    noise_wav = load_wave(noise_path, self.sample_frequency)
                    snr_db = self._get_random_snr(aug_type)
                    return add_noise(waveform, noise_wav, snr_db)
        except Exception as e:
            pass

        return waveform

    def __getitem__(self, idx):
        utt_id, wav_path = self.data[idx]
        spk_str = self._get_spk_id_from_utt(utt_id)
        spk_id = self.spk2idx[spk_str]

        waveform = load_wave(wav_path, self.sample_frequency)

        total_samples = waveform.shape[0]
        target_frames = torch.randint(low=self.n_min, high=self.n_max + 1, size=(1,)).item()
        # 1 frame ~ 160 samples (10ms shift)
        target_samples = target_frames * 160 

        if total_samples >= target_samples:
            start = torch.randint(0, total_samples - target_samples + 1, size=(1,)).item()
            waveform = waveform[start : start + target_samples]
        else:
            repeat = (target_samples // total_samples) + 1
            waveform = waveform.repeat(repeat)[:target_samples]

        if self.do_augment:
            waveform = self._augment(waveform)

        # 返回 Raw Waveform
        return waveform, spk_id

class Vox1DataLoader:
    def __init__(
        self,
        scp_path: str,
        noise_scp: str = None,
        speech_scp: str = None,
        music_scp: str = None,
        rir_scp: str = None,
        batch_size: int = 128,  
        num_workers: int = 8,   
        pin_memory: bool = True, 
        shuffle: bool = True,   
        drop_last: bool = False, 
        n_min: int = 300,       
        n_max: int = 800        
    ):
        self.dataset = Vox1Dataset(
            scp_path=scp_path,
            noise_scp=noise_scp,
            speech_scp=speech_scp,
            music_scp=music_scp,
            rir_scp=rir_scp,
            n_min=n_min,
            n_max=n_max
        )
        
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    
    def _collate_fn(self, batch):
        waveforms, spk_ids = zip(*batch)
        # Pad waveforms to max length in batch
        max_len = max(wav.shape[0] for wav in waveforms)
        padded_wavs = []
        for wav in waveforms:
            pad_len = max_len - wav.shape[0]
            if pad_len > 0:
                padded = torch.nn.functional.pad(wav, (0, pad_len), mode='constant', value=0.0)
            else:
                padded = wav
            padded_wavs.append(padded)

        batch_wavs = torch.stack(padded_wavs) # (B, T)
        batch_spk_ids = torch.tensor(spk_ids, dtype=torch.long)  
        return batch_wavs, batch_spk_ids
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)