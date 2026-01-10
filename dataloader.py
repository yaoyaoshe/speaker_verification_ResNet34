import torch
import torchaudio
import random
from torch.utils.data import Dataset, DataLoader
# 导入 add_reverb
from utils import load_scp_data, load_wave, add_noise, add_reverb

class Vox1Dataset(Dataset):
    def __init__(self, 
                 scp_path, 
                 noise_scp=None, 
                 speech_scp=None, 
                 music_scp=None,
                 rir_scp=None,  # [新增] RIR 混响文件列表
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
        
        # === [核心修改开始]：自动统计说话人并建立映射 ===
        self.spk2idx = {}
        unique_spks = sorted(list(set([self._get_spk_id_from_utt(utt) for utt, _ in self.data])))
        
        # 建立 原始ID -> 0...N-1 的映射
        for i, spk_id in enumerate(unique_spks):
            self.spk2idx[spk_id] = i
            
        self.num_spk = len(unique_spks) # 对外暴露类别数量
        print(f"Dataset 统计完毕: 发现 {self.num_spk} 个说话人。")
        
        self.noise_data = load_scp_data(noise_scp) if noise_scp else []
        self.speech_data = load_scp_data(speech_scp) if speech_scp else []
        self.music_data = load_scp_data(music_scp) if music_scp else []
        self.rir_data = load_scp_data(rir_scp) if rir_scp else [] # [新增]
        self.do_augment = (len(self.noise_data) + len(self.speech_data) + 
                           len(self.music_data) + len(self.rir_data)) > 0

    def _get_spk_id_from_utt(self, utt_id):
        """辅助函数：从 utt_id 解析 speaker id"""
        try:
            return utt_id.split('-')[0]
        except:
            raise ValueError(f"无法解析ID: {utt_id}")

    def __len__(self):
        return len(self.data)

    def _get_random_snr(self, noise_type):
        """根据噪声类型返回随机 SNR"""
        if noise_type == 'noise':
            return random.uniform(0, 15)
        elif noise_type == 'speech':
            return random.uniform(10, 30)
        elif noise_type == 'music':
            return random.uniform(5, 15)
        else:
            return random.uniform(0, 15)

    def _augment(self, waveform):
        """执行加噪或混响"""
        # 1. 概率触发
        if random.random() > self.aug_prob:
            return waveform

        # 2. 收集可用的增强类型
        available_types = []
        if self.noise_data: available_types.append('noise')
        if self.speech_data: available_types.append('speech')
        if self.music_data: available_types.append('music')
        if self.rir_data: available_types.append('rir') # [新增]

        if not available_types:
            return waveform

        # 3. 随机选择一种类型 (WeSpeaker 逻辑：混响和加噪互斥)
        aug_type = random.choice(available_types)

        try:
            # === 分支 A: 混响增强 ===
            if aug_type == 'rir':
                _, rir_path = random.choice(self.rir_data)
                rir_wav = load_wave(rir_path, self.sample_frequency)
                # 调用 utils.add_reverb
                return add_reverb(waveform, rir_wav)
            
            # === 分支 B: 加性噪声增强 ===
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
            # 容错：如果读取增强文件失败，打印警告并返回原语音
            # print(f"Augmentation failed ({aug_type}): {e}")
            pass

        return waveform

    def __getitem__(self, idx):
        utt_id, wav_path = self.data[idx]
        spk_str = self._get_spk_id_from_utt(utt_id)
        spk_id = self.spk2idx[spk_str]  # 获取 0 ~ N-1 的索引


        waveform = load_wave(wav_path, self.sample_frequency)

        total_samples = waveform.shape[0]
        target_frames = torch.randint(low=self.n_min, high=self.n_max + 1, size=(1,)).item()
        target_samples = target_frames * 160 

        if total_samples >= target_samples:
            start = torch.randint(0, total_samples - target_samples + 1, size=(1,)).item()
            waveform = waveform[start : start + target_samples]
        else:
            repeat = (target_samples // total_samples) + 1
            waveform = waveform.repeat(repeat)[:target_samples]

        # 3. 数据增强 (加噪 或 混响)
        if self.do_augment:
            waveform = self._augment(waveform)

        # 4. 提取特征 (Fbank)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform=waveform.unsqueeze(0),
            sample_frequency=self.sample_frequency,
            num_mel_bins=self.n_mels,
            frame_length=25.0, 
            frame_shift=10.0, 
            dither=0.0, 
            snip_edges=True
        )
        
        # 5. CMVN
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        fbank = fbank.transpose(0, 1) 
        return fbank, spk_id

class Vox1DataLoader:
    def __init__(
        self,
        scp_path: str,
        noise_scp: str = None,
        speech_scp: str = None,
        music_scp: str = None,
        rir_scp: str = None, # [新增]
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
            rir_scp=rir_scp, # [新增]
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
        fbanks, spk_ids = zip(*batch)
        max_frames = max(fbank.shape[1] for fbank in fbanks)
        padded_fbanks = []
        for fbank in fbanks:
            pad_frames = max_frames - fbank.shape[1]
            if pad_frames > 0:
                padded = torch.nn.functional.pad(fbank, (0, pad_frames), mode='constant', value=0.0)
            else:
                padded = fbank
            padded_fbanks.append(padded)

        batch_fbanks = torch.stack(padded_fbanks)
        batch_spk_ids = torch.tensor(spk_ids, dtype=torch.long)  
        return batch_fbanks, batch_spk_ids
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)