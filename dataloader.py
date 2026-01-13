import torch
import torchaudio
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import load_scp_data, load_wave, add_noise, add_reverb

def collate_fn_pad(batch):
    """
    用于处理变长音频的 Collate Function
    返回: (padded_waveforms, spk_ids, speed_types)
    """
    waveforms, spk_ids, speed_types = zip(*batch)
    
    # 找到 Batch 中最长的波形长度
    max_len = max(wav.shape[0] for wav in waveforms)
    padded_wavs = []
    
    for wav in waveforms:
        pad_len = max_len - wav.shape[0]
        if pad_len > 0:
            # Pad 最后一个维度 (Time)
            padded = F.pad(wav, (0, pad_len), mode='constant', value=0.0)
        else:
            padded = wav
        padded_wavs.append(padded)

    batch_wavs = torch.stack(padded_wavs) # (B, T)
    batch_spk_ids = torch.tensor(spk_ids, dtype=torch.long)
    batch_speed_types = torch.tensor(speed_types, dtype=torch.long)
    
    return batch_wavs, batch_spk_ids, batch_speed_types

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
                 n_max=800,
                 speed_perturb=False,  # [接口] 是否启用语速音高扩增
                 enable_aug=True       # [接口] 是否启用加噪/混响增强
                 ):
        super(Vox1Dataset, self).__init__()
        self.sample_frequency = sample_frequency
        self.n_mels = n_mels 
        self.n_min = n_min
        self.n_max = n_max
        self.aug_prob = aug_prob
        self.speed_perturb = speed_perturb
        self.enable_aug = enable_aug

        # 1. 加载原始数据
        original_data = load_scp_data(scp_path)
        
        # 2. 统计原始说话人
        self.spk2idx = {}
        # 简单解析说话人ID，假设格式为 id-xxxxx
        unique_spks = sorted(list(set([self._get_spk_id_from_utt(utt) for utt, _ in original_data])))
        
        for i, spk_id in enumerate(unique_spks):
            self.spk2idx[spk_id] = i
            
        self.num_original_spk = len(unique_spks)
        
        # 3. 处理语速扰动逻辑
        if self.speed_perturb:
            # 开启扩充：总分类数 = 原始 * 3
            self.num_spk = self.num_original_spk * 3
            print(f"[Dataset] 语速扰动已启用: 原始说话人 {self.num_original_spk} -> 扩充后 {self.num_spk}")
        else:
            # 关闭扩充
            self.num_spk = self.num_original_spk
            print(f"[Dataset] 语速扰动未启用: 说话人数量 {self.num_original_spk}")
        
        # 4. 构建数据列表
        # speed_type: 0=1.0x, 1=1.1x, 2=0.9x
        self.data = []
        for utt_id, wav_path in original_data:
            self.data.append((utt_id, wav_path, 0)) # 始终添加原速
            
            if self.speed_perturb:
                self.data.append((utt_id, wav_path, 1)) # 1.1倍速 (Label: x + N)
                self.data.append((utt_id, wav_path, 2)) # 0.9倍速 (Label: x + 2N)

        # 5. 加载增强数据 (仅当 enable_aug=True 时生效)
        self.noise_data = []
        self.speech_data = []
        self.music_data = []
        self.rir_data = []
        
        if self.enable_aug:
            self.noise_data = load_scp_data(noise_scp) if noise_scp else []
            self.speech_data = load_scp_data(speech_scp) if speech_scp else []
            self.music_data = load_scp_data(music_scp) if music_scp else []
            self.rir_data = load_scp_data(rir_scp) if rir_scp else []
            
            total_noise = len(self.noise_data) + len(self.speech_data) + len(self.music_data) + len(self.rir_data)
            if total_noise > 0:
                print(f"[Dataset] 环境增强已启用: 加载了 {total_noise} 条噪声/RIR素材")
            else:
                print(f"[Dataset] 环境增强已启用，但未提供噪声文件 (scp paths 为空)")
        else:
            print(f"[Dataset] 环境增强已关闭 (enable_aug=False)")

        self.do_augment = (len(self.noise_data) + len(self.speech_data) + 
                           len(self.music_data) + len(self.rir_data)) > 0 and self.enable_aug

    def _get_spk_id_from_utt(self, utt_id):
        try:
            return utt_id.split('-')[0]
        except:
            # 如果无法解析，返回整个字符串作为ID (fallback)
            return utt_id

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
        if not self.do_augment:
            return waveform
            
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
        except Exception:
            pass

        return waveform

    def __getitem__(self, idx):
        utt_id, wav_path, speed_type = self.data[idx]
        
        spk_str = self._get_spk_id_from_utt(utt_id)
        original_spk_id = self.spk2idx[spk_str]

        # 标签映射
        if speed_type == 0:
            spk_id = original_spk_id
        elif speed_type == 1:
            spk_id = original_spk_id + self.num_original_spk
        else:
            spk_id = original_spk_id + (2 * self.num_original_spk)

        waveform = load_wave(wav_path, self.sample_frequency)

        # 随机裁剪 / 循环填充
        total_samples = waveform.shape[0]
        target_frames = torch.randint(low=self.n_min, high=self.n_max + 1, size=(1,)).item()
        target_samples = target_frames * 160 

        if total_samples >= target_samples:
            start = torch.randint(0, total_samples - target_samples + 1, size=(1,)).item()
            waveform = waveform[start : start + target_samples]
        else:
            repeat = (target_samples // total_samples) + 1
            waveform = waveform.repeat(repeat)[:target_samples]

        # 环境增强
        if self.do_augment:
            waveform = self._augment(waveform)

        return waveform, spk_id, speed_type

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
        n_max: int = 800,
        speed_perturb: bool = False, # 新增
        enable_aug: bool = True      # 新增
    ):
        self.dataset = Vox1Dataset(
            scp_path=scp_path,
            noise_scp=noise_scp,
            speech_scp=speech_scp,
            music_scp=music_scp,
            rir_scp=rir_scp,
            n_min=n_min,
            n_max=n_max,
            speed_perturb=speed_perturb,
            enable_aug=enable_aug
        )
        
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_pad, # 直接使用模块级函数
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)