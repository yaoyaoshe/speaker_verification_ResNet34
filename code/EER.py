import torch
import torchaudio
import numpy as np
import os
import multiprocessing
import time
import torch.nn.functional as F
from tqdm import tqdm
from module import SpeakerNet  # 修改 1: 引用新的 SpeakerNet

def get_path_from_uttid(utt_id, root_dir):
    spk_id = utt_id[:7]      
    clip_id = utt_id[-5:]    
    vid_id = utt_id[8:-6]    
    return os.path.join(root_dir, spk_id, vid_id, clip_id)

def compute_eer(scores, labels):
    from sklearn.metrics import roc_curve
    
    scores = np.array(scores)
    labels = np.array(labels)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    
    eer = fpr[idx] 
    threshold = thresholds[idx]
    
    return eer, threshold

def compute_embeddings_batch(utt_ids, model_path, audio_dir, device):
    # 修改 2: 初始化 SpeakerNet
    # 注意：我们给一个 dummy 的 num_spk (如 100)，因为在 EER 验证阶段，
    # 我们只使用 Backbone 提取特征，不使用分类头 (ArcFace)。
    # 加载权重时使用 strict=False 来忽略分类头的大小不匹配。
    model = SpeakerNet(num_spk=100) 
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint 

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                name = k[7:] # 去掉 'module.'
            else:
                name = k
            new_state_dict[name] = v
        
        # 修改 3: strict=False，允许忽略 checkpoint 中的 classifier 权重
        # 因为训练时的类别数(num_spk)和我们这里初始化的(100)不一致，
        # 但这不影响特征提取部分 (backbone + feature_extractor)。
        model.load_state_dict(new_state_dict, strict=False)
        
    except Exception as e:
        print(f"[{device}] 模型加载严重错误: {e}")
        return {}

    model.eval()
    model.to(device)
    
    embeddings = {}
    gpu_id = int(str(device).split(':')[-1]) if ':' in str(device) else 0
    
    with torch.no_grad():
        for utt_id in tqdm(utt_ids, desc=f"GPU {gpu_id}", position=gpu_id, leave=False):
            try:
                base_path = get_path_from_uttid(utt_id, audio_dir)
    
                if os.path.exists(base_path):
                    full_path = base_path
                elif os.path.exists(base_path + ".wav"):
                    full_path = base_path + ".wav"
                else:
                    continue
                
                # 读取波形
                waveform, sr = torchaudio.load(full_path)
                waveform = waveform.squeeze(0) # (Time,)
                
                if waveform.dim() > 1:
                    waveform = waveform[0]

                if waveform.numel() == 0: continue

                # 确保采样率匹配 (ResNet 前端的 FeatureExtractor 默认 16k)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                
                # 修改 4: 移除手动 Fbank 提取，直接将波形送入模型
                # SpeakerNet 内部会自动处理: WaveAugmentation(skip) -> FeatureExtractor -> ResNet
                
                # Input shape: (Batch=1, Samples)
                inp = waveform.unsqueeze(0).to(device)
                
                # 模型推断 (label=None, 只返回 embedding)
                embedding = model(inp) # (1, D)
                
                # 归一化
                embedding = F.normalize(embedding, p=2, dim=1)
                
                embeddings[utt_id] = embedding.squeeze(0).cpu().numpy()
                
            except Exception as e:
                # print(f"Error processing {utt_id}: {e}")
                continue
                
    return embeddings

def calculate_eer_parallel(model_path, trials_path, audio_dir, num_spk=1211, device_ids=None):
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    print(f"正在读取 Trials: {trials_path}")
    trials = []
    with open(trials_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                # label, enrollment_id, test_id
                trials.append((int(parts[0]), parts[1], parts[2]))

    all_utt_ids = set()
    for _, u1, u2 in trials:
        all_utt_ids.add(u1)
        all_utt_ids.add(u2)
    all_utt_ids = list(all_utt_ids)
    
    print(f"总计需要提取音频数: {len(all_utt_ids)}")

    if len(all_utt_ids) == 0:
        return 0.0

    # 分配任务给 GPU
    batch_size = len(all_utt_ids) // len(device_ids) + 1
    id_batches = [all_utt_ids[i:i + batch_size] for i in range(0, len(all_utt_ids), batch_size)]
    
    ctx = multiprocessing.get_context('spawn')
    results = {}
    
    print(f"启动 {len(device_ids)} 个进程进行特征提取...")
    start_time = time.time()
    
    with ctx.Pool(processes=len(device_ids)) as pool:
        tasks = []
        for i in range(len(device_ids)):
            if i < len(id_batches):
                tasks.append((id_batches[i], model_path, audio_dir, f"cuda:{device_ids[i]}"))
        
        feature_maps = pool.starmap(compute_embeddings_batch, tasks)
        for fmap in feature_maps:
            results.update(fmap)
            
    print(f"特征提取耗时: {time.time()-start_time:.2f}s. 成功: {len(results)}/{len(all_utt_ids)}")
    
    scores = []
    labels = []
    missing_count = 0
    
    for label, u1, u2 in trials:
        if u1 in results and u2 in results:
            emb1 = results[u1]
            emb2 = results[u2]
        
            score = np.dot(emb1, emb2)
            scores.append(score)
            labels.append(label)
        else:
            missing_count += 1
            
    if missing_count > 0:
        print(f"警告: 有 {missing_count} 对样本因特征缺失被跳过。")

    if len(scores) == 0:
        print("错误: 没有有效的评分数据。")
        return 1.0 # Error state

    eer, threshold = compute_eer(scores, labels)
    return eer

if __name__ == "__main__":
    # 配置部分
    trials_path = "/Netdata/2025/wjc/data/trials"
    audio_dir = "/DKUdata/mcheng/corpus/voxceleb1/voxceleb1_wav"
    checkpoint_dir = "/Netdata/2025/wjc/checkpoints_SIM_remake"
    # 确保这里指向你训练好的新模型
    model_name = "best_model_epoch_60.pth" 
    model_path = os.path.join(checkpoint_dir, model_name)
    
    device_ids = [0, 1, 2, 3] 

    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}")
    else:
        try:
            final_eer = calculate_eer_parallel(
                model_path=model_path,
                trials_path=trials_path,
                audio_dir=audio_dir,
                device_ids=device_ids
            )
            print("="*40)
            print(f"Result EER: {final_eer * 100:.4f}%")
            print("="*40)
        except Exception as e:
            import traceback
            traceback.print_exc()