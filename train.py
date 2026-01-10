# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import csv
from tqdm import tqdm
from module import test_module1
from dataloader import Vox1DataLoader
from EER import calculate_eer_parallel 
from ArcFace import ArcFace 
import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

plt.rcParams['axes.unicode_minus'] = False 

def train_model(
    train_scp,
    trials_path,  
    audio_dir,
    checkpoint_dir,
    noise_scp=None,
    speech_scp=None,
    music_scp=None,
    rir_scp=None,
    num_epochs=60,
    batch_size=64,
    accumulation_steps=2,
    lr=0.001,  
    gpus=[0, 1, 2, 3],
    device='cuda',
    resume_path=None  
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_csv_path = os.path.join(checkpoint_dir, 'training_log.csv')
    file_exists = os.path.exists(log_csv_path)

    if not file_exists:
        with open(log_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Phase', 'Loss', 'EER', 'LR', 'Time(s)'])
        print(f"新建日志文件: {log_csv_path}")
    else:
        print(f"追加日志到: {log_csv_path}")
    
    train_loader = Vox1DataLoader(
        scp_path=train_scp,
        noise_scp=noise_scp,
        speech_scp=speech_scp,
        music_scp=music_scp,
        rir_scp=rir_scp, 
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        drop_last=True 
    )
    num_spk=train_loader.dataset.num_spk
    print(f"数据集说话人数量：{num_spk}")
    
    model = test_module1()
    main_device = torch.device(f"cuda:{gpus[0]}" if len(gpus) > 0 else 'cpu')
    
    metric_fc = ArcFace(in_features=512, out_features=num_spk, s=30.0, m=0.0, easy_margin=True)
    metric_fc = metric_fc.to(main_device)

    if len(gpus) > 1:
        model = model.to(main_device)
        model = nn.DataParallel(model, device_ids=gpus)
        print(f"使用 {len(gpus)} 个GPU并行训练: {gpus}")
    else:
        model = model.to(main_device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD([
        {'params': model.parameters()},
        {'params': metric_fc.parameters()}
    ], lr=lr, momentum=0.9, weight_decay=1e-4) 
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    start_epoch = 0
    best_eer = float('inf') 
    train_losses = []
    eer_scores = []
    
    if resume_path and os.path.exists(resume_path):
        print(f"--> 正在加载断点: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=main_device,weights_only=False)
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        if 'metric_fc_state_dict' in checkpoint:
            metric_fc.load_state_dict(checkpoint['metric_fc_state_dict'])
            
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1 
        if 'eer' in checkpoint:
            best_eer = checkpoint['eer']
            
        for _ in range(start_epoch):
            scheduler.step()
            
        print(f"从 Epoch {start_epoch+1} 继续训练. 当前 LR: {scheduler.get_last_lr()}")

    

    
    warmup_epoch = 5
    class_epoch=20
    print(f"Physical Batch={batch_size}, Accumulation={accumulation_steps}, Logical Batch={batch_size*accumulation_steps}")

    for epoch in range(start_epoch, num_epochs):
        if epoch < warmup_epoch:
            current_m = 0.0
            train_loader.dataset.n_min=200
            train_loader.dataset.n_max=400
            phase_name = "Softmax Pre-train"
        elif epoch>=warmup_epoch and epoch<num_epochs-class_epoch:
            progress = (epoch - warmup_epoch) / (num_epochs - warmup_epoch-class_epoch)
            current_m = 0.2 + 0.3 * progress
            train_loader.dataset.n_min=300
            train_loader.dataset.n_max=500
            phase_name = "litter ArcFace Fine-tune"
            metric_fc.easy_margin = False 
        else:
            current_m = 0.5
            train_loader.dataset.n_min=600
            train_loader.dataset.n_max=800
            phase_name = "complete ArcFace Fine-tune"
            metric_fc.easy_margin = False 
        
        print(f'n_min = {train_loader.dataset.n_min/100}s')
        print(f'n_max = {train_loader.dataset.n_max/100}s')
        metric_fc.m = current_m

        model.train()
        metric_fc.train()
        
        total_loss = 0
        start_time = time.time()
        
        optimizer.zero_grad() 
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [{phase_name}]')
        
        for batch_idx, (fbanks, spk_ids) in enumerate(pbar):
            fbanks = fbanks.to(main_device)
            spk_ids = spk_ids.to(main_device)
 
            features = model(fbanks) 
            outputs = metric_fc(features, spk_ids)
            
            loss = criterion(outputs, spk_ids)
            loss = loss / accumulation_steps 
            
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                torch.nn.utils.clip_grad_norm_(metric_fc.parameters(), max_norm=5)
                
                optimizer.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * accumulation_steps
            total_loss += current_loss
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'm': f"{current_m:.2f}"})
        
        if (len(train_loader) % accumulation_steps) != 0:
             optimizer.step()
             optimizer.zero_grad()

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        temp_model_path = os.path.join(checkpoint_dir, f'temp_model_epoch_{epoch+1}.pth')
        
        state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        save_dict = {
            'epoch': epoch,
            'model_state_dict': state_dict_to_save,
            'metric_fc_state_dict': metric_fc.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss
        }
        
        torch.save(save_dict, temp_model_path)
        
        current_eer = calculate_eer_parallel(
            model_path=temp_model_path,
            trials_path=trials_path,
            audio_dir=audio_dir,
            device_ids=gpus[1:3]
        )
        
        eer_scores.append(current_eer)
        save_dict['eer'] = current_eer 
        
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        
        elapsed = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Phase: {phase_name}, '
              f'Loss: {avg_train_loss:.4f}, '
              f'EER: {current_eer:.4f}, '
              f'Time: {elapsed:.2f}s, '
              f'LR: {current_lr:.6f}')
        
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                phase_name, 
                f"{avg_train_loss:.4f}", 
                f"{current_eer:.4f}", 
                f"{current_lr:.6f}", 
                f"{elapsed:.2f}"
            ])
        
        if current_eer < best_eer:
            best_eer = current_eer
            save_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
            torch.save(save_dict, save_path) 
            print(f"保存最佳模型: {save_path}")
    
    if len(train_losses) > 0:
        plt.figure(figsize=(12, 5))
        ax1 = plt.gca()
        line1 = ax1.plot(range(start_epoch+1, num_epochs+1), train_losses, 'b-', label='Training Loss')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        line2 = ax2.plot(range(start_epoch+1, num_epochs+1), eer_scores, 'r-', label='EER')
        ax2.set_ylabel("EER", color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title(f"Training Curve (Resume from Epoch {start_epoch})")  
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, f'training_curves_resume_from_{start_epoch}.png'))
        plt.close()
    
    return train_losses, best_eer

if __name__ == "__main__":
    train_scp = "/Netdata/2025/wjc/data/train.scp"
    trials_path = "/Netdata/2025/wjc/data/trials"
    audio_dir = "/DKUdata/mcheng/corpus/voxceleb1/voxceleb1_wav"
    checkpoint_dir = "/Netdata/2025/wjc/checkpoints_SIM_remake"
    
    # 增强数据文件配置
    noise_scp = "/Netdata/2025/wjc/data/musan_noise.scp"
    speech_scp = "/Netdata/2025/wjc/data/musan_speech.scp"
    music_scp = "/Netdata/2025/wjc/data/musan_music.scp"
    rir_scp = "/Netdata/2025/wjc/data/rir.scp" 
    
    # 推荐参数 (配合8卡)
    num_epochs = 60
    batch_size = 32
    accumulation_steps = 8
    learning_rate = 0.1 
    gpus = [0,1,2,3] 
    
    resume_checkpoint = "/Netdata/2025/wjc/checkpoints_SIM_remake/best_model_epoch_60.pth"
    
    print("="*50)
    print("开始训练 (Resume Training)")
    print(f"使用GPU: {gpus}")
    print(f"数据增强: 开启")
    print(f"  - Noise: {noise_scp}")
    print(f"  - Speech: {speech_scp}")
    print(f"  - Music: {music_scp}")
    print(f"  - Reverb: {rir_scp}")
    print("="*50)
    
    train_losses, final_eer = train_model(
        train_scp=train_scp,
        trials_path=trials_path,
        audio_dir=audio_dir,
        checkpoint_dir=checkpoint_dir,
        noise_scp=noise_scp,
        speech_scp=speech_scp,
        music_scp=music_scp,
        rir_scp=rir_scp, 
        num_epochs=num_epochs,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        lr=learning_rate,
        gpus=gpus,
        resume_path=resume_checkpoint 
    )
    
    print(f"最佳 EER: {final_eer:.4f}")
