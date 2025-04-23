# import torch
# from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# # 定义Unet模型
# model = Unet(
#     dim=64,                    # 模型的基础维度，影响卷积的深度
#     dim_mults=(1, 2, 4, 8),    # 每一层的扩展倍数，决定网络的扩展方式
#     flash_attn=False            # 是否使用更高效的注意力机制
# )

# # 定义扩散模型
# diffusion = GaussianDiffusion(
#     model,                      # 之前定义的Unet模型
#     image_size=256,              # 输入图像的大小
#     timesteps=1000,              # 扩散步数（越多越精确，但训练时间更长）
#     sampling_timesteps=250      # 采样步数（使用DDIM进行加速推理）
# )

# # 创建训练器
# trainer = Trainer(
#     diffusion=diffusion,                  # 定义的扩散模型
#     folder='/home/sunyunlei01/qdn/denoising-diffusion-pytorch/datasets',         # 训练图像文件夹路径
#     train_batch_size=2,                  # 批量大小
#     train_lr=8e-5,                        # 学习率
#     train_num_steps=70000,               # 总训练步骤数
#     gradient_accumulate_every=2,          # 梯度累积步数
#     ema_decay=0.995,                      # EMA衰减因子
#     amp=True,                             # 混合精度训练
#     calculate_fid=True                    # 是否计算FID（Fréchet Inception Distance）指标
# )

# # 开始训练
# trainer.train()

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# 定义模型
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=False
)

# 定义扩散过程
diffusion = GaussianDiffusion(
    model,
    image_size=256,
    timesteps=1000,           # 训练步数
    sampling_timesteps=250    # 采样步数
)

# 使用 Trainer 类进行训练
trainer = Trainer(
    diffusion_model=diffusion,
    folder='/home/sunyunlei01/qdn/denoising-diffusion-pytorch/datasets',  # 数据集路径
    train_batch_size=8,    # 每批次大小
    train_lr=8e-5,          # 学习率
    train_num_steps=60000, # 训练总步数  1472/8=184   184*200=368
    gradient_accumulate_every=2, # 梯度累积步数
    ema_decay=0.995,        # EMA 衰减
    amp=True,               # 开启混合精度训练
    calculate_fid=False      # 是否计算 FID
)
trainer.load(36) 
trainer.train()
