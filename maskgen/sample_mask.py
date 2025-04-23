import os
from PIL import Image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch  # 导入torch库

# 定义UNet模型
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
    train_num_steps=36800, # 训练总步数
    gradient_accumulate_every=2, # 梯度累积步数
    ema_decay=0.995,        # EMA 衰减
    amp=True,               # 开启混合精度训练
    calculate_fid=False      # 是否计算 FID
)

trainer.load(28)  # 加载训练好的模型，加载检查点 model-36.pt


import os
from PIL import Image

# 生成样本，设置生成276张图片
num_images = 14720
batch_size = 32
num_batches = num_images // batch_size  # 计算生成多少批次

# 生成图像
sampled_images = []
for _ in range(num_batches):
    sampled_images_batch = diffusion.sample(batch_size=batch_size)
    sampled_images.append(sampled_images_batch)

sampled_images = torch.cat(sampled_images, dim=0)  # 合并所有生成的图像

# 保存生成的图像
samples_root = './samples_28-14720'  # 设置保存路径
os.makedirs(samples_root, exist_ok=True)  # 创建保存样本的文件夹

# 获取当前文件夹中已有的样本数量，用于命名新生成的图像
len_samples = len(os.listdir(samples_root))

# 循环处理生成的每张图像
for i in range(sampled_images.size(0)):
    current_image_tensor = sampled_images[i]  # 获取当前图像的张量
    # 将张量转换为 PIL 图像
    current_image = Image.fromarray((current_image_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype('uint8'))
    
    # 定义文件名
    file_name = f"generated_image_{i + len_samples}.png"
    
    # 保存图像到指定路径
    current_image.save(os.path.join(samples_root, file_name))
    print(f"保存第{i + len_samples}张图像到文件夹 {samples_root}")

print("所有样本已保存至文件夹")
