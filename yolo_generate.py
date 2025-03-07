import os
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from utils.attack_methods import fgsm_attack  # 对抗攻击方法
from utils.basicUtils_yolo import parse_image  # 解析图像的自定义函数

if __name__ == '__main__':
    # 定义一些超参数
    IMG_SIZE = 640  # YOLO通常使用较大的输入尺寸
    epsilon = 0.05  # 攻击强度

    # 定义文件夹路径
    images_folder = 'F:\\Datasets\\segmentation\\Raw'
    label_folder = 'F:\\Datasets\\segmentation\\labels'
    attacked_images_folder = 'F:\\Datasets\\segmentation\\FGSM_images'
    os.makedirs(attacked_images_folder, exist_ok=True)

    # 加载训练好的YOLO模型
    model = YOLO("runs/train/MM/best.pt")  # 加载训练好的模型

    # 定义图像转换
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 调整图像大小
        transforms.ToTensor(),
    ])

    # 读取图像并进行对抗攻击
    for image_name in os.listdir(images_folder):
        if image_name.endswith('.jpg'):
            # 读取图片
            image_path = os.path.join(images_folder, image_name)
            try:
                sample = parse_image(image_path)  # 使用之前定义的parse_image函数
                image = sample['image']
            except Exception as e:
                print(f"解析图像失败: {image_name}, 错误: {e}")
                continue

            # 对图像进行转换
            image_tensor = train_transform(image).unsqueeze(0)  # 添加批次维度

            # 获取标签路径
            label_path = os.path.join(label_folder, image_name.replace('.jpg', '.txt'))
            if not os.path.exists(label_path):
                print(f"标签文件不存在: {label_path}, 跳过此图像")
                continue

            # 读取标签（YOLO格式：class x_center y_center width height）
            try:
                with open(label_path, 'r') as f:
                    labels = f.readlines()
            except Exception as e:
                print(f"读取标签失败: {label_path}, 错误: {e}")
                continue

            # 对抗攻击：使用FGSM进行攻击
            try:
                attacked_image = fgsm_attack(model, image_tensor, labels, epsilon)
            except Exception as e:
                print(f"对抗攻击失败: {image_name}, 错误: {e}")
                continue

            # 转换回PIL图像
            attacked_image = attacked_image.squeeze(0)  # 移除批次维度
            attacked_image = transforms.ToPILImage()(attacked_image)

            # 保存攻击后的图像
            attacked_image_path = os.path.join(attacked_images_folder, image_name)
            attacked_image.save(attacked_image_path)

    print("所有图像已处理并保存")
