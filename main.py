import cv2
import os
import numpy as np
from ultralytics import YOLO

# 加载自定义训练的 YOLO 模型
model = YOLO("runs/train/MM/best.pt")  # 替换为您的模型路径
# result = model.predict('F:\\Datasets\\segmentation\\split\\images\\val', save=True)
results = model("F:\\Datasets\\segmentation\\split\\images\\val", save=True)
# 定义希望显示检测框的类别

# target_classes = ['ship']  # 替换为需要的类别名称
#
# # 指定图片文件夹路径
# input_folder = "F:\\Datasets\\segmentation\\split\\images\\test"
# output_folder = "F:\\Datasets\\segmentation\\split\\predict"
#
# # 创建输出文件夹（如果不存在）
# os.makedirs(output_folder, exist_ok=True)
#
# # 遍历文件夹中的每张图片
# for filename in os.listdir(input_folder):
#     if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 支持的图片格式
#         # 读取图片
#         image_path = os.path.join(input_folder, filename)
#         image = cv2.imread(image_path)
#
#         # 执行预测
#         results = model.predict(source=image)
#
#         # 处理预测结果
#         for result in results:  # 遍历每个预测结果
#             boxes = result.boxes  # 获取检测框
#             masks = result.masks  # 获取分割掩码
#             class_names = result.names  # 获取类别名称
#
#             # 遍历所有检测的实例
#             for i in range(len(boxes)):
#                 # 获取类别 ID 和名称
#                 class_id = int(boxes[i].cls)
#                 class_name = class_names[class_id]
#
#                 # 判断是否为需要显示检测框的类别
#                 if class_name in target_classes:
#                     # 获取检测框坐标
#                     x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])  # 转为整数
#
#                     # 绘制检测框
#                     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色检测框
#                     cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#                 # 绘制分割掩码
#                 mask_data = masks.data[i].cpu().numpy()  # 获取掩码的 numpy 格式数据
#                 binary_mask = mask_data > 0.5  # 应用阈值，生成二值掩码
#
#                 # 确保二值掩码与图像尺寸一致
#                 binary_mask_resized = cv2.resize(binary_mask.astype('uint8'), (image.shape[1], image.shape[0]))
#
#                 # 将二值掩码扩展到三个通道
#                 mask_3d = cv2.merge([binary_mask_resized] * 3)
#
#                 # 使用红色显示掩码区域，分别赋值给每个通道
#                 image[:, :, 0] = cv2.bitwise_and(image[:, :, 0], 255 - mask_3d[:, :, 0]) + mask_3d[:, :, 0] * 255  # 红色通道
#                 image[:, :, 1] = cv2.bitwise_and(image[:, :, 1], 255 - mask_3d[:, :, 1]) + mask_3d[:, :, 1] * 0  # 绿色通道
#                 image[:, :, 2] = cv2.bitwise_and(image[:, :, 2], 255 - mask_3d[:, :, 2]) + mask_3d[:, :, 2] * 0  # 蓝色通道
#
#         # 将处理后的图片保存到输出文件夹
#         output_path = os.path.join(output_folder, filename)
#         cv2.imwrite(output_path, image)
#         print(f"Processed and saved: {output_path}")
#
# print("All images processed and saved to output folder.")
