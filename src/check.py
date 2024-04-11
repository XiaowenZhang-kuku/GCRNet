from PIL import Image
import os

# 文件夹路径包含PNG图片的目录
folder_path = "/data/CDdata/ChangeDetectionDataset/Real/subset/train/OUT"

# # 初始化计数器
# pixel_count = 0
#
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     print(filename)
#     filename1 = os.path.join(folder_path, filename)
#     for filename11 in os.listdir(filename1):
#         if filename11.endswith(".png"):
#             # 构建完整的文件路径
#             file_path = os.path.join(folder_path, filename1, filename11)
#
#             # 打开图片
#             img = Image.open(file_path)
#
#             # # 转换为灰度图像（如果需要）
#             # img = img.convert("L")
#
#             # 获取图像的像素数据
#             pixels = list(img.getdata())
#
#             # 统计像素值为255的像素点数量
#             pixel_count += pixels.count(0)
#
# # 打印结果
# print(f"Total pixels with value 255 in all PNG images: {pixel_count}")

pixel_count_0 = 0
pixel_count_255 = 0
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    print(filename)
    if filename.endswith(".jpg"):
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)

        # 打开图片
        img = Image.open(file_path)

        # # 转换为灰度图像（如果需要）
        # img = img.convert("L")

        # 获取图像的像素数据
        pixels = list(img.getdata())

        # 统计像素值为255的像素点数量
        pixel_count_0 += pixels.count(0)
        pixel_count_255 += pixels.count(255)

# 打印结果
print(f"Total pixels with value 255 in all PNG images: {pixel_count_255}")
print(f"Total pixels with value 0 in all PNG images: {pixel_count_0}")