import numpy as np


SaveList = []  # 存档列表
# 读取文本内容到列表
with open("/home/notebook/data/group/zhangrongyu/code/BEVDepth/scripts/a_night.txt", "r", encoding='utf-8') as file:
    for line in file:
        line = line.strip('\n')  # 删除换行符
        SaveList.append(line)
        print(line)
    file.close()
print("zry sb")