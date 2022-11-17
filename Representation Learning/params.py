# 参数配置
DOWN_SAMPLE_SIZE = (60,80)  # 降采样大小
SELECTED_SAMPLE_SIZE = 100  # 数据集中不同类样本的个数
SAMPLE_LENGTH = DOWN_SAMPLE_SIZE[0]*DOWN_SAMPLE_SIZE[1]     # reshape成一列数据特征长度

# 分块
IS_PATCH = True             # 是否进行分块处理
PATCH_SIZE = (20,30)        # 每一块的大小参数
PATCH_SAMPLE_LENGTH = PATCH_SIZE[0]*PATCH_SIZE[1]

# one_hot
ONE_HOT_THRESHOLD = 0.3     # One-hot类似方法阈值
FOR_DICT = True             # one_hot方法是否用来处理dict特征字典

# =====以下参数无需修改=====
# 每一张图片中分成的patch数
row_length = DOWN_SAMPLE_SIZE[1]//PATCH_SIZE[0]
col_length = DOWN_SAMPLE_SIZE[0]//PATCH_SIZE[1]
PATCH_NUM = row_length*col_length
# 行列分割点
row_dividers = []
col_dividers = []
for i in range(row_length):
    row_dividers.append(i*PATCH_SIZE[0])
for j in range(col_length):
    col_dividers.append(j*PATCH_SIZE[1])
