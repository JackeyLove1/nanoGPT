i_n     # index: 程序索引
o_d     # offset: 维度偏移数组
m_d     # mask: 边界掩码
b_x     # block: 输入数据块
b_m     # block: 最大值（max）
b_p     # block: 概率输出（probability）

x,                  # 输入张量（scores）
p,                  # 输出张量（probabilities）
D: tl.constexpr,    # 实际的特征维度大小
B: tl.constexpr,    # 块大小（Block size）