import torch
import torch.nn.functional as F

# 初始化Query, Key, Value
seqlen = 2048 * 8
npes = 2
warmup = 50
test_loop = 50

Q = torch.randn((seqlen*npes,64),dtype=torch.float16,device="cuda")
K = torch.randn_like(Q)
V = torch.randn_like(Q)

# for i in range(seqlen * npes):
#     for j in range(64):
#         Q[i][j] = (i % seqlen) / 130.0 + j / 110.0 + (i % seqlen) * j / 600
#         K[i][j] = (i % seqlen) / 130.0 + j / 110.0 + (i % seqlen) * j / 600
#         V[i][j] = (i % seqlen) / 130.0 + j / 110.0 + (i % seqlen) * j / 600

# 计算注意力权重
for _ in range(warmup):
    scores = torch.matmul(Q, K.transpose(0, 1))
    weights = F.softmax(scores, dim=-1)

    # 计算注意力输出
    output = torch.matmul(weights, V)

stream = torch.cuda.current_stream()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

stream.synchronize()
start_event.record(stream)
for _ in range(test_loop):
    scores = torch.matmul(Q, K.transpose(0, 1))
    weights = F.softmax(scores, dim=-1)

    # 计算注意力输出
    output = torch.matmul(weights, V)
end_event.record(stream)
stream.synchronize()
print("elapsed time:{:.3f}".format(start_event.elapsed_time(end_event)/test_loop))

# print(Q)
print(output)
