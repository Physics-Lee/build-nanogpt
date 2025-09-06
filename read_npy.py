import numpy as np
import tiktoken

# 读取数据分片
data = np.load('edu_fineweb10B/edufineweb_train_000001.npy')

print(f"Token数量: {len(data)}")
print(f"数据类型: {data.dtype}")
print(f"前20个token: {data[:20]}")

# 如果想看原始文本（需要解码）
enc = tiktoken.get_encoding("gpt2")
# 只解码前100个token，避免输出太长
decoded_text = enc.decode(data[:100].tolist())
print(f"解码后的文本预览:\n{decoded_text}")