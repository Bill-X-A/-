from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# 1. 读取车型资料
with open("车型知识库.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 2. 把资料切成小段（每辆车一段）
chunks = [c.strip() for c in content.split("\n\n") if c.strip()]

# 3. 把文字转成向量
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
vectors = model.encode(chunks)

# 4. 存入FAISS向量库
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))

# 5. 保存到文件
faiss.write_index(index, "car_index.faiss")
with open("car_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False)

print(f"完成！共存入 {len(chunks)} 段资料")
