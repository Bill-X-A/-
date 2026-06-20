import streamlit as st
from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

client = ZhipuAI(api_key=st.secrets["ZHIPU_API_KEY"])

# 加载向量库和原始资料
@st.cache_resource
def load_resources():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    index = faiss.read_index("car_index.faiss")
    with open("car_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return model, index, chunks

model, index, chunks = load_resources()

st.title("🚗 选车助手")
st.write("告诉我你的需求，我帮你推荐合适的车型")

user_question = st.text_input("你想了解什么？", placeholder="比如：预算30万左右，想要操控好的车")

if st.button("提问"):
    if user_question:
        with st.spinner("正在思考..."):
            # 1. 把问题转成向量
            question_vector = model.encode([user_question])
            
            # 2. 在向量库里搜索最相关的2段资料
            distances, indices = index.search(np.array(question_vector), k=2)
            
            # 3. 取出相关的原始文字
            relevant_chunks = [chunks[i] for i in indices[0]]
            context = "\n\n".join(relevant_chunks)
            
            # 4. 把相关资料 + 问题发给AI
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": "你是一个专业的选车顾问，根据提供的车型资料回答用户问题，给出具体推荐理由。"},
                    {"role": "user", "content": f"""车型资料：
{context}

用户问题：{user_question}

请基于以上资料回答，如果资料中没有合适的车型，请诚实告知。"""}
                ]
            )
            st.write(response.choices[0].message.content)
    else:
        st.warning("请先输入问题")
