import os
from server.knowledge_base.kb_service.bm25_kb_service import BM25KBService
from server.knowledge_base.utils import KnowledgeFile
from configs.model_config import NLTK_DATA_PATH
import nltk
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path
import pickle
import jieba
from server.knowledge_base.migrate import folder2db,create_tables
import pyopenjtalk
from audio.audio_generate import AudioGenerator
from configs.audio_config import DEFAULT_EMOTION

path = './knowledge_base/test/docs_store/data.txt'
text = '今天是我的幸运日。'

ag = AudioGenerator('白河萤')
audio_content,rate = ag.convert_text_to_audio(
    text=text,
    speaker='白河萤',
    sdp_ratio=0.5,
    noise_scale= 0.6,
    noise_scale_w=0.9,
    length_scale=1.0,
    reference_audio= None,
    emotion= DEFAULT_EMOTION,
    prompt_mode= 'Text prompt'
)

# create_tables()
# folder2db(['MO'],'recreate_vs','bm25')

# BM25_Service = BM25KBService("MO")
# docs = BM25_Service.search_docs('秋之回忆8是哪一年发售的？',10,5)
# for i,doc in enumerate(docs):
#     print('-----------------------')
#     print(f'文档{i+1}：')
#     print(doc[0].page_content)
#     print(f'文档{i+1}得分：{doc[1]}')
#     print('-----------------------')
# file = KnowledgeFile('三城柚莉人物介绍.md','test')
# print(file.filepath)
# docs = file.file2text()
# f = open(path,'wb')
# pickle.dump(docs,f)
# f.close()

# f = open(path,'rb')
# docs = pickle.load(f)
# f.close()
# print(docs[0].page_content)


# BM25_Service.create_kb()