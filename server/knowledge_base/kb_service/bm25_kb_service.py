from typing import List,Dict, Optional, Tuple
from configs import SCORE_THRESHOLD
from langchain.schema import Document
from server.knowledge_base.kb_service.base import KBService, SupportedVSType

from server.knowledge_base.kb_service.base import KBService
from server.knowledge_base.utils import KnowledgeFile, get_kb_path, get_vs_path
import os
import shutil
import jieba
import pickle
import hashlib

import numpy as np
from collections import Counter

jieba.load_userdict('./embeddings/embedding_keywords.txt')

class Keyword_File:
    def __init__(self,doc:Document) -> None:
        self.doc = doc
        self.source = doc.metadata.get("source")
        self.key_words = self.get_key_words(doc.page_content)
        self.doc_id = self.get_doc_id(doc.page_content)
        
    def get_key_words(self,content:str)->List[str]:
        return list(jieba.cut(content))
    
    def get_doc_id(self,content:str) ->str:
        sha256 = hashlib.sha256()
        sha256.update(content.encode('utf-8'))
        return sha256.hexdigest()

class BM25_Model(object):
    def __init__(self,k1=2, k2=1, b=0.5):
        self.documents_list = []
        self.documents_number = 0
        self.avg_documents_len = 0
        self.f = []
        self.idf = {}
        self.df = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
    
    def clear(self):
        self.documents_list = []
        self.documents_number = 0
        self.avg_documents_len = 0
        self.f = []
        self.idf = {}
        self.df = {}
    
    def add(self,documents_list:List[Keyword_File]):
        self.documents_list = self.documents_list+documents_list
        self.documents_number = self.documents_number+len(documents_list)
        if self.documents_number == 0:
            self.avg_documents_len = 0
        else:
            self.avg_documents_len = sum([len(document.key_words) for document in self.documents_list]) / self.documents_number
        
        for document in documents_list:
            temp = {}
            for word in document.key_words:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                self.df[key] = self.df.get(key, 0) + 1
        
        for key, value in self.df.items():
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))
            
    def delete(self,kb_file: KnowledgeFile):
        new_doc_list = []
        new_f = []
        ids = []
        for i,file in enumerate(self.documents_list):
            if file.source != kb_file.filepath:
                new_doc_list.append(file)
                new_f.append(self.f[i])
            else:
                ids.append(file.doc_id)
                self.documents_number -=1
                temp = self.f[i]
                for key in temp.keys():
                    self.df[key] = max(self.df.get(key) - 1, 0)
        
        self.documents_list = new_doc_list
        self.f = new_f
        if self.documents_number == 0:
            self.avg_documents_len = 0
        else:
            self.avg_documents_len = sum([len(document.key_words) for document in self.documents_list]) / self.documents_number
        
        for key, value in self.df.items():
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))
        
        return ids

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                        self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                                 qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

        return score

    def get_documents_score(self, query):
        score_list = []
        query_cut = list(jieba.cut(query))
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query_cut))
        return score_list
    
    def similarity_search_with_score_by_key_word(self,query:str,
                                                 top_k:int,
                                                 score_threshold: float) -> List[Tuple[Document, float]]:
        scores = self.get_documents_score(query)
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1])[::-1][:top_k]
        results = []
        for item in sorted_scores:
            if item[1]>score_threshold:
                results.append((self.documents_list[item[0]].doc,item[1]))
        
        return results

class BM25KBService(KBService):
    vs_path: str
    kb_path: str
    bm25_model_path: str
    bm25_model: BM25_Model
    
    def vs_type(self) -> str:
        return SupportedVSType.BM25
    
    def get_vs_path(self):
        return os.path.join(get_kb_path(self.kb_name), "vector_store")

    def get_kb_path(self):
        return get_kb_path(self.kb_name)
    
    def do_create_kb(self):
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)
        
    def do_clear_vs(self):
        self.bm25_model.clear()
        try:
            shutil.rmtree(self.vs_path)
        except Exception:
            ...
        os.makedirs(self.vs_path, exist_ok=True)
            
    def do_drop_kb(self):
        self.do_clear_vs()
        try:
            shutil.rmtree(self.kb_path)
        except Exception:
            ...

    def do_add_doc(self, docs: List[Document],**kwargs)-> List[Dict]:
        li_keyword_files = []
        ids = []
        for doc in docs:
            file = Keyword_File(doc)
            ids.append(file.doc_id)
            li_keyword_files.append(file)
        
        self.bm25_model.add(li_keyword_files)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        
        return doc_infos

    def do_init(self):
        self.kb_path = self.get_kb_path()
        self.vs_path = self.get_vs_path()
        self.bm25_model_path = os.path.join(self.vs_path,'bm25.pkl')
        self.load_vector_store()
    
    def load_vector_store(self):
        if os.path.exists(self.bm25_model_path):
            f = open(self.bm25_model_path,'rb')
            self.bm25_model = pickle.load(f)
            f.close()
        else:
            self.bm25_model = BM25_Model()
    
    def save_vector_store(self):
        f = open(self.bm25_model_path,'wb')
        pickle.dump(self.bm25_model,f)
        f.close()
        
    def do_search(self,
                  query: str,
                  top_k: int,
                  score_threshold: float = SCORE_THRESHOLD,
                  ) -> List[Tuple[Document, float]]:
        
        docs = self.bm25_model.similarity_search_with_score_by_key_word(query,top_k,4/(score_threshold+0.01))
        return docs
        
    
    def do_delete_doc(self,kb_file: KnowledgeFile,**kwargs):
        ids = self.bm25_model.delete(kb_file)
        return ids

    def do_insert_multi_knowledge(self):
        pass

    def do_insert_one_knowledge(self):
        pass