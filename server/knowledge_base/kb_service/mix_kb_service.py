import os
import shutil

from configs import SCORE_THRESHOLD,MIX_VS_TYPES
from server.knowledge_base.kb_service.base import KBService, SupportedVSType
from server.knowledge_base.utils import KnowledgeFile, get_kb_path
from langchain.docstore.document import Document
from typing import List, Dict, Tuple
from server.knowledge_base.kb_service.base import KBServiceFactory

import hashlib

class MixKBService(KBService):
    vs_path: str
    kb_path: str
    kb_services: List[KBService]
    mix_vs_types: List[str]
    
    def vs_type(self) -> str:
        return SupportedVSType.MIX
    
    def get_vs_path(self):
        return os.path.join(get_kb_path(self.kb_name), "vector_store")

    def get_kb_path(self):
        return get_kb_path(self.kb_name)
    
    def init_mix_vs_types(self,vs_types: List[str] = MIX_VS_TYPES):
        self.mix_vs_types = vs_types
    
    def do_init(self):
        self.kb_path = self.get_kb_path()
        self.vs_path = self.get_vs_path()
        self.init_mix_vs_types()
        services = []
        for vs_type in self.mix_vs_types:
            kb = KBServiceFactory.get_service(self.kb_name,vector_store_type=vs_type,embed_model=self.embed_model)
            services.append(kb)
            
        self.kb_services = services
        
    def do_create_kb(self):
        for service in self.kb_services:
            service.do_create_kb()

    def do_drop_kb(self):
        self.do_clear_vs()
        try:
            shutil.rmtree(self.kb_path)
        except Exception:
            ...
            
    def do_clear_vs(self):
        for service in self.kb_services:
            service.do_clear_vs()

    def do_add_doc(self, docs: List[Document],**kwargs)-> List[Dict]:
        doc_infos = []
        for service in self.kb_services:
            doc_infos = service.do_add_doc(docs,**kwargs)
        
        return doc_infos
    
    def do_delete_doc(self,kb_file: KnowledgeFile,**kwargs):
        ids = []
        for service in self.kb_services:
            ids = service.do_delete_doc(kb_file,**kwargs)
        return ids
    
    def save_vector_store(self):
        for service in self.kb_services:
            service.save_vector_store()
            
    def do_search(self,
                  query: str,
                  top_k: int,
                  score_threshold: float = SCORE_THRESHOLD)-> List[Tuple[Document, float]]:
        docs_info = dict()
        results = []
        for service in self.kb_services:
            docs = service.do_search(query,top_k,score_threshold)
            for doc in docs:
                md5 = hashlib.md5()
                md5.update(doc[0].page_content.encode('utf-8'))
                key = md5.hexdigest()
                if service.vs_type() == 'bm25':
                    score = 4/doc[1] - 0.01
                else:
                    score = doc[1]
                if key not in docs_info:
                    docs_info[key] = {'doc':doc[0],'score':score,'cnt':1}
                else:
                    docs_info[key]['cnt'] += 1
                    docs_info[key]['score'] += score
        
        for key in docs_info.keys():
            score = docs_info[key]['score']/docs_info[key]['cnt']
            pair = (docs_info[key]['doc'],score)
            results.append(pair)
        
        results = sorted(results,key=lambda t: t[1])
        if len(results) > top_k:
            return results[:top_k]
        else:
            return results

    def do_insert_multi_knowledge(self):
        pass

    def do_insert_one_knowledge(self):
        pass

    

