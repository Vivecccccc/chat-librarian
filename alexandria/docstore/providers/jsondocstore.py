import os
import json
from typing import Dict, List
from alexandria.docstore.docstore import DocStore
from models.document import ArchivedVersions, DocumentVersion, MultipleDocuments, SingleDocument, SingleDocumentWithChunks


class JsonDocStore(DocStore):
    DOC_FILES_ADMIN = 'reserve/_session/docs/bibliography'
    DOC_FILES_USER = "transient/_session-%s/docs/bibliography"
    VERSIONS = 'reserve/_session/docs/versions'
    def __init__(self,
                 storage_root: str):
        self.storage_root = storage_root

    async def _squash(
            self, 
            documents: List[SingleDocument],
            session_id: str
    ) -> MultipleDocuments:
        index = await self.__read_doc_index()
        existed_docs = [doc for doc in documents if hash(doc) in index]
        _iter = iter(existed_docs)
        curr_doc = next(_iter, None)
        while curr_doc is not None:
            curr_doc_content_hash = curr_doc.metadata.version.version_id
            existed_doc = index.get(hash(curr_doc))
            assert existed_doc is not None
            existed_doc_content_hash = existed_doc.metadata.version.version_id
            if existed_doc_content_hash == curr_doc_content_hash:
                # TODO add logs here
                documents.remove(curr_doc)
                curr_doc = next(_iter, None)
                continue
            # await self.__archive_version(curr_doc)
            curr_doc = next(_iter, None)
        return MultipleDocuments(theme=session_id, contents=documents)
    
    async def _upsert(
            self, 
            multi_docs: MultipleDocuments,
            transient: bool
    ) -> Dict[str, List[str]]:
        record_paths = []
        session_id = multi_docs.theme
        doc_root = os.path.join(self.storage_root, self.DOC_FILES_USER % (str(session_id)))
        if not transient:
            doc_root = os.path.join(self.storage_root, self.DOC_FILES_ADMIN)
            os.makedirs(doc_root, exist_ok=True)
            index = await self.__read_doc_index()
        else:
            os.makedirs(doc_root, exist_ok=True)
        docs = multi_docs.contents
        assert isinstance(docs, List)
        for doc in docs:
            doc_hash = hash(doc)
            serialized = doc.json()
            doc_path = os.path.join(doc_root, f"{doc.doc_id}.json")
            with open(doc_path, 'w') as f:
                f.write(serialized)
            if not transient:
                _doc = SingleDocument(doc_id=doc.doc_id,
                                      text=None,
                                      metadata=doc.metadata)
                index.update({doc_hash: _doc})
                await self.__archive_version(_doc)
            record_paths.append(doc_path)
        if not transient:
            index = {k: v.json() for k, v in index.items()}
            with open(os.path.join(self.storage_root, self.DOC_FILES_ADMIN, 'index.json'), 'w') as f:
                json.dump(index, f)
        return {session_id: record_paths}
            

    async def __read_doc_index(self) -> Dict[str, SingleDocument]:
        index_path = os.path.join(self.storage_root, self.DOC_FILES_ADMIN, 'index.json')
        if not os.path.isfile(index_path):
            return {}
        with open(index_path, 'r') as f:
            D = json.load(f)
        O = {}
        for k, v in D.items():
            if isinstance(v, str):
                v = json.loads(v)
            try:
                O.update({int(k): SingleDocument.parse_obj(v)})
            except Exception as e:
                print(f"error occurred when deserializing key {k} from json: {e}")
                raise ValueError(f"value {v} at key {k} not set properly")
        return O
    
    async def __read_doc_content_list(
            self,
            documents: List[SingleDocument]
    ) -> List[SingleDocument]:
        for doc in documents:
            self.__read_doc_content_single(document=doc)
    
    async def __read_doc_content_single(
            self,
            document: SingleDocument
    ):
        doc_id = document.doc_id
        doc_path = os.path.jon(self.storage_root, self.DOC_FILES_ADMIN, f"{doc_id}.json")
        raise NotImplemented
    
    async def __archive_version(
            self,
            document: SingleDocument
    ):
        doc_id = document.doc_id
        doc_version = document.metadata.version
        assert doc_version is not None
        doc_version_root = os.path.join(self.storage_root, self.VERSIONS)
        os.makedirs(doc_version_root, exist_ok=True)
        doc_version_file = os.path.join(doc_version_root, f"{doc_id}.json")
        O = {}
        if os.path.isfile(doc_version_file):
            with open(doc_version_file, 'r') as f:
                D = json.load(f)
            for k, v in D.items():
                try:
                    assert isinstance(k, str) and isinstance(v, str)
                    # _k = DocumentVersion.parse_obj(k)
                    # _v = SingleDocument.parse_obj(v)
                    O.update({k: v})
                except Exception as e:
                    print(f"error occurred when deserializing {k} or its value from json: {e}")
                    raise ValueError(f"key {k} or its value not set properly")
        O.update({document.metadata.version.json(): document.json()})
        # _archived = ArchivedVersions(doc_id=doc_id, versions=O)
        with open(doc_version_file, 'w') as f:
            # f.write(_archived.json())
            json.dump(O, f)