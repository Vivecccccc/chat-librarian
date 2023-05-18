import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from alexandria.chatstore.openai import OpenAIChatCompletion
from alexandria.docstore.docstore import DocStore
from alexandria.docstore.router import get_docstore
from alexandria.vectorstore.router import get_vecstore
from alexandria.vectorstore.vectorstore import VectorStore
from handler.embedding.router import get_vectorize
from handler.embedding.vectorize import Vectorize, embed_bundle
from handler.utils import hash_components, hash_string
from models.api import Settings
from models.conversation import Conversation, MultipleConversation, SingleConversation
from models.generic import Bundle
from server.constants import VECTORSTORE_CONV_SAVE_ROOT_FOR_USER, VECTORSTORE_DOC_SAVE_ROOT_FOR_ADMIN


"""
Prompt: prompt = (Context[i], Request[i], Response[i]);
Referred for embedding: ref = (Response[i], Request[i+1]);
Using `ref` to find: grounds = (Document[k], Document[k+1], ...);
Uinsg `ref` to find: dialogs = ((Requst[k], Response[k]), (Request[k+1], Response[k+1]));
Context-by-Searching: context1 = {Ground-Truth: grounds, Referring Dialog: dialogs}
Context-by-Order: context2 = ((Request[i-k], Response[i-k]), ..., (Request[i], Response[i]))
```
Using the given *CONTEXT* to answer the *QUESTION* below:
*CONTEXT*: Related Documents: context1["Ground-Truth"]; \n 
           Related Dialogs: context1["Referring Dialog"].difference(context2)
*QUESTION*: (context2, Request[i])
*ANSWER*:
```
"""
class ChatStore:
    def __init__(self,
                 session_id: int,
                 transient: bool,
                 holdings: Dict[str, Any],
                 settings: Settings):
        self.session_id = session_id
        self.transient = transient
        self.docstore = None
        self.vecstore = None
        self.chat_vecstore = None
        self.chat_model = None
        self.chunk_size = settings.chunk_size
        self._setup_storage(holdings, settings)
        self._setup_chat_model(settings)
        self.conversations: Optional[Conversation] = None
        self.conv_dict: Dict[str, SingleConversation] = {}
        assert isinstance(self.docstore, DocStore)  \
        and isinstance(self.vecstore, VectorStore) \
        and isinstance(self.chat_vecstore, VectorStore), \
        "either doc storage or vector storage has not been initialized properly"
        assert isinstance(self.vectorize, Vectorize), \
        "vectorization not initialized properly"

    def _setup_chat_model(self, settings: Settings):
        api_key = settings.openai_api_key
        api_type = settings.openai_api_type
        api_base = settings.openai_api_base
        api_version = settings.openai_api_version
        self.chat_model = OpenAIChatCompletion(openai_api_key=api_key,
                                               openai_api_base=api_base,
                                               openai_api_type=api_type,
                                               openai_api_version=api_version)

    def _setup_storage(self, holdings, settings: Settings):
        if self.transient:
            self._setup_temp_storage(holdings)
        else:
            self.docstore = get_docstore(session_id=self.session_id,
                                         transient=self.transient)
            self.vecstore = get_vecstore(session_id=self.session_id,
                                         transient=self.transient,
                                         vecstore=settings.vectorstore,
                                         restore_root=VECTORSTORE_DOC_SAVE_ROOT_FOR_ADMIN)
        self.chat_vecstore = get_vecstore(session_id=self.session_id,
                                          transient=self.transient,
                                          vecstore=settings.vectorstore,
                                          restore_root=VECTORSTORE_CONV_SAVE_ROOT_FOR_USER)
        self.vectorize = get_vectorize(settings)

    def _setup_temp_storage(self, holdings: Dict[str, Any]):
        if "_docstore" not in holdings or "_vecstore" not in holdings:
            raise ValueError("either doc storage or vector storage has not been initialized")
        docstore: DocStore = holdings.get("_docstore")
        vecstore: VectorStore = holdings.get("_vecstore")
        self.docstore = docstore
        self.vecstore = vecstore

    async def _embed_text(self, s: str) -> List[float]:
        return await self.vectorize.embed_text(s)

    async def embed_single_conv(self, conversation: SingleConversation, prompt_template: Dict[str, str]={}):
        prompt = conversation.prompt_for_embedding(prompt_template)
        return await self._embed_text(prompt)

    async def embed_chain_conv(self, 
                               conversations: Conversation,
                               chrono: bool = True,
                               homo: bool = True,
                               max_trace: int = 0,
                               prompt_template: Dict[str, str] = {}) -> Bundle:
        chains = self._get_chain(conversations=conversations,
                                 chrono=chrono,
                                 homo=homo,
                                 max_trace=max_trace)
        conv_list = [SingleConversation(conv_id=hash_components(*elem),
                                        context=None,
                                        request=elem[0],
                                        response=elem[1],
                                        metadata=None) for elem in chains]
        bundle = MultipleConversation(theme=self.session_id,
                                      contents=conv_list,
                                      embedding=None)
        bundle = await embed_bundle(bundle, self.vectorize, prompt_template=prompt_template)
        return bundle

    def _get_chain(self, 
                   conversations: Conversation,
                   chrono: bool = True,
                   homo: bool = True,
                   max_trace: int = 0):
        chains = []
        if 0 <= max_trace and conversations and conversations.curr_conv:
            move_to = conversations.next_conv if chrono else conversations.prev_conv
            container = None
            if homo:
                container = (conversations.curr_conv.request, conversations.curr_conv.response)
            else:
                if move_to is not None:
                    container = (conversations.curr_conv.response, move_to.curr_conv.request) if chrono \
                    else (move_to.curr_conv.response, conversations.curr_conv.request)
                else:
                    container = (conversations.curr_conv.response, None) if chrono \
                    else (None, conversations.curr_conv.request)
            if container:
                chains.append(container)
            else:
                return chains
            max_trace -= 1
            chains.extend(self._get_chain(conversations=move_to,
                                          homo=homo,
                                          max_trace=max_trace))
        return chains
    
    async def eloquence(self, query):
        STANDARD_PROMPT_TEMPLATE = {"request": "USER",
                                    "response": "ASSISTANT",
                                    "context": "RELEVANT MATERIALS"}
        curr_conv = SingleConversation(conv_id=hash_components(query, str(datetime.utcnow().timestamp)),
                                       request=query)
        conv = Conversation(curr_conv=curr_conv)
        if self.conversations:
            self.conversations.add_next(conv)
            self.conversations = self.conversations.next_conv
        else:
            self.conversations = conv
        self.conversations.update_dict(existed=self.conv_dict)
        emb_query = await self._get_query_pair(query)
        valid_docs_texts = await self._get_relevant_docs(emb_query)
        prev_convs, previous_conv_ids = await self._get_previous_convs(max_trace=2)
        relv_convs, relevant_conv_ids = await self._get_relevant_convs(emb_query)
        pair_map_to_msg = lambda i, p: Msg(role="user", content=p).dict() if i % 2 == 0 else Msg(role="assistant", content=p).dict()
        valid_convs = []
        for r_id, r_conv in zip(relevant_conv_ids, relv_convs):
            if r_id not in previous_conv_ids:
                valid_convs.extend(r_conv)
        valid_convs.extend([r for p in prev_convs for r in p])
        valid_convs_texts = list(map(pair_map_to_msg, valid_convs))
        self.conversations.curr_conv.context = "\n".join(valid_docs_texts)
        to_ask = self.conversations.curr_conv.prompt_for_embedding(prompt_template=STANDARD_PROMPT_TEMPLATE)
        to_ask = Msg(role="user", content=to_ask).dict()
        valid_convs_texts.append(to_ask)
        return valid_convs_texts
    
    async def chat(self, msgs: List[Dict[str, str]]):
        response = self.chat_model.respond(msgs=msgs)
        return response
    
    async def echo_response(self, pair: tuple[str]):
        request = pair[0]
        response = pair[1]
        curr_conv = self.conversations.curr_conv
        assert request == curr_conv.request
        curr_conv.response = response
        # self.conv_dict's reference should also be updated
        await self._embed_chat(conversation=curr_conv)

    async def _embed_chat(self, conversation: SingleConversation):
        STANDARD_PROMPT_TEMPLATE = {"request": "USER INPUT",
                                    "response": "ASSISTANT RESPONSE",
                                    "context": None}
        vector = await self.embed_single_conv(conversation, prompt_template=STANDARD_PROMPT_TEMPLATE)
        self.chat_vecstore._add(vectors=[vector], ids=[conversation.conv_id])

    async def _get_relevant_convs(self, vectors: List[List[float]]):
        relv_conv_ids = await self.chat_vecstore._query(vectors, k=3)
        if not relv_conv_ids:
            return [], []
        valid_conv_ids = list(set(relv_conv_ids[0]).union(set(relv_conv_ids[1])))
        relv_convs = [(self.conv_dict[id].request, self.conv_dict[id].response) for id in valid_conv_ids]
        return relv_convs, valid_conv_ids
    
    async def _get_previous_convs(self, max_trace=2):
        trace = 0
        conv_chains = []
        ids = []
        conv = self.conversations
        while trace <= max_trace and conv:
            s = ""
            curr_req = conv.curr_conv.request
            # s += f"USER_INPUT: {curr_req}" if curr_req else ""
            curr_resp = conv.curr_conv.response
            # s += f"ASSISTANT RESPONSE: {curr_resp}" if curr_resp else ""
            if curr_req and curr_resp:
                ids.append(conv.curr_conv.conv_id)
                conv_chains.append((curr_req, curr_resp))
            conv = conv.prev_conv
            trace += 1
        conv_chains.reverse()
        return conv_chains, ids

    async def _get_relevant_docs(self, emb_query):
        chunk_ids = await self.vecstore._query(emb_query)
        chunk_query, chunk_pair = chunk_ids
        valid_chunks = list(set(chunk_query).union(set(chunk_pair)))
        chunk_map = self.vecstore.reverse_doc_map()
        if not chunk_map:
            raise ValueError("chunk-doc mapping not initialized")
        valid_docs_chunks = [(chunk_map[chunk], chunk) for chunk in valid_chunks if chunk in chunk_map]
        valid_docs_texts = await self.docstore.retrieve(valid_docs_chunks)
        return valid_docs_texts

    async def _get_query_pair(self, query):
        bundle_embed = await self.embed_chain_conv(conversations=self.conversations,
                                                   chrono=False,
                                                   homo=False,
                                                   max_trace=0,
                                                   prompt_template={"request": "ASSISTANT PREVIOUS RESPONSE",
                                                                    "response": "USER CURRENT INPUT"})
        assert isinstance(bundle_embed, MultipleConversation) and len(bundle_embed) == 1
        emb_query = [await self._embed_text(query)]
        emb_query.extend(list(bundle_embed.embedding.embeddings.values()))
        assert len(emb_query) == 2
        return emb_query
    

class Msg(BaseModel):
    role: str
    content: str