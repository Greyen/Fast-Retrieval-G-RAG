import asyncio
import traceback
from datetime import datetime , timezone
from ylightRag.yutils import get_env_value
from .ytypes import KnowledgeGraph
from dotenv import load_dotenv

from typing import Optional,Callable , List ,Dict,Any , cast ,AsyncIterator , final
from dataclasses import field ,dataclass , asdict 
from datetime import datetime

from .ykg.neo4j_impl import Neo4JStorage
from .ybase import BaseGraphStorage
from .ybase import BaseVectorStorage
from .ybase import BaseKVStorage
from .yVectordb.qdrant_impl import QdrantVectorDBStorage
from .yKV.json_kv_impl import JsonKVStorage
from .yKV.json_doc_status_impl import JsonDocStatusStorage

from ylightRag.ykg.shared_storage import (
    get_namespace_data,
    get_pipeline_status_lock,
)
from .yutils import (
    Tokenizer ,
    EmbeddingFunc,
    clean_text ,
    compute_mdhash_id ,
    logger,
    get_content_summary,
    TiktokenTokenizer
    )
from .yoperate import  (chunking_by_token_size,
                        merge_nodes_and_edges,
                        kg_query,
                        extract_entities)
import os

from .ynamespace import NameSpace , make_namespace

from .ybase import (
    StoragesStatus,
    DocStatus,
    DocStatusStorage,
    DocProcessingStatus,
    StorageNameSpace,
    QueryParam,
    
)

from ylightRag.yconstants import (
    DEFAULT_MAX_TOKEN_SUMMARY,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
)

from functools import partial

@final
@dataclass
class LightRAG:
    """LightRAG: Simple and Fast Retrieval-Augmented Generation."""

    # Directory
    # ---

    working_dir: str = field(
        default=f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    """Directory where cache and temporary files are stored."""


    # Storage
    # ---

    # """Storage backend for key-value data."""
    # using JSON kv for llm cache response etc
    kv_storage: BaseKVStorage = JsonKVStorage
    
    # """Storage backend for vector embeddings."""
    # using qdrant for storing the vector 
    vector_storage: BaseVectorStorage  = QdrantVectorDBStorage
    

    # """Storage backend for knowledge graphs."""
    # using neo4J for the graphs
    graph_storage: BaseGraphStorage  = Neo4JStorage

    doc_status_storage: BaseKVStorage = JsonDocStatusStorage
    """Storage type for tracking document processing statuses."""
    

    # Entity extraction
    # ---

    entity_extract_max_gleaning: int = field(default=1)
    """Maximum number of entity extraction attempts for ambiguous content."""

    summary_to_max_tokens: int = field(
        default=get_env_value("MAX_TOKEN_SUMMARY", DEFAULT_MAX_TOKEN_SUMMARY, int)
    )

    force_llm_summary_on_merge: int = field(
        default=get_env_value(
            "FORCE_LLM_SUMMARY_ON_MERGE", DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE, int
        )
    )

    # Text chunking
    # ---

    chunk_token_size: int = field(default=int(os.getenv("CHUNK_SIZE", 1200)))
    """Maximum number of tokens per text chunk when splitting documents."""

    chunk_overlap_token_size: int = field(
        default=int(os.getenv("CHUNK_OVERLAP_SIZE", 100))
    )
    """Number of overlapping tokens between consecutive text chunks to preserve context."""

    tokenizer: Optional[Tokenizer] = field(default=None)
    """
    A function that returns a Tokenizer instance.
    If None, and a `tiktoken_model_name` is provided, a TiktokenTokenizer will be created.
    If both are None, the default TiktokenTokenizer is used.
    """

    tiktoken_model_name: str = field(default="gpt-4o-mini")
    """Model name used for tokenization when chunking text with tiktoken. Defaults to `gpt-4o-mini`."""

    chunking_func: Callable[
        [
            Tokenizer,
            str,
            Optional[str],
            bool,
            int,
            int,
        ],
        List[Dict[str, Any]],
    ] = field(default_factory=lambda: chunking_by_token_size)
    """
    Custom chunking function for splitting text into chunks before processing.

    The function should take the following parameters:

        - `tokenizer`: A Tokenizer instance to use for tokenization.
        - `content`: The text to be split into chunks.
        - `split_by_character`: The character to split the text on. If None, the text is split into chunks of `chunk_token_size` tokens.
        - `split_by_character_only`: If True, the text is split only on the specified character.
        - `chunk_token_size`: The maximum number of tokens per chunk.
        - `chunk_overlap_token_size`: The number of overlapping tokens between consecutive chunks.

    The function should return a list of dictionaries, where each dictionary contains the following keys:
        - `tokens`: The number of tokens in the chunk.
        - `content`: The text content of the chunk.

    Defaults to `chunking_by_token_size` if not specified.

    """

    # Embedding
    # ---

    embedding_func: EmbeddingFunc | None = field(default=None)
    """Function for computing text embeddings. Must be set before use."""

    embedding_batch_num: int = field(default=int(os.getenv("EMBEDDING_BATCH_NUM", 32)))
    """Batch size for embedding computations."""

    embedding_func_max_async: int = field(
        default=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", 16))
    )
    """Maximum number of concurrent embedding function calls."""

    embedding_cache_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    """Configuration for embedding cache.
    - enabled: If True, enables caching to avoid redundant computations.
    - similarity_threshold: Minimum similarity score to use cached embeddings.
    - use_llm_check: If True, validates cached embeddings using an LLM.
    """

    # LLM Configuration
    # ---

    llm_model_func: Callable[..., object] | None = field(default=None)
    """Function for interacting with the large language model (LLM). Must be set before use."""

    llm_model_name: str = field(default="gpt-4o-mini")
    """Name of the LLM model used for generating responses."""

    llm_model_max_token_size: int = field(default=int(os.getenv("MAX_TOKENS", 32768)))
    """Maximum number of tokens allowed per LLM response."""

    llm_model_max_async: int = field(default=int(os.getenv("MAX_ASYNC", 4)))
    """Maximum number of concurrent LLM calls."""

    llm_model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to the LLM model function."""

    # Storage
    # ---

    vector_db_storage_cls_kwargs: dict[str, Any] = field(default_factory=dict)
    # """Additional parameters for vector database storage."""

    # # TODO：deprecated, remove in the future, use WORKSPACE instead
    # namespace_prefix: str = field(default="")
    # """Prefix for namespacing stored data across different environments."""

    # enable_llm_cache: bool = field(default=True)
    # """Enables caching for LLM responses to avoid redundant computations."""

    # enable_llm_cache_for_entity_extract: bool = field(default=True)
    # """If True, enables caching for entity extraction steps to reduce LLM costs."""

    # Extensions
    # ---

    max_parallel_insert: int = field(default=int(os.getenv("MAX_PARALLEL_INSERT", 2)))
    """Maximum number of parallel insert operations."""

    addon_params: dict[str, Any] = field(
        default_factory=lambda: {
            "language": get_env_value("SUMMARY_LANGUAGE", "English", str)
        }
    )

    # Storages Management
    # ---

    auto_manage_storages_states: bool = field(default=True)
    """If True, lightrag will automatically calls initialize_storages and finalize_storages at the appropriate times."""

    # Storages Management
    # ---

    # convert_response_to_json_func: Callable[[str], dict[str, Any]] = field(
    #     default_factory=lambda: convert_response_to_json
    # )
    """
    Custom function for converting LLM responses to JSON format.

    The default function is :func:`.utils.convert_response_to_json`.
    """

    cosine_better_than_threshold: float = field(
        default=float(os.getenv("COSINE_THRESHOLD", 0.2))
    )

    _storages_status: StoragesStatus = field(default=StoragesStatus.NOT_CREATED)
    
    def __post_init__(self):
        # have to check it
        from ylightRag.ykg.shared_storage import (
            initialize_share_data,
        )


        # have to check it

        # Handle deprecated parameters
        # if self.log_level is not None:
        #     warnings.warn(
        #         "WARNING: log_level parameter is deprecated, use setup_logger in utils.py instead",
        #         UserWarning,
        #         stacklevel=2,
        #     )
        # if self.log_file_path is not None:
        #     warnings.warn(
        #         "WARNING: log_file_path parameter is deprecated, use setup_logger in utils.py instead",
        #         UserWarning,
        #         stacklevel=2,
        #     )

        # Remove these attributes to prevent their use
        if hasattr(self, "log_level"):
            delattr(self, "log_level")
        if hasattr(self, "log_file_path"):
            delattr(self, "log_file_path")



        # have to check it
        initialize_share_data()

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # Verify storage implementation compatibility and environment variables
        storage_configs = [
            ("KV_STORAGE", self.kv_storage),
            ("VECTOR_STORAGE", self.vector_storage),
            ("GRAPH_STORAGE", self.graph_storage),
            ("DOC_STATUS_STORAGE", self.doc_status_storage),
        ]

        #  redudant
        # for storage_type, storage_name in storage_configs:
        #     # Verify storage implementation compatibility
        #     verify_storage_implementation(storage_type, storage_name)
        #     # Check environment variables
        #     check_storage_env_vars(storage_name)
      

        # have to look into it 
        # this it to intialize the vector arguments

        # Ensure vector_db_storage_cls_kwargs has required fields
        self.vector_db_storage_cls_kwargs = {
            "cosine_better_than_threshold": self.cosine_better_than_threshold,
            **self.vector_db_storage_cls_kwargs,
        }

        # Init Tokenizer
        # Post-initialization hook to handle backward compatabile tokenizer initialization based on provided parameters
        if self.tokenizer is None:
            if self.tiktoken_model_name:
                self.tokenizer = TiktokenTokenizer(self.tiktoken_model_name)
            else:
                self.tokenizer = TiktokenTokenizer()

        # Fix global_config now
        global_config = asdict(self)
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")


        #  have to check it after 
        # Init Embedding
        # self.embedding_func = priority_limit_async_func_call(
        #     self.embedding_func_max_async
        # )(self.embedding_func)

        # Initialize all storages

        # ---------------------------------------------------------------------------------------
        self.key_string_value_json_storage_cls: type[BaseKVStorage] = self.kv_storage  # type: ignore

        self.key_string_value_json_storage_cls = partial(  # type: ignore
            self.key_string_value_json_storage_cls, global_config=global_config
        )

        # ---------------------------------------------------------------------------------------
        self.vector_db_storage_cls: type[BaseVectorStorage] = self.vector_storage  # type: ignore


        self.vector_db_storage_cls = partial(  # type: ignore
            self.vector_db_storage_cls, global_config=global_config
        )

    
        # ---------------------------------------------------------------------------------------
        self.graph_storage_cls: type[BaseGraphStorage] = self.graph_storage # type: ignore

        self.graph_storage_cls = partial(  # type: ignore
            self.graph_storage_cls, global_config=global_config
        )

        
        # have to check it

        # Initialize document status storage

        self.doc_status_storage_cls:type[BaseKVStorage] = self.doc_status_storage

        self.llm_response_cache: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                "WORKSPACE", NameSpace.KV_STORE_LLM_RESPONSE_CACHE
            ),
            global_config=asdict(
                self
            ),  # Add global_config to ensure cache works properly
            embedding_func=self.embedding_func,
        )

        self.full_docs: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                "WORKSPACE", NameSpace.KV_STORE_FULL_DOCS
            ),
            embedding_func=self.embedding_func,
        )

        # TODO: deprecating, text_chunks is redundant with chunks_vdb
        self.text_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                "WORKSPACE", NameSpace.KV_STORE_TEXT_CHUNKS
            ),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage_cls(  # type: ignore
            namespace=make_namespace(
                "WORKSPACE", NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
            ),
            embedding_func=self.embedding_func,
        )

        self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                "WORKSPACE", NameSpace.VECTOR_STORE_ENTITIES
            ),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "source_id", "content", "file_path"},
        )

        self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                "WORKSPACE", NameSpace.VECTOR_STORE_RELATIONSHIPS
            ),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "source_id", "content", "file_path"},
        )

        self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                "WORKSPACE", NameSpace.VECTOR_STORE_CHUNKS
            ),
            embedding_func=self.embedding_func,
            meta_fields={"full_doc_id", "content", "file_path"},
        )

        # Initialize document status storage
        self.doc_status: DocStatusStorage = self.doc_status_storage_cls(
            namespace=make_namespace("WORKSPACE", NameSpace.DOC_STATUS),
            global_config=global_config,
            embedding_func=None,
        )


        #  have to look at caching

        # # Directly use llm_response_cache, don't create a new object
        # hashing_kv = self.llm_response_cache

        # self.llm_model_func = priority_limit_async_func_call(self.llm_model_max_async)(
        #     partial(
        #         self.llm_model_func,  # type: ignore
        #         hashing_kv=hashing_kv,
        #         **self.llm_model_kwargs,
        #     )
        # )

        # my llm_model_func without hashing 

        self.llm_model_func = partial( self.llm_model_func,  # type: ignore
                                    #   hashing_kv=hashing_kv,
                                      **self.llm_model_kwargs,
                                     )
         




        self._storages_status = StoragesStatus.CREATED
        
        #  have to look at it 
        # if self.auto_manage_storages_states:
        #     self._run_async_safely(self.initialize_storages, "Storage Initialization")


    #  have to look at the cast logic 
    async def _insert_done(
        self, pipeline_status=None, pipeline_status_lock=None
    ) -> None:
        tasks = [
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                self.full_docs,
                self.text_chunks,
                self.llm_response_cache,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
            ]
            if storage_inst is not None
        ]
        await asyncio.gather(*tasks)

        log_message = "In memory DB persist to disk"
        logger.info(log_message)

        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)



    async def _process_entity_relation_graph(
        self, chunk: dict[str, Any], pipeline_status=None, pipeline_status_lock=None
    ) -> list:
        try:
            chunk_results = await extract_entities(
                chunk,
                global_config=asdict(self),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
            )
            return chunk_results
        except Exception as e:
            error_msg = f"Failed to extract entities and relationships: {str(e)}"
            logger.error(error_msg)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = error_msg
                pipeline_status["history_messages"].append(error_msg)
            raise e
        

    async def initialize_storages(self):
        """Asynchronously initialize the storages"""
        if self._storages_status == StoragesStatus.CREATED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.initialize())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.INITIALIZED
            logger.debug("Initialized Storages")

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, and updating the
        document status.

        1. Get all pending, failed, and abnormally terminated processing documents.
        2. Split document content into chunks
        3. Process each chunk for entity and relation extraction
        4. Update the document status
        """

        # Get pipeline status shared data and lock
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Check if another process is already processing the queue
        async with pipeline_status_lock:
            # Ensure only one worker is processing documents
            if not pipeline_status.get("busy", False):
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs: dict[str, DocProcessingStatus] = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

                if not to_process_docs:
                    logger.info("No documents to process")
                    return

                pipeline_status.update(
                    {
                        "busy": True,
                        "job_name": "Default Job",
                        "job_start": datetime.now(timezone.utc).isoformat(),
                        "docs": 0,
                        "batchs": 0,  # Total number of files to be processed
                        "cur_batch": 0,  # Number of files already processed
                        "request_pending": False,  # Clear any previous request
                        "latest_message": "",
                    }
                )
                # Cleaning history_messages without breaking it as a shared list object
                del pipeline_status["history_messages"][:]
            else:
                # Another process is busy, just set request flag and return
                pipeline_status["request_pending"] = True
                logger.info(
                    "Another process is already processing the document queue. Request queued."
                )
                return

        try:
            # Process documents until no more documents or requests
            while True:
                if not to_process_docs:
                    log_message = "All documents have been processed or are duplicates"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                    break

                log_message = f"Processing {len(to_process_docs)} document(s)"
                logger.info(log_message)

                # Update pipeline_status, batchs now represents the total number of files to be processed
                pipeline_status["docs"] = len(to_process_docs)
                pipeline_status["batchs"] = len(to_process_docs)
                pipeline_status["cur_batch"] = 0
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Get first document's file path and total count for job name
                first_doc_id, first_doc = next(iter(to_process_docs.items()))
                first_doc_path = first_doc.file_path
                path_prefix = first_doc_path[:20] + (
                    "..." if len(first_doc_path) > 20 else ""
                )
                total_files = len(to_process_docs)
                job_name = f"{path_prefix}[{total_files} files]"
                pipeline_status["job_name"] = job_name

                # Create a counter to track the number of processed files
                processed_count = 0
                # Create a semaphore to limit the number of concurrent file processing
                semaphore = asyncio.Semaphore(self.max_parallel_insert)

                async def process_document(
                    doc_id: str,
                    status_doc: DocProcessingStatus,
                    split_by_character: str | None,
                    split_by_character_only: bool,
                    pipeline_status: dict,
                    pipeline_status_lock: asyncio.Lock,
                    semaphore: asyncio.Semaphore,
                ) -> None:
                    """Process single document"""
                    file_extraction_stage_ok = False
                    async with semaphore:
                        nonlocal processed_count
                        current_file_number = 0
                        try:
                            # Get file path from status document
                            file_path = getattr(
                                status_doc, "file_path", "unknown_source"
                            )

                            async with pipeline_status_lock:
                                # Update processed file count and save current file number
                                processed_count += 1
                                current_file_number = (
                                    processed_count  # Save the current file number
                                )
                                pipeline_status["cur_batch"] = processed_count

                                log_message = f"Extracting stage {current_file_number}/{total_files}: {file_path}"
                                logger.info(log_message)
                                pipeline_status["history_messages"].append(log_message)
                                log_message = f"Processing d-id: {doc_id}"
                                logger.info(log_message)
                                pipeline_status["latest_message"] = log_message
                                pipeline_status["history_messages"].append(log_message)

                            # Generate chunks from document
                            chunks: dict[str, Any] = {
                                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                    **dp,
                                    "full_doc_id": doc_id,
                                    "file_path": file_path,  # Add file path to each chunk
                                }
                                for dp in self.chunking_func(
                                    self.tokenizer,
                                    status_doc.content,
                                    split_by_character,
                                    split_by_character_only,
                                    self.chunk_overlap_token_size,
                                    self.chunk_token_size,
                                )
                            }

                            # Process document (text chunks and full docs) in parallel
                            # Create tasks with references for potential cancellation
                            doc_status_task = asyncio.create_task(
                                self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.PROCESSING,
                                            "chunks_count": len(chunks),
                                            "content": status_doc.content,
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                        }
                                    }
                                )
                            )
                            chunks_vdb_task = asyncio.create_task(
                                self.chunks_vdb.upsert(chunks)
                            )
                            entity_relation_task = asyncio.create_task(
                                self._process_entity_relation_graph(
                                    chunks, pipeline_status, pipeline_status_lock
                                )
                            )
                            full_docs_task = asyncio.create_task(
                                self.full_docs.upsert(
                                    {doc_id: {"content": status_doc.content}}
                                )
                            )
                            text_chunks_task = asyncio.create_task(
                                self.text_chunks.upsert(chunks)
                            )
                            tasks = [
                                doc_status_task,
                                chunks_vdb_task,
                                entity_relation_task,
                                full_docs_task,
                                text_chunks_task,
                            ]
                            await asyncio.gather(*tasks)
                            file_extraction_stage_ok = True

                        except Exception as e:
                            # Log error and update pipeline status
                            logger.error(traceback.format_exc())
                            error_msg = f"Failed to extrat document {current_file_number}/{total_files}: {file_path}"
                            logger.error(error_msg)
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(
                                    traceback.format_exc()
                                )
                                pipeline_status["history_messages"].append(error_msg)

                                # Cancel other tasks as they are no longer meaningful
                                for task in [
                                    chunks_vdb_task,
                                    entity_relation_task,
                                    full_docs_task,
                                    text_chunks_task,
                                ]:
                                    if not task.done():
                                        task.cancel()

                            # have to check it
                            # # Persistent llm cache
                            # if self.llm_response_cache:
                            #     await self.llm_response_cache.index_done_callback()

                            # Update document status to failed
                            await self.doc_status.upsert(
                                {
                                    doc_id: {
                                        "status": DocStatus.FAILED,
                                        "error": str(e),
                                        "content": status_doc.content,
                                        "content_summary": status_doc.content_summary,
                                        "content_length": status_doc.content_length,
                                        "created_at": status_doc.created_at,
                                        "updated_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "file_path": file_path,
                                    }
                                }
                            )

                    # Semphore released, concurrency controlled by graph_db_lock in merge_nodes_and_edges instead

                    if file_extraction_stage_ok:
                        try:
                            # Get chunk_results from entity_relation_task
                            chunk_results = await entity_relation_task
                            await merge_nodes_and_edges(
                                chunk_results=chunk_results,  # result collected from entity_relation_task
                                knowledge_graph_inst=self.chunk_entity_relation_graph,
                                entity_vdb=self.entities_vdb,
                                relationships_vdb=self.relationships_vdb,
                                global_config=asdict(self),
                                pipeline_status=pipeline_status,
                                pipeline_status_lock=pipeline_status_lock,
                                llm_response_cache=self.llm_response_cache,
                                current_file_number=current_file_number,
                                total_files=total_files,
                                file_path=file_path,
                            )

                            await self.doc_status.upsert(
                                {
                                    doc_id: {
                                        "status": DocStatus.PROCESSED,
                                        "chunks_count": len(chunks),
                                        "content": status_doc.content,
                                        "content_summary": status_doc.content_summary,
                                        "content_length": status_doc.content_length,
                                        "created_at": status_doc.created_at,
                                        "updated_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "file_path": file_path,
                                    }
                                }
                            )

                            # Call _insert_done after processing each file
                            await self._insert_done()

                            async with pipeline_status_lock:
                                log_message = f"Completed processing file {current_file_number}/{total_files}: {file_path}"
                                logger.info(log_message)
                                pipeline_status["latest_message"] = log_message
                                pipeline_status["history_messages"].append(log_message)

                        except Exception as e:
                            # Log error and update pipeline status
                            logger.error(traceback.format_exc())
                            error_msg = f"Merging stage failed in document {current_file_number}/{total_files}: {file_path}"
                            logger.error(error_msg)
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(
                                    traceback.format_exc()
                                )
                                pipeline_status["history_messages"].append(error_msg)

                            # have to check it
                            # # Persistent llm cache
                            # if self.llm_response_cache:
                            #     await self.llm_response_cache.index_done_callback()

                            # Update document status to failed
                            await self.doc_status.upsert(
                                {
                                    doc_id: {
                                        "status": DocStatus.FAILED,
                                        "error": str(e),
                                        "content": status_doc.content,
                                        "content_summary": status_doc.content_summary,
                                        "content_length": status_doc.content_length,
                                        "created_at": status_doc.created_at,
                                        "updated_at": datetime.now().isoformat(),
                                        "file_path": file_path,
                                    }
                                }
                            )

                # Create processing tasks for all documents
                doc_tasks = []
                for doc_id, status_doc in to_process_docs.items():
                    doc_tasks.append(
                        process_document(
                            doc_id,
                            status_doc,
                            split_by_character,
                            split_by_character_only,
                            pipeline_status,
                            pipeline_status_lock,
                            semaphore,
                        )
                    )

                # Wait for all document processing to complete
                await asyncio.gather(*doc_tasks)

                # Check if there's a pending request to process more documents (with lock)
                has_pending_request = False
                async with pipeline_status_lock:
                    has_pending_request = pipeline_status.get("request_pending", False)
                    if has_pending_request:
                        # Clear the request flag before checking for more documents
                        pipeline_status["request_pending"] = False

                if not has_pending_request:
                    break

                log_message = "Processing additional documents due to pending request"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Check for pending documents again
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

        finally:
            log_message = "Document processing pipeline completed"
            logger.info(log_message)
            # Always reset busy status when done or if an exception occurs (with lock)
            async with pipeline_status_lock:
                pipeline_status["busy"] = False
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)


    async def apipeline_enqueue_documents(
        self,
        input: str | list[str],
        ids: list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        """
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs
        2. Remove duplicate contents
        3. Generate document initial status
        4. Filter out already processed documents
        5. Enqueue document in status

        Args:
            input: Single document string or list of document strings
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
        """
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # If file_paths is provided, ensure it matches the number of documents
        if file_paths is not None:
            if isinstance(file_paths, str):
                file_paths = [file_paths]
            if len(file_paths) != len(input):
                raise ValueError(
                    "Number of file paths must match the number of documents"
                )
        else:
            # If no file paths provided, use placeholder
            file_paths = ["unknown_source"] * len(input)

        # 1. Validate ids if provided or generate MD5 hash IDs
        if ids is not None:
            # Check if the number of IDs matches the number of documents
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")

            # Check if IDs are unique
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")

            # Generate contents dict of IDs provided by user and documents
            contents = {
                id_: {"content": doc, "file_path": path}
                for id_, doc, path in zip(ids, input, file_paths)
            }
        else:

            ## TO-DO Can we do a change here for removing duplicates 

            # Clean input text and remove duplicates
            cleaned_input = [
                (clean_text(doc), path) for doc, path in zip(input, file_paths)
            ]

            unique_content_with_paths = {}

            # Keep track of unique content and their paths
            for content, path in cleaned_input:
                if content not in unique_content_with_paths:
                    unique_content_with_paths[content] = path

            # Generate contents dict of MD5 hash IDs and documents with paths
            contents = {
                compute_mdhash_id(content, prefix="doc-"): {
                    "content": content,
                    "file_path": path,
                }
                for content, path in unique_content_with_paths.items()
            }

        # 2. Remove duplicate contents
        unique_contents = {}
        for id_, content_data in contents.items():
            content = content_data["content"]
            file_path = content_data["file_path"]
            if content not in unique_contents:
                unique_contents[content] = (id_, file_path)

        # Reconstruct contents with unique content
        contents = {
            id_: {"content": content, "file_path": file_path}
            for content, (id_, file_path) in unique_contents.items()
        }

        # 3. Generate document initial status
        new_docs: dict[str, Any] = {
            id_: {
                "status": DocStatus.PENDING,
                "content": content_data["content"],
                "content_summary": get_content_summary(content_data["content"]),
                "content_length": len(content_data["content"]),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": content_data[
                    "file_path"
                ],  # Store file path in document status
            }
            for id_, content_data in contents.items()
        }

        # 4. Filter out already processed documents
        # Get docs ids
        all_new_doc_ids = set(new_docs.keys())
        # Exclude IDs of documents that are already in progress
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

        # Log ignored document IDs
        ignored_ids = [
            doc_id for doc_id in unique_new_doc_ids if doc_id not in new_docs
        ]
        if ignored_ids:
            logger.warning(
                f"Ignoring {len(ignored_ids)} document IDs not found in new_docs"
            )
            for doc_id in ignored_ids:
                logger.warning(f"Ignored document ID: {doc_id}")

        # Filter new_docs to only include documents with unique IDs
        new_docs = {
            doc_id: new_docs[doc_id]
            for doc_id in unique_new_doc_ids
            if doc_id in new_docs
        }

        if not new_docs:
            logger.info("No new unique documents were found.")
            return

        # 5. Store status document
        await self.doc_status.upsert(new_docs)
        logger.info(f"Stored {len(new_docs)} new unique documents")


    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        """Async Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
        """
        await self.apipeline_enqueue_documents(input, ids, file_paths)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )

    
    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Perform a async query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
                If param.model_func is provided, it will be used instead of the global model.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        # If a custom model is provided in param, temporarily update global config
        global_config = asdict(self)
        # Save original query for vector search
        param.original_query = query

        if param.mode in ["local", "global", "hybrid", "mix"]:
            response = await kg_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=system_prompt,
                chunks_vdb=self.chunks_vdb,
            )
        # elif param.mode == "naive":
        #     response = await naive_query(
        #         query.strip(),
        #         self.chunks_vdb,
        #         param,
        #         global_config,
        #         hashing_kv=self.llm_response_cache,
        #         system_prompt=system_prompt,
        #     )
        # elif param.mode == "bypass":
        #     # Bypass mode: directly use LLM without knowledge retrieval
        #     use_llm_func = param.model_func or global_config["llm_model_func"]
        #     # Apply higher priority (8) to entity/relation summary tasks
        #     use_llm_func = partial(use_llm_func, _priority=8)

        #     param.stream = True if param.stream is None else param.stream
        #     response = await use_llm_func(
        #         query.strip(),
        #         system_prompt=system_prompt,
        #         history_messages=param.conversation_history,
        #         stream=param.stream,
        #     )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response
    async def finalize_storages(self):
        """Asynchronously finalize the storages"""
        if self._storages_status == StoragesStatus.INITIALIZED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.finalize())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.FINALIZED
            logger.debug("Finalized Storages")

    async def get_graph_labels(self):
        text = await self.chunk_entity_relation_graph.get_all_labels()
        return text

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """Get knowledge graph for a given label

        Args:
            node_label (str): Label to get knowledge graph for
            max_depth (int): Maximum depth of graph
            max_nodes (int, optional): Maximum number of nodes to return. Defaults to 1000.

        Returns:
            KnowledgeGraph: Knowledge graph containing nodes and edges
        """

        return await self.chunk_entity_relation_graph.get_knowledge_graph(
            node_label, max_depth, max_nodes
        )

    def insert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        """Sync Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: single string of the file path or list of file paths, used for citation
        """
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert(
                input, split_by_character, split_by_character_only, ids, file_paths
            )
        )


