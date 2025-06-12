from typing import (
    Any,
    TypedDict,
)

class TextChunkSchema(TypedDict):
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int

from enum import Enum

class StoragesStatus(str, Enum):
    """Storages status"""

    NOT_CREATED = "not_created"
    CREATED = "created"
    INITIALIZED = "initialized"
    FINALIZED = "finalized"

class DocStatus(str, Enum):
    """Document processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"

@dataclass
class DocProcessingStatus:
    """Document processing status data structure"""

    content: str
    """Original content of the document"""
    content_summary: str
    """First 100 chars of document content, used for preview"""
    content_length: int
    """Total length of document"""
    file_path: str
    """File path of the document"""
    status: DocStatus
    """Current processing status"""
    created_at: str
    """ISO format timestamp when document was created"""
    updated_at: str
    """ISO format timestamp when document was last updated"""
    chunks_count: int | None = None
    """Number of chunks after splitting, used for processing"""
    error: str | None = None
    """Error message if failed"""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

@dataclass
class DocStatusStorage(BaseKVStorage, ABC):
    """Base class for document status storage"""

    @abstractmethod
    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""

    @abstractmethod
    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""

    async def drop_cache_by_modes(self, modes: list[str] | None = None) -> bool:
        """Drop cache is not supported for Doc Status storage"""
        return False


# Graph Storage 
#  hAVE TO LOOK AT THIS 
# - what are ABC
@dataclass
class BaseGraphStorage(StorageNameSpace, ABC):
    embedding_func: EmbeddingFunc

    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            node_id: The ID of the node to check

        Returns:
            True if the node exists, False otherwise
        """

    @abstractmethod
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node

        Returns:
            True if the edge exists, False otherwise
        """

    @abstractmethod
    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connected edges) of a node.

        Args:
            node_id: The ID of the node

        Returns:
            The number of edges connected to the node
        """

    @abstractmethod
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of an edge (sum of degrees of its source and target nodes).

        Args:
            src_id: The ID of the source node
            tgt_id: The ID of the target node

        Returns:
            The sum of the degrees of the source and target nodes
        """

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its ID, returning only node properties.

        Args:
            node_id: The ID of the node to retrieve

        Returns:
            A dictionary of node properties if found, None otherwise
        """

    @abstractmethod
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node

        Returns:
            A dictionary of edge properties if found, None otherwise
        """

    @abstractmethod
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges connected to a node.

        Args:
            source_node_id: The ID of the node to get edges for

        Returns:
            A list of (source_id, target_id) tuples representing edges,
            or None if the node doesn't exist
        """

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Get nodes as a batch using UNWIND

        Default implementation fetches nodes one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for node_id in node_ids:
            node = await self.get_node(node_id)
            if node is not None:
                result[node_id] = node
        return result

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Node degrees as a batch using UNWIND

        Default implementation fetches node degrees one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for node_id in node_ids:
            degree = await self.node_degree(node_id)
            result[node_id] = degree
        return result

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """Edge degrees as a batch using UNWIND also uses node_degrees_batch

        Default implementation calculates edge degrees one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for src_id, tgt_id in edge_pairs:
            degree = await self.edge_degree(src_id, tgt_id)
            result[(src_id, tgt_id)] = degree
        return result

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Get edges as a batch using UNWIND

        Default implementation fetches edges one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for pair in pairs:
            src_id = pair["src"]
            tgt_id = pair["tgt"]
            edge = await self.get_edge(src_id, tgt_id)
            if edge is not None:
                result[(src_id, tgt_id)] = edge
        return result

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Get nodes edges as a batch using UNWIND

        Default implementation fetches node edges one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for node_id in node_ids:
            edges = await self.get_node_edges(node_id)
            result[node_id] = edges if edges is not None else []
        return result

    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert a new node or update an existing node in the graph.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            node_id: The ID of the node to insert or update
            node_data: A dictionary of node properties
        """

    @abstractmethod
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert a new edge or update an existing edge in the graph.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node
            edge_data: A dictionary of edge properties
        """

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """Delete a node from the graph.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            node_id: The ID of the node to delete
        """

    @abstractmethod
    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            nodes: List of node IDs to be deleted
        """

    @abstractmethod
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """

    @abstractmethod
    async def get_all_labels(self) -> list[str]:
        """Get all labels in the graph.

        Returns:
            A list of all node labels in the graph, sorted alphabetically
        """

    @abstractmethod
    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node，* means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return, Defaults to 1000（BFS if possible)

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """

# Vector Storage 

@dataclass
class BaseVectorStorage(StorageNameSpace, ABC):
    embedding_func: EmbeddingFunc
    cosine_better_than_threshold: float = field(default=0.2)
    meta_fields: set[str] = field(default_factory=set)

    @abstractmethod
    async def query(
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Query the vector storage and retrieve top_k results."""

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update vectors in the storage.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

    @abstractmethod
    async def delete_entity(self, entity_name: str) -> None:
        """Delete a single entity by its name.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

    @abstractmethod
    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete relations for a given entity.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        pass

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        pass

    @abstractmethod
    async def delete(self, ids: list[str]):
        """Delete vectors with specified IDs

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            ids: List of vector IDs to be deleted
        """

