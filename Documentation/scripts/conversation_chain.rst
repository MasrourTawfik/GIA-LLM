ConversationChain Class
=======================

The ``ConversationChain`` class is part of a project that implements the Retrieval-Augmented Generation (RAG) method to interact and retrieve information from documents, such as PDFs and text files. This class handles the management of conversation data, establishment of a retriever and execution of embedding models to compute text similarities within conversations.

Attributes
----------

- ``current_conversation_id`` : str
    A unique identifier for the current conversation, typically a timestamp.

- ``current_conversation_file`` : str
    The file path to the active conversation text file.

- ``directory`` : str
    The directory path where conversation documents are stored.

- ``docs`` : list
    A list that holds the loaded documents after processing.

- ``model`` : object
    Placeholder for an instance of a language model used for downstream tasks.

- ``chain`` : object
    Placeholder for the RAG chain that connects retrieval to downstream tasks.

- ``retriever`` : object
    An instance of a retriever that manages document indexing and retrieval.

- ``bge_embeddings`` : object
    An embedding model from Hugging Face used for generating embeddings.

- ``vectorstore`` : object
    A storage system for embeddings, used for efficient querying and retrieval.

- ``store`` : object
    A key-value document store to hold chunked documents.

- ``done`` : bool
    A flag indicating if the initialization process has been completed.

- ``conv_names`` : list
    A list of conversation document names that have been loaded.

- ``embedding_size`` : int
    The dimensionality of the text embedding vectors.

- ``index`` : object
    A FAISS index for efficient similarity search in large scale vector databases.

- ``retriever2`` : object
    Duplicate attribute, likely a mistake as it isn't assigned elsewhere in the class.

- ``memory_vectorstore`` : object
    A vector store used for short-term storage of conversation vectors.

- ``store2`` : object
    Duplicate attribute, likely a mistake as it isn't assigned elsewhere in the class.

- ``memory`` : object
    A retriever instance for managing the current conversation's memory.

Methods
-------

- ``__init__()``
    Initializes the ``ConversationChain`` with default values and directory set to ``"./conversations"``.

- ``start_new_conversation()``
    Creates a new unique conversation identified by the current timestamp and initializes a text file for storing conversation content.

- ``add_to_conversation(conversation: str)``
    Appends a string of conversation to the current conversation file.

- ``init_embedding_model()``
    Initializes the embedding model with pre-trained Hugging Face embeddings.

- ``load_documents()``
    Loads all documents from the specified directory and processes them into the docs list, printing the count of loaded conversations.

- ``init_retriever()``
    Initializes the retrievers with vector stores and document stores, then, if documents have been loaded, adds them to the retriever for indexing.

- ``init_model()``
    Placeholder method for initializing a language model for downstream tasks.

- ``init_chain()``
    Orchestrates the initialization of the RAG components including the embedding model, language model, document loading, and retriever, and sets the done flag to True once the initialization is complete.

- ``add_new_document(filename: str)``
    Adds a new conversation document to the retriever if it's not already present in the conv_names list.

- ``add_to_memory()``
    Adds the active conversation file to the memory retriever for short-term retrieval operations.

Usage Example
-------------

.. code-block:: python

    # Create a ConversationChain instance
    conv_chain = ConversationChain()

    # Start a new conversation
    conv_chain.start_new_conversation()

    # Add a dialogue to the conversation
    conv_chain.add_to_conversation("Hello, how can I assist you today?")

    # Initialize the RAG components
    conv_chain.init_chain()

    # Load and add documents to the retriever
    conv_chain.load_documents()

.. note:: 
    Classes such as ``HuggingFaceBgeEmbeddings``, ``LLM_model``, ``ParentDocumentRetriever``, ``InMemoryStore``, ``TextLoader``, and ``Chroma`` should be properly installed and imported for the ``ConversationChain`` to function.

Important Considerations
------------------------

- Ensure that the `directory` exists and is writable.
- For the RAG method to work, the backend must be properly set up with vector storage and retrieval functionality.
- Memory management can be a concern when loading many documents or managing long conversations. Performance optimization may be necessary.

If you require further information or assistance with using the ``ConversationChain`` class, consult the full documentation or contact the project maintainer.

---------------------
Document generated for the RAG implementation module by readthedocs_conversion_tool.