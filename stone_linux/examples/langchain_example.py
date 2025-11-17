"""
Example: Using LangChain with PyTorch RTX 50-series support

This example demonstrates how to use LangChain with the stone-linux optimized PyTorch
build for local LLM inference on RTX 50-series GPUs.

Requirements:
    pip install stone-linux langchain langchain-community

GPU Requirements:
    - NVIDIA RTX 5090, 5080, 5070 Ti, or 5070
    - NVIDIA Driver >= 570.00
    - CUDA 13.0+
"""

import torch
from typing import List, Optional
import time


def verify_rtx_setup():
    """Verify that we're running on RTX 50-series with proper setup."""
    print("=" * 60)
    print("RTX 50-Series Setup Verification")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    cuda_version = torch.version.cuda

    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"CUDA Version: {cuda_version}")

    if compute_cap != (12, 0):
        print("⚠️  Warning: Not running on SM 12.0 (Blackwell) GPU")
    else:
        print("✓ SM 12.0 (Blackwell) support detected!")

    print("=" * 60)


def langchain_basic_example():
    """Basic LangChain example with local LLM."""
    try:
        from langchain.llms import HuggingFacePipeline
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except ImportError as e:
        raise ImportError(
            f"Required libraries not installed: {e}\n"
            "Install with: pip install langchain transformers accelerate"
        )

    verify_rtx_setup()

    print("\n" + "=" * 60)
    print("LangChain Basic Example")
    print("=" * 60)

    # Model setup
    model_name = "gpt2"  # Small model for demo; use larger models for production
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.7,
        device=0,  # Use first GPU
    )

    # Create LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)

    # Create prompt template
    template = """Question: {question}

Answer: Let me think about this step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Test questions
    questions = [
        "What are the benefits of using RTX 50-series GPUs for AI?",
        "How does PyTorch leverage GPU acceleration?",
        "What is the Blackwell architecture?",
    ]

    print("\n" + "=" * 60)
    print("Running Inference")
    print("=" * 60)

    for question in questions:
        print(f"\nQ: {question}")
        start_time = time.time()
        result = chain.run(question=question)
        end_time = time.time()
        print(f"A: {result}")
        print(f"Time: {end_time - start_time:.2f}s")
        print("-" * 60)


def langchain_rag_example():
    """RAG (Retrieval-Augmented Generation) example with LangChain."""
    try:
        from langchain.llms import HuggingFacePipeline
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.text_splitter import CharacterTextSplitter
        from langchain.chains import RetrievalQA
        from langchain.document_loaders import TextLoader
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except ImportError as e:
        raise ImportError(
            f"Required libraries not installed: {e}\n"
            "Install with: pip install langchain transformers sentence-transformers faiss-cpu"
        )

    verify_rtx_setup()

    print("\n" + "=" * 60)
    print("LangChain RAG Example")
    print("=" * 60)

    # Sample documents about RTX 50-series
    documents_text = """
    NVIDIA GeForce RTX 5090 is the flagship GPU featuring the Blackwell architecture.
    It has 32GB of GDDR7 memory and delivers exceptional AI performance.

    RTX 5080 offers great performance with 16GB GDDR7 memory.
    It's designed for creators and AI developers who need high-end capabilities.

    The Blackwell architecture introduces 5th generation Tensor Cores.
    These cores are optimized for AI workloads and mixed-precision training.

    PyTorch 2.10 with SM 12.0 support unlocks full Blackwell potential.
    Native support provides 20-30% better performance than PTX compatibility mode.

    CUDA 13.0 introduces new features for Blackwell architecture.
    It includes improved memory management and kernel optimization.
    """

    # Create temporary document
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(documents_text)
        temp_path = f.name

    # Load and split documents
    print("\nLoading and processing documents...")
    loader = TextLoader(temp_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create embeddings using GPU
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )

    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Create LLM
    print("Loading language model...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.7,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
    )

    # Test questions
    questions = [
        "What memory does RTX 5090 have?",
        "What are the benefits of SM 12.0 support?",
        "What CUDA version should I use?",
    ]

    print("\n" + "=" * 60)
    print("RAG Question Answering")
    print("=" * 60)

    for question in questions:
        print(f"\nQ: {question}")
        start_time = time.time()
        result = qa_chain({"query": question})
        end_time = time.time()

        print(f"A: {result['result']}")
        print(f"\nSources:")
        for doc in result['source_documents']:
            print(f"  - {doc.page_content[:100]}...")
        print(f"Time: {end_time - start_time:.2f}s")
        print("-" * 60)

    # Cleanup
    import os
    os.unlink(temp_path)


def langchain_agent_example():
    """Example of using LangChain agents with tools."""
    try:
        from langchain.agents import initialize_agent, Tool, AgentType
        from langchain.llms import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except ImportError as e:
        raise ImportError(
            f"Required libraries not installed: {e}\n"
            "Install with: pip install langchain transformers"
        )

    verify_rtx_setup()

    print("\n" + "=" * 60)
    print("LangChain Agent Example")
    print("=" * 60)

    # Create custom tools
    def get_gpu_info(query: str) -> str:
        """Get information about the current GPU."""
        if not torch.cuda.is_available():
            return "No GPU available"

        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        compute_cap = torch.cuda.get_device_capability(0)

        return f"GPU: {gpu_name}, Memory: {memory_gb:.1f}GB, Compute: {compute_cap[0]}.{compute_cap[1]}"

    def calculate_throughput(query: str) -> str:
        """Calculate theoretical throughput for the GPU."""
        # Simplified calculation
        return "Estimated throughput: ~500 tokens/second for 7B model"

    tools = [
        Tool(
            name="GPU Info",
            func=get_gpu_info,
            description="Get information about the current GPU setup"
        ),
        Tool(
            name="Throughput Calculator",
            func=calculate_throughput,
            description="Calculate estimated throughput for AI workloads"
        ),
    ]

    # Create LLM
    print("\nLoading language model...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Initialize agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Test agent
    print("\n" + "=" * 60)
    print("Agent Execution")
    print("=" * 60)

    queries = [
        "What GPU am I using?",
        "What's my expected throughput?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        try:
            response = agent.run(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 60)


def main():
    """Main function to run examples."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LangChain examples for RTX 50-series GPUs"
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=["basic", "rag", "agent", "all"],
        default="basic",
        help="Which example to run",
    )

    args = parser.parse_args()

    if args.example == "basic" or args.example == "all":
        langchain_basic_example()

    if args.example == "rag" or args.example == "all":
        print("\n\n")
        langchain_rag_example()

    if args.example == "agent" or args.example == "all":
        print("\n\n")
        langchain_agent_example()


if __name__ == "__main__":
    main()
