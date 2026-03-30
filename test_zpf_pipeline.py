"""Quick end-to-end test for the ZPF RAG pipeline."""
import tempfile
import os

TEST_CONTENT = """# Attention in Transformers

## Definition
Attention is a mechanism that allows neural networks to focus on relevant parts
of the input when producing an output. The self-attention mechanism computes
a weighted sum of all positions in a sequence.

## Key Formula
The scaled dot-product attention is computed as:
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

Where Q is queries, K is keys, V is values, and d_k is the key dimension.

## Steps to Implement
1. Project input into Q, K, V using learned weight matrices
2. Compute attention scores using dot product of Q and K
3. Scale by sqrt(d_k) to prevent vanishing gradients
4. Apply softmax to get attention weights
5. Multiply weights by V to get the output

## Multi-Head Attention
Instead of performing a single attention function, multi-head attention
runs h parallel attention heads with different learned projections.
This allows the model to attend to information from different
representation subspaces at different positions.
"""

with tempfile.TemporaryDirectory() as tmpdir:
    # Write test file
    test_file = os.path.join(tmpdir, "attention.md")
    with open(test_file, "w") as f:
        f.write(TEST_CONTENT)

    # Test the pipeline
    from zse.core.zrag.pipeline import RAGPipeline

    pipeline = RAGPipeline(store_dir=os.path.join(tmpdir, "store"))

    # 1. Ingest
    doc_id = pipeline.ingest(test_file, title="Attention Mechanisms")
    print(f"1. Ingested: doc_id={doc_id}")

    # 2. List documents
    docs = pipeline.list_documents()
    print(f"2. Documents: {docs}")

    # 3. Search
    results = pipeline.search("how does attention work?", top_k=3)
    print(f"\n3. Search results ({len(results)} hits):")
    for r in results:
        from zse.core.zrag.zpf_spec import BlockType
        btype = BlockType(r.block_type).name if r.block_type <= 10 else "TEXT"
        print(f"   [{btype}] score={r.score:.3f}: {r.content[:80]}...")

    # 4. Get context
    ctx = pipeline.get_context("what is multi-head attention?", max_tokens=500)
    print(f"\n4. Context length: {len(ctx)} chars")

    # 5. Stats
    print(f"5. Stats: {pipeline.stats}")

    # 6. List zpf files
    zpfs = pipeline.list_zpf_files()
    print(f"6. ZPF files: {len(zpfs)}")

    # 7. Inspect
    if zpfs:
        info = pipeline.inspect_zpf(zpfs[0]["path"])
        print(f"7. Inspect: blocks={info['block_count']}, tokens={info['total_tokens']}, compression={info['compression_ratio']}x")

    # 8. Convert only
    conv_path = pipeline.convert(test_file, output_path=os.path.join(tmpdir, "out.zpf"))
    print(f"8. Converted to: {conv_path}")

    # 9. ZPFReader round-trip
    from zse.core.zrag.zpf_reader import ZPFReader
    reader = ZPFReader(conv_path)
    print(f"9. Reader: title={reader.header.title}, blocks={reader.block_count}")
    block0 = reader.read_block(0)
    print(f"   Block 0: [{block0.block_type.name}] {block0.content[:60]}...")

    # 10. Embeddings
    embs = reader.embeddings()
    print(f"10. Embeddings shape: {embs.shape}")

    # 11. Ingest .zpf directly
    doc_id2 = pipeline.ingest_zpf(conv_path)
    print(f"11. Ingested .zpf: doc_id={doc_id2}")
    docs2 = pipeline.list_documents()
    print(f"    Now {len(docs2)} documents in store")

    print("\nALL TESTS PASSED")
