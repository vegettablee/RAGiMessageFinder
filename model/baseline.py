# this is a baseline I am algorithm to compare my model against. 
# more information can found inside of baseline.md 


# at a high-level 
#1. Sort messages by timestamp (if not already sorted)
#2. Iterate through messages sequentially
#3. When gap between consecutive messages > threshold:
#   - Close current chunk
#   - Start new chunk
#4. Return list of chunks


def strong_baseline(burst):
    """
    Time-gapped hierarchical agglomerative clustering
    with pre-trained sentence embeddings
    """
    # 1. Time-based pre-chunking (5min threshold)
    time_chunks = split_on_time_gaps(burst, threshold=300)
    
    # 2. Semantic clustering within chunks
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_threads = []
    
    for chunk in time_chunks:
        embeddings = model.encode([m.text for m in chunk])
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
        threads = group_by_labels(chunk, labels)
        all_threads.extend(threads)
    
    return all_threads