import time
import json
import tracemalloc
from tests.adapters import run_train_bpe

# Start memory tracking
tracemalloc.start()

# Train BPE
start = time.time()
vocab, merges = run_train_bpe(
    input_path="data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
)
elapsed = time.time() - start

# Memory usage
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Training took {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
print(f"Peak memory usage: {peak / 1e9:.2f} GB")

# Serialize vocab and merges
with open("vocab.json", "w") as f:
    # Convert bytes values to lists of ints for JSON serialization
    json.dump({str(k): list(v) for k, v in vocab.items()}, f)

with open("merges.txt", "w") as f:
    for t1, t2 in merges:
        f.write(f"{t1} {t2}\n")

print(f"Saved vocab ({len(vocab)} entries) to vocab.json")
print(f"Saved merges ({len(merges)} entries) to merges.txt")

# Find longest token
longest = max(vocab.values(), key=len)
print(f"Longest token: {longest} ({len(longest)} bytes)")
print(f"Longest token decoded: {longest.decode('utf-8', errors='replace')}")
