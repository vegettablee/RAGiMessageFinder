import torch
from sentence_transformers import SentenceTransformer
import torch.optim as optim

from CD_model import CDModel
from save import load_checkpoint

# Replace with your checkpoint path
CHECKPOINT_PATH = "checkpoints/checkpoint_20251127_202931_epoch30_f10.846.pt"

# Load model once
cd_model = CDModel()
optimizer = optim.Adam(cd_model.parameters(), lr=0.001)
epoch, loss = load_checkpoint(cd_model, optimizer, filepath=CHECKPOINT_PATH)
cd_model.eval()

model = SentenceTransformer("all-mpnet-base-v2")

def test_example(example):
    messages = example["raw"]
    threads = example["threads"]

    print(f"\n{'='*80}")
    print("CONVERSATION:")
    for i, msg in enumerate(messages):
        print(f"  [{i+1}] {msg}")

    print(f"\nGROUND TRUTH ({len(threads)} threads):")
    for i, thread in enumerate(threads):
        print(f"  Thread {i+1}: {thread}")

    print(f"\nMODEL PREDICTIONS:")

    remaining_nodes = list(range(1, len(messages) + 1))

    with torch.no_grad():
        for thread_idx, correct_thread in enumerate(threads):
            # Encode remaining messages
            current_messages = [messages[node_id - 1] for node_id in remaining_nodes]
            messages_emb = model.encode(current_messages)
            input_tensor = torch.tensor(messages_emb).unsqueeze(0)

            # Predict
            thread_logits = cd_model(input_tensor)
            thread_probs = torch.softmax(thread_logits[0], dim=-1)
            top_k = min(len(correct_thread), len(thread_probs))
            top_values, top_indices = torch.topk(thread_probs, k=top_k)

            pred_ids = [remaining_nodes[idx.item()] for idx in top_indices]

            # Show results
            match = "" if pred_ids == correct_thread else ""
            print(f"  Thread {thread_idx + 1}: {correct_thread} | Predicted {pred_ids} {match}")
            print(f"  Thread sorted {thread_idx + 1}: {sorted(correct_thread)} | Predicted {sorted(pred_ids)} {match}")

            # Remove used nodes
            for node_id in correct_thread:
                remaining_nodes.remove(node_id)

    print(f"{'='*80}\n")


first_example = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    
    'raw': [
        "[08:10] <mia> good morning everyone, anyone know a good iced latte recipe?",
        "[08:11] <sam> mia: depends, do you like it sweet or strong?",
        "[08:11] <lily> morning! anyone going to the farmer's market today?",
        "[08:12] <mia> sam: probably sweet, something similar to starbucks caramel",
        "[08:12] <jake> lily: I might go later, what are you looking to buy?",
        "[08:13] <sam> mia: try adding a teaspoon of vanilla syrup + caramel drizzle",
        "[08:14] <lily> jake: just flowers and maybe some peaches",
        "[08:14] <mia> sam: oooh that sounds perfect thank you!",
        "[08:15] <oliver> anyone free to grab coffee later today?",
        "[08:16] <jake> oliver: I could, around noon maybe?"
    ],

    'date': [
        '2025-01-12', '2025-01-12', '2025-01-12', '2025-01-12', 
        '2025-01-12', '2025-01-12', '2025-01-12', '2025-01-12',
        '2025-01-12', '2025-01-12'
    ],

    # ----------------------------
    # MICRO-THREADS
    # ----------------------------
    'threads': [
        [1, 2, 4, 6, 8],    # Thread A: Mia asking about iced latte recipe
        [3, 5, 7],          # Thread B: Lily going to the farmer’s market
        [9, 10]             # Thread C: Oliver asking who is free for coffee
    ],

    # Optional, readable version of threads
    'full_thread_text': [
        {
            'thread_id': 'A',
            'nodes': [1, 2, 4, 6, 8],
            'messages': [
                "[08:10] <mia> good morning everyone, anyone know a good iced latte recipe?",
                "[08:11] <sam> mia: depends, do you like it sweet or strong?",
                "[08:12] <mia> sam: probably sweet, something similar to starbucks caramel",
                "[08:13] <sam> mia: try adding a teaspoon of vanilla syrup + caramel drizzle",
                "[08:14] <mia> sam: oooh that sounds perfect thank you!"
            ]
        },
        {
            'thread_id': 'B',
            'nodes': [3, 5, 7],
            'messages': [
                "[08:11] <lily> morning! anyone going to the farmer's market today?",
                "[08:12] <jake> lily: I might go later, what are you looking to buy?",
                "[08:14] <lily> jake: just flowers and maybe some peaches"
            ]
        },
        {
            'thread_id': 'C',
            'nodes': [9, 10],
            'messages': [
                "[08:15] <oliver> anyone free to grab coffee later today?",
                "[08:16] <jake> oliver: I could, around noon maybe?"
            ]
        }
    ]
}

if __name__ == "__main__":
    test_example(first_example)

