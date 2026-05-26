import os
import json
import time
from facets import load_facets
from scorer import score_conversation
from dotenv import load_dotenv

load_dotenv()

def score_all_conversations():
    print("Loading facets...")
    facets = load_facets()
    print(f"Loaded {len(facets)} facets")

    with open("conversations/conversations.json", "r") as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations)} conversations")

    os.makedirs("results", exist_ok=True)

    # Resume from existing results
    try:
        with open("results/all_scores.json", "r") as f:
            all_results = json.load(f)
        done_ids = {r["id"] for r in all_results}
        print(f"Resuming — {len(done_ids)} already scored")
    except:
        all_results = []
        done_ids = set()

    for conv in conversations:
        if conv["id"] in done_ids:
            print(f"Skipping conversation {conv['id']} (already done)")
            continue

        print(f"\nScoring conversation {conv['id']}/50: {conv['scenario'][:50]}...")

        try:
            scores = score_conversation(
                conv["conversation"],
                facets,
                batch_size=30
            )

            result = {
                "id": conv["id"],
                "scenario": conv["scenario"],
                "conversation": conv["conversation"],
                "scores": scores,
                "total_facets_scored": len(scores)
            }

            all_results.append(result)

            # Save after every conversation
            with open("results/all_scores.json", "w") as f:
                json.dump(all_results, f, indent=2)

            print(f"  Done! Scored {len(scores)} facets")

            time.sleep(2)

        except Exception as e:
            print(f"  Error on conversation {conv['id']}: {e}")
            time.sleep(3)
            continue

    print(f"\nALL DONE!")
    print(f"Scored {len(all_results)} conversations")
    print(f"Results saved to results/all_scores.json")
    return all_results


if __name__ == "__main__":
    score_all_conversations()