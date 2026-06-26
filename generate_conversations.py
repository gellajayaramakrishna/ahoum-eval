import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CONVERSATION_TYPES = [
    "a job interview where candidate is nervous but competent",
    "two friends arguing about a political topic",
    "a therapy session where patient discusses anxiety",
    "a romantic couple having a disagreement",
    "a student asking professor for grade reconsideration",
    "a customer complaining to support agent",
    "a manager giving harsh feedback to employee",
    "two strangers bonding on a long flight",
    "a parent and teenager arguing about curfew",
    "a person asking for help after a breakup",
    "a negotiation between business partners",
    "a doctor delivering bad news to patient",
    "two colleagues competing for same promotion",
    "a person confessing a mistake to their friend",
    "a spiritual guide and seeker discussing life purpose",
    "a coach motivating an athlete before competition",
    "a person dealing with grief talking to friend",
    "a debate between two scientists on climate change",
    "a shy person trying to make new friends",
    "a leader inspiring their team during a crisis",
    "a person admitting addiction to family member",
    "two philosophers debating free will",
    "a child asking parent about death",
    "an entrepreneur pitching to skeptical investor",
    "a couple discussing having children",
    "a whistleblower talking to journalist",
    "a person overcoming fear with therapist help",
    "two rivals reconciling after long dispute",
    "a mentor guiding a lost student",
    "a person setting boundaries with toxic friend",
    "a soldier returning home talking to spouse",
    "a creative person defending unconventional ideas",
    "a person being manipulated by a colleague",
    "two friends planning a risky adventure",
    "a religious debate between believer and atheist",
    "a person receiving unexpected praise",
    "an introvert forced into leadership role",
    "a teacher inspiring a failing student",
    "a person confessing love for the first time",
    "two neighbors resolving a long-standing conflict",
    "a person dealing with imposter syndrome at work",
    "a humanitarian worker discussing moral dilemmas",
    "a hacker justifying actions to friend",
    "a person rejecting societal expectations",
    "a caregiver burning out discussing with sibling",
    "two old friends reconnecting after years apart",
    "a person being gaslit by partner",
    "a team celebrating an unexpected victory",
    "a person quitting toxic job",
    "a wise elder giving life advice to grandchild",
]

def generate_conversation(scenario):
    prompt = f"""Generate a realistic conversation for this scenario: {scenario}

Requirements:
- 6 to 10 exchanges (back and forth)
- Feel natural and human
- Show emotions clearly
- Format exactly like this:
Person A: [message]
Person B: [message]
Person A: [message]
...

Return ONLY the conversation, no titles or explanations."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    os.makedirs("conversations", exist_ok=True)
    all_conversations = []

    for i, scenario in enumerate(CONVERSATION_TYPES):
        print(f"Generating conversation {i+1}/50: {scenario[:50]}...")
        text = generate_conversation(scenario)
        entry = {
            "id": i + 1,
            "scenario": scenario,
            "conversation": text
        }
        all_conversations.append(entry)

    # Save to JSON
    with open("conversations/conversations.json", "w") as f:
        json.dump(all_conversations, f, indent=2)

    print(f"\nDone! Generated {len(all_conversations)} conversations")
    print("Saved to conversations/conversations.json")