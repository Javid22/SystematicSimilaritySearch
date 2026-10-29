
import csv
import random
import itertools
from pathlib import Path

TARGET = 5000
OUTPUT = Path("systematic_similarity_search_dataset.csv")
random.seed(42)

TOPICS = {
    "Technology": [
        ("artificial intelligence", ["AI", "automation", "machine learning"]),
        ("cybersecurity", ["security", "privacy", "passwords"]),
        ("programming", ["coding", "software", "developers"]),
        ("smartphones", ["mobile", "apps", "devices"]),
        ("cloud computing", ["cloud", "servers", "infrastructure"]),
        ("social media", ["social networks", "content", "engagement"]),
        ("data science", ["analytics", "data", "visualization"]),
        ("robotics", ["robots", "automation", "engineering"]),
        ("web development", ["websites", "frontend", "backend"]),
        ("digital privacy", ["privacy", "tracking", "online safety"]),
    ],
    "Education": [
        ("online learning", ["courses", "learning", "education"]),
        ("exam preparation", ["exams", "revision", "study"]),
        ("time management", ["planning", "productivity", "schedule"]),
        ("language learning", ["languages", "vocabulary", "practice"]),
        ("career skills", ["skills", "career", "professional growth"]),
        ("reading habits", ["books", "reading", "knowledge"]),
        ("student life", ["students", "college", "campus"]),
        ("study techniques", ["memory", "focus", "study methods"]),
        ("coding education", ["programming", "practice", "tutorials"]),
        ("digital classrooms", ["technology", "teachers", "classrooms"]),
    ],
    "Health": [
        ("sleep quality", ["sleep", "rest", "wellness"]),
        ("stress management", ["stress", "relaxation", "mental wellness"]),
        ("healthy eating", ["nutrition", "food", "health"]),
        ("hydration", ["water", "hydration", "daily habits"]),
        ("preventive care", ["health", "prevention", "wellbeing"]),
        ("work life balance", ["work", "rest", "balance"]),
        ("daily walking", ["walking", "activity", "fitness"]),
        ("healthy routines", ["habits", "health", "routine"]),
        ("mindfulness", ["mindfulness", "focus", "relaxation"]),
        ("health awareness", ["awareness", "wellness", "prevention"]),
    ],
    "Fitness": [
        ("strength training", ["strength", "gym", "exercise"]),
        ("home workouts", ["home fitness", "exercise", "routine"]),
        ("running", ["running", "cardio", "endurance"]),
        ("yoga", ["yoga", "flexibility", "balance"]),
        ("muscle recovery", ["recovery", "rest", "training"]),
        ("fitness goals", ["goals", "fitness", "consistency"]),
        ("walking workouts", ["walking", "steps", "activity"]),
        ("sports nutrition", ["nutrition", "sports", "energy"]),
        ("mobility exercises", ["mobility", "stretching", "movement"]),
        ("beginner fitness", ["beginners", "exercise", "fitness"]),
    ],
    "Travel": [
        ("budget travel", ["budget", "travel", "planning"]),
        ("solo travel", ["solo", "travel", "independence"]),
        ("road trips", ["road trip", "driving", "destinations"]),
        ("local tourism", ["local travel", "culture", "sightseeing"]),
        ("travel photography", ["photography", "travel", "memories"]),
        ("packing tips", ["packing", "luggage", "travel"]),
        ("family holidays", ["family", "holidays", "planning"]),
        ("mountain trips", ["mountains", "hiking", "nature"]),
        ("beach holidays", ["beaches", "relaxation", "holidays"]),
        ("travel safety", ["safety", "travel", "preparation"]),
    ],
    "Food": [
        ("home cooking", ["cooking", "recipes", "home food"]),
        ("meal preparation", ["meal prep", "planning", "nutrition"]),
        ("healthy snacks", ["snacks", "nutrition", "food"]),
        ("regional cuisine", ["cuisine", "culture", "food"]),
        ("vegetarian meals", ["vegetarian", "recipes", "meals"]),
        ("food waste", ["food waste", "sustainability", "storage"]),
        ("coffee culture", ["coffee", "cafes", "beverages"]),
        ("baking", ["baking", "desserts", "recipes"]),
        ("street food", ["street food", "local food", "culture"]),
        ("food photography", ["photography", "food styling", "cooking"]),
    ],
    "Finance": [
        ("monthly budgeting", ["budget", "expenses", "saving"]),
        ("emergency funds", ["emergency fund", "savings", "planning"]),
        ("long term investing", ["investing", "wealth", "long term"]),
        ("financial literacy", ["finance", "knowledge", "money"]),
        ("saving habits", ["saving", "money", "habits"]),
        ("retirement planning", ["retirement", "planning", "investing"]),
        ("debt management", ["debt", "repayment", "budget"]),
        ("expense tracking", ["expenses", "tracking", "budget"]),
        ("financial goals", ["goals", "saving", "finance"]),
        ("responsible spending", ["spending", "budget", "money"]),
    ],
    "Entertainment": [
        ("movie discussions", ["movies", "cinema", "reviews"]),
        ("music discovery", ["music", "artists", "playlists"]),
        ("book reviews", ["books", "reviews", "reading"]),
        ("documentaries", ["documentaries", "learning", "films"]),
        ("television series", ["series", "episodes", "entertainment"]),
        ("live concerts", ["concerts", "music", "events"]),
        ("podcasts", ["podcasts", "audio", "learning"]),
        ("animation", ["animation", "films", "creativity"]),
        ("comedy", ["comedy", "humor", "entertainment"]),
        ("creative storytelling", ["stories", "creativity", "writing"]),
    ],
    "Environment": [
        ("plastic reduction", ["plastic", "recycling", "environment"]),
        ("renewable energy", ["renewable energy", "solar", "sustainability"]),
        ("water conservation", ["water", "conservation", "environment"]),
        ("urban gardening", ["gardening", "plants", "urban"]),
        ("wildlife protection", ["wildlife", "conservation", "nature"]),
        ("reusable products", ["reusable", "waste reduction", "sustainability"]),
        ("public transport", ["transport", "commuting", "environment"]),
        ("climate awareness", ["climate", "awareness", "environment"]),
        ("composting", ["compost", "gardening", "organic waste"]),
        ("energy efficiency", ["energy", "efficiency", "sustainability"]),
    ],
    "Lifestyle": [
        ("morning routines", ["morning", "habits", "productivity"]),
        ("minimalist living", ["minimalism", "organization", "lifestyle"]),
        ("home organization", ["home", "organization", "decluttering"]),
        ("personal goals", ["goals", "growth", "motivation"]),
        ("digital wellbeing", ["screen time", "wellbeing", "technology"]),
        ("weekend planning", ["weekends", "planning", "leisure"]),
        ("creative hobbies", ["hobbies", "creativity", "leisure"]),
        ("daily journaling", ["journaling", "reflection", "habits"]),
        ("workplace productivity", ["work", "focus", "productivity"]),
        ("building habits", ["habits", "consistency", "personal growth"]),
    ],
}

INTENTS = [
    "practical tips for",
    "a beginner's guide to",
    "common mistakes in",
    "personal experiences with",
    "questions people ask about",
    "new ideas for",
    "ways to improve",
    "challenges associated with",
    "useful lessons about",
    "a discussion on",
    "myths and facts about",
    "future trends in",
    "a checklist for",
    "benefits and limitations of",
    "things to consider when exploring",
]

AUDIENCES = [
    "beginners", "students", "professionals", "families",
    "enthusiasts", "small business owners", "young adults",
    "working people", "communities", "people learning new skills",
]

FORMATS = [
    "discussion", "question", "guide", "experience",
    "opinion", "checklist", "tips", "explainer",
    "recommendation", "reflection",
]

OPENINGS = [
    "One useful starting point is to understand the basics of {topic}.",
    "People exploring {topic} often have different goals and constraints.",
    "A practical approach to {topic} begins with identifying the main priorities.",
    "When discussing {topic}, it helps to compare different approaches.",
    "An important part of {topic} is understanding what works in a given situation.",
    "There is more than one way to approach {topic}, depending on the circumstances.",
    "Learning about {topic} can become easier when the subject is broken into smaller steps.",
    "A thoughtful discussion about {topic} should consider both benefits and limitations.",
    "People interested in {topic} can start by reviewing their current habits and needs.",
    "A useful way to explore {topic} is to compare examples and practical outcomes.",
]

MIDDLES = [
    "Consider the available options, the time required, and the resources involved.",
    "Compare different experiences instead of assuming that one solution suits everyone.",
    "Start with manageable steps, review the results, and adjust the approach over time.",
    "Pay attention to common challenges, practical constraints, and lessons learned.",
    "Ask questions, collect relevant information, and make decisions based on clear priorities.",
    "Look at both short-term outcomes and the possible long-term effects.",
    "Keep the intended audience in mind when evaluating different recommendations.",
    "Use specific examples to understand how the idea applies in everyday situations.",
    "Avoid relying on assumptions when reliable information or direct experience is available.",
    "Review what worked, what did not work, and what could be improved next time.",
]

ENDINGS = [
    "What has your experience been, and what would you recommend to others?",
    "Which approach would you choose, and why?",
    "Sharing different perspectives can help people make better-informed decisions.",
    "The best next step depends on individual needs and circumstances.",
    "What other factors should people consider before getting started?",
    "Small improvements can provide a useful starting point for further learning.",
    "Which challenges have you encountered while exploring this subject?",
    "A comparison of real outcomes can make the discussion more useful.",
    "Feel free to share additional ideas, examples, or questions about this topic.",
    "Understanding the trade-offs is an important part of making a thoughtful choice.",
]

def create_records():
    combinations = list(itertools.product(
        CATEGORIES := list(TOPICS.keys()),
        range(10),
        INTENTS,
        AUDIENCES,
        FORMATS,
        range(10),
    ))
    random.shuffle(combinations)

    records = []
    seen_titles = set()
    seen_descriptions = set()

    for category, topic_index, intent, audience, fmt, variation in combinations:
        topic, keywords = TOPICS[category][topic_index]

        title = (
            f"{intent.title()} {topic.title()}: "
            f"{fmt.title()} for {audience.title()} "
            f"({variation + 1})"
        )

        opening = OPENINGS[(variation + topic_index) % len(OPENINGS)]
        middle = MIDDLES[(variation * 3 + topic_index) % len(MIDDLES)]
        ending = ENDINGS[(variation + topic_index * 2) % len(ENDINGS)]

        description = (
            f"{opening.format(topic=topic)} "
            f"{middle} {ending} "
            f"This {fmt} is intended for {audience} interested in {topic}."
        )

        title_key = title.casefold()
        desc_key = description.casefold()

        if title_key in seen_titles or desc_key in seen_descriptions:
            continue

        seen_titles.add(title_key)
        seen_descriptions.add(desc_key)

        records.append({
            "id": f"SSS{len(records) + 1:06d}",
            "title": title,
            "description": description,
            "keywords": ", ".join([topic] + keywords),
            "type": category,
        })

        if len(records) == TARGET:
            break

    if len(records) < TARGET:
        raise RuntimeError(
            f"Only generated {len(records)} records. "
            "Add more topics or templates."
        )

    random.shuffle(records)

    # Reassign sequential IDs after shuffling.
    for i, record in enumerate(records, start=1):
        record["id"] = f"SSS{i:06d}"

    return records


def main():
    records = create_records()

    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "title", "description", "keywords", "type"]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} records.")
    print(f"Unique IDs: {len({r['id'] for r in records})}")
    print(f"Unique titles: {len({r['title'] for r in records})}")
    print(f"CSV saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
