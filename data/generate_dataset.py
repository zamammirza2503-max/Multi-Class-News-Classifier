"""
generate_dataset.py
--------------------
Generates a synthetic 5-category news dataset (news_dataset.csv) so this
project runs end-to-end WITHOUT needing to download anything from the
internet or sign up for a Kaggle API key.

Categories: Business, Sports, Technology, Entertainment, Politics

Each "article" is built by randomly combining category-specific vocabulary
into short news-style sentences. This keeps the project 100% beginner
friendly and reproducible on any machine.

If you'd rather train on a REAL dataset, see the "Using a real dataset"
section in the README for instructions on swapping in the BBC News dataset
(Kaggle) or scikit-learn's 20 Newsgroups dataset.

Usage:
    python data/generate_dataset.py
"""

import random
import csv
import os

random.seed(42)

CATEGORIES = ["Business", "Sports", "Technology", "Entertainment", "Politics"]

# Category-specific vocabulary banks used to build varied, semi-realistic
# news sentences via templates.
VOCAB = {
    "Business": {
        "subjects": ["The company", "Shareholders", "The startup", "The retailer",
                     "The bank", "Investors", "The manufacturer", "The CEO",
                     "The stock market", "The airline"],
        "actions": ["reported record quarterly profits", "announced a merger deal",
                    "cut jobs amid falling revenue", "raised prices on key products",
                    "posted a sharp decline in earnings", "unveiled a new expansion plan",
                    "secured a multi-million dollar investment", "filed for bankruptcy protection",
                    "launched an initial public offering", "signed a major trade agreement"],
        "extras": ["as inflation concerns grow", "following a volatile trading week",
                   "amid rising interest rates", "after a period of slow growth",
                   "in a move welcomed by analysts", "citing supply chain disruptions",
                   "as consumer demand shifts", "ahead of the fiscal year-end report",
                   "despite ongoing economic uncertainty", "boosting investor confidence"],
    },
    "Sports": {
        "subjects": ["The football team", "The tennis champion", "The Olympic committee",
                     "The basketball coach", "The national squad", "The cricket board",
                     "The star striker", "The marathon runner", "The hockey league",
                     "The boxing federation"],
        "actions": ["clinched the championship title", "suffered a shock defeat",
                    "signed a record-breaking contract", "announced the retirement of a star player",
                    "qualified for the world finals", "broke a long-standing world record",
                    "faced a doping investigation", "won the season opener in overtime",
                    "appointed a new head coach", "postponed the match due to weather"],
        "extras": ["in front of a sold-out crowd", "after a dramatic penalty shootout",
                   "following months of intense training", "in a stunning comeback performance",
                   "ahead of next season", "sparking celebrations among fans",
                   "in what was called a historic night", "despite a string of injuries",
                   "in the final minutes of the game", "setting up a thrilling rematch"],
    },
    "Technology": {
        "subjects": ["The tech giant", "The startup founder", "Researchers", "The software company",
                     "The chipmaker", "The social media platform", "Engineers", "The AI lab",
                     "The smartphone maker", "The cybersecurity firm"],
        "actions": ["unveiled a new artificial intelligence model", "launched the latest smartphone",
                    "patched a critical security vulnerability", "released an open-source software update",
                    "announced layoffs in its engineering division", "developed a breakthrough battery technology",
                    "faced backlash over a data privacy issue", "acquired a promising startup",
                    "rolled out a major app update", "warned users about a large-scale data breach"],
        "extras": ["with faster processing speeds", "aimed at improving user privacy",
                   "that could reshape the industry", "after months of testing",
                   "drawing criticism from privacy advocates", "in a bid to compete with rivals",
                   "at a highly anticipated product event", "raising questions about AI safety",
                   "boosting the company's market share", "as demand for cloud services grows"],
    },
    "Entertainment": {
        "subjects": ["The actor", "The film director", "The pop star", "The streaming service",
                     "The award show", "The music festival", "The Hollywood studio",
                     "The celebrity couple", "The comedian", "The television network"],
        "actions": ["announced a new blockbuster film", "topped the music charts",
                    "won the top award of the night", "released a highly anticipated album",
                    "confirmed a surprise reunion tour", "faced backlash over a controversial scene",
                    "signed a lucrative streaming deal", "postponed the premiere indefinitely",
                    "revealed the cast for the upcoming sequel", "broke box office records"],
        "extras": ["to the delight of fans worldwide", "after months of speculation",
                   "in a star-studded ceremony", "drawing mixed reviews from critics",
                   "ahead of the holiday season", "sparking a wave of social media reactions",
                   "in its opening weekend", "following a lengthy production delay",
                   "marking a major career milestone", "amid rumors of a spinoff series"],
    },
    "Politics": {
        "subjects": ["The president", "The prime minister", "Lawmakers", "The senate",
                     "The opposition party", "The governor", "The parliament", "The mayor",
                     "The finance minister", "The election commission"],
        "actions": ["signed a new bill into law", "called for early elections",
                    "proposed sweeping tax reforms", "faced a vote of no confidence",
                    "announced a new foreign policy initiative", "criticized the budget proposal",
                    "held emergency talks with allies", "approved a controversial spending plan",
                    "launched an investigation into corruption allegations", "unveiled a new immigration policy"],
        "extras": ["amid growing public pressure", "ahead of the upcoming election",
                   "sparking protests across the country", "after weeks of negotiations",
                   "drawing criticism from opposition leaders", "in a closely watched parliamentary session",
                   "as tensions rise over the reforms", "in an effort to restore public trust",
                   "following months of political deadlock", "in a widely televised address"],
    },
}


def make_article(category: str) -> str:
    bank = VOCAB[category]
    subject = random.choice(bank["subjects"])
    action = random.choice(bank["actions"])
    extra = random.choice(bank["extras"])
    sentence1 = f"{subject} {action} {extra}."

    subject2 = random.choice(bank["subjects"])
    action2 = random.choice(bank["actions"])
    extra2 = random.choice(bank["extras"])
    sentence2 = f"{subject2} also {action2} {extra2}."

    return f"{sentence1} {sentence2}"


def generate_dataset(samples_per_category: int = 140) -> list:
    rows = []
    for category in CATEGORIES:
        seen = set()
        while len(seen) < samples_per_category:
            text = make_article(category)
            if text not in seen:  # avoid exact duplicates
                seen.add(text)
                rows.append({"text": text, "category": category})
    random.shuffle(rows)
    return rows


def main():
    out_path = os.path.join(os.path.dirname(__file__), "news_dataset.csv")
    rows = generate_dataset(samples_per_category=140)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} articles across {len(CATEGORIES)} categories.")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
