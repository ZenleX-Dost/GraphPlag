#!/usr/bin/env python3
"""Analyze what makes the test texts look AI or human"""

test_cases = {
    "ChatGPT Example": """The intersection of artificial intelligence and human creativity represents one of the most compelling frontiers in contemporary technology. As machine learning algorithms continue to evolve in sophistication and capability, the question of whether AI systems can genuinely create original content has transitioned from theoretical speculation to practical reality. This paradigm shift necessitates a careful examination of the criteria by which we evaluate authenticity and originality in an increasingly digital world.""",
    
    "Formal AI Style": """Artificial intelligence systems have demonstrated remarkable proficiency in natural language generation tasks. The deployment of transformer-based architectures has fundamentally altered our understanding of computational linguistic capability. Contemporary models exhibit behavior patterns that closely approximate human-level performance across numerous benchmarks, thereby challenging conventional assumptions about the nature of human cognitive uniqueness.""",
    
    "ChatGPT Intro Pattern": """I appreciate your interest in this topic. As an AI language model, I can provide insights into the mechanisms underlying modern machine learning. The exponential growth in computational resources, coupled with innovations in neural network design, has enabled unprecedented advances in artificial intelligence applications. These developments carry profound implications for society at large.""",
}

transitions = [
    "furthermore", "moreover", "additionally", "consequently",
    "ultimately", "notably", "specifically", "therefore",
    "hence", "thus", "accordingly", "as such",
    "in conclusion", "to summarize", "in summary",
    "it must be noted", "it is worth noting", "it is important",
    "the fact that", "the notion that", "the concept of",
    "in light of", "by contrast", "in contrast", "on the other hand",
    "it is clear", "it is evident", "it is important to note"
]

for name, text in test_cases.items():
    text_lower = text.lower()
    word_count = len(text.split())
    
    found = []
    for t in transitions:
        if t in text_lower:
            found.append(t)
            count = text_lower.count(t)
            if count > 1:
                found[-1] += f" ({count}x)"
    
    print(f"\n{name}:")
    print(f"  Word count: {word_count}")
    print(f"  Transition words found: {found if found else 'NONE'}")
    print(f"  Count: {len(found)}")
    if found:
        print(f"  Formula: {len(found)} / ({word_count} / 40) = {len(found) / (word_count / 40):.2f}")
