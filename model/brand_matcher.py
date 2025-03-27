import json
import unicodedata
import re
import ahocorasick

MINIMUM_BRAND_SCORE = 5   # Lowered threshold
BONUS_PER_TOKEN = 5        # Bonus per token remains the same

def normalize_text(text):
    """
    Normalize text by converting to lowercase, removing accents,
    stripping punctuation, and collapsing whitespace.
    """
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode("utf-8")
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
        # Remove parenthesized content
    text = re.sub(r'\([^)]*\)', ' ', text)
    # Remove tokens that match size patterns (e.g. "750 ml", "1.75 l", "12 x 12 fl oz")
    text = re.sub(r'\b\d+(\.\d+)?\s*(ml|l|fl oz|x)\b', ' ', text, flags=re.IGNORECASE)
    # Normalize Unicode and remove punctuation
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text

def is_full_token(product_norm, start_index, end_index):
    """
    Returns True if the substring product_norm[start_index:end_index+1]
    is a full token (i.e. preceded and followed by a space or at a boundary).
    """
    if start_index > 0 and product_norm[start_index - 1] != " ":
        return False
    if end_index < len(product_norm) - 1 and product_norm[end_index + 1] != " ":
        return False
    return True

def build_brand_automaton(data):
    """
    Builds an Aho-Corasick automaton from all normalized brand names and their variations.
    For each pattern, we store a tuple: (canonical_brand, length_of_pattern).
    Also builds a dictionary mapping canonical brands to a list of all their normalized variations.
    """
    automaton = ahocorasick.Automaton()
    brand_variations_map = {}
    for brand, info in data.items():
        norm_brand = normalize_text(brand)
        variations = [norm_brand]
        for alias in info.get("brand_variations", []):
            variations.append(normalize_text(alias))
        variations = list(set(variations))
        brand_variations_map[brand] = variations
        for var in variations:
            if var:
                automaton.add_word(var, (brand, len(var)))
    automaton.make_automaton()
    return automaton, brand_variations_map

# Load the JSON dataset (root keys are the brand names)
with open("product_groupings_v2.json", encoding="utf-8") as f:
    data = json.load(f)

# Build the automaton and brand variations map.
automaton, brand_variations_map = build_brand_automaton(data)

def match_brand(product_name):
    """
    Matches a product name to candidate brands using the pre-built Aho-Corasick automaton.
    Only counts a match if it is a full-token match.
    
    Then, for each candidate brand, it applies a bonus: for each token in any of its normalized variations
    that is present in the product's tokens, add BONUS_PER_TOKEN points.
    
    If the highest cumulative score is below MINIMUM_BRAND_SCORE, returns an empty dictionary.
    
    Returns:
      A dictionary mapping candidate brand to its cumulative score.
    """
    product_norm = normalize_text(product_name)
    product_tokens = set(product_norm.split())
    matched_scores = {}
    
    # Gather base scores from the automaton.
    for end_index, (brand, pattern_length) in automaton.iter(product_norm):
        start_index = end_index - pattern_length + 1
        if not is_full_token(product_norm, start_index, end_index):
            continue
        weight = pattern_length
        matched_scores[brand] = matched_scores.get(brand, 0) + weight
    
    # Apply bonus: for each candidate, for each variation, if all tokens of the variation appear in product_tokens,
    # add bonus = BONUS_PER_TOKEN * (# tokens in the variation). Use the maximum bonus among variations.
    for candidate in list(matched_scores.keys()):
        bonus = 0
        variations = brand_variations_map.get(candidate, [])
        for var in variations:
            var_tokens = set(var.split())
            if var_tokens.issubset(product_tokens):
                bonus = max(bonus, BONUS_PER_TOKEN * len(var_tokens))
        matched_scores[candidate] += bonus

    if matched_scores and max(matched_scores.values()) < MINIMUM_BRAND_SCORE:
        return {}
    
    return matched_scores

if __name__ == "__main__":
    user_input = input("Enter the drink product name to match: ")
    matches = match_brand(user_input)
    if matches:
        print("Candidate brand matches (with scores):", matches)
        best_brand = max(matches, key=matches.get)
        print("Best matching brand:", best_brand, f"(score: {matches[best_brand]})")
    else:
        print("No matching brand found.")
