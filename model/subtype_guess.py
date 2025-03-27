import json
import csv
from brand_matcher import match_brand, normalize_text
import unicodedata
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging (set level=DEBUG for full logs; here set to ERROR by default)
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

MINIMUM_BRAND_SCORE = 10
use_exact_match = False  # For testing; in production, set to True

###############################################
# FLAVOR AND PACK KEYWORDS
###############################################
FLAVOR_KEYWORDS = {
    "black cherry", "pineapple", "apple", "strawberry", "vanilla", "cinnamon",
    "peach", "cherry", "orange", "lemon", "coconut", "honey",
    "blueberry", "raspberry", "mango", "passion fruit", "guava", "watermelon",
    "kiwi", "lime", "grapefruit", "cranberry", "pomegranate", "apricot", "banana",
    "pear", "plum", "elderberry", "mulberry", "tangerine", "mandarin",
    "fig", "date", "caramel", "butterscotch", "almond", "hazelnut", "macadamia",
    "chocolate", "mocha", "coffee", "espresso", "toffee", "smoke", "oak", "maple",
    "spice", "pepper", "clove", "nutmeg", "ginger", "anise", "licorice",
    "sage", "rosemary", "thyme", "basil", "bay", "turmeric",
    "sour apple", "green apple", "red apple", "crabapple", "blackberry", "boysenberry",
    "currant", "kumquat", "lychee", "nectarine", "cantaloupe", "honeydew",
    "sour cherry", "sweet cherry", "blood orange", "cara cara orange",
    "sour lemon", "sweet lemon", "bergamot", "iced tea"
    "eucalyptus", "mint", "peppermint", "spearmint", "lavender", "rose",
    "jasmine", "violet", "chamomile", "elderflower", "cucumber",
    "tamarind", "saffron", "sesame", "wasabi", "miso", "umami",
    "salty", "briny", "creamy",
    "butter", "yogurt", "cheese", "nutty", "pistachio", "walnut", "cashew",
    "pecan", "sunflower", "pumpkin", "chili", "sriracha",
    "red tag", "red stag", "double oak", "devils cut", "winter reserve", "iced tea", "ice tea"
    "vanilla flavored", "cherry flavor", "devil", "single barrel", "cask strength", "cask", "private select", "personal selection", "personal select"
}
# Synonyms for pack phrases. All of these map to the canonical term "party pack".
PACK_SYNONYMS = {
    "party pack": "party pack",
    "variety pack": "party pack",
    "assorted": "party pack"
}

def canonicalize_pack_tokens(text):
    """Replace pack synonyms with a canonical term."""
    norm_text = normalize_text(text)
    for synonym, canonical in PACK_SYNONYMS.items():
        if synonym in norm_text:
            norm_text = norm_text.replace(synonym, canonical)
    return norm_text

def extract_flavor_words(text):
    """
    Extracts flavor words from the text.
    Uses FLAVOR_KEYWORDS to search within the normalized text.
    """
    norm_text = normalize_text(text)
    found = set()
    for flavor in FLAVOR_KEYWORDS:
        if flavor in norm_text:
            found.add(flavor)
    return found

###############################################
# EXACT-MATCH INDEX
###############################################
def build_exact_product_index(data):
    index = {}
    for brand, brand_info in data.items():
        for subgroup in brand_info.get("subgroups", []):
            true_subgroup = subgroup.get("display_name", "")
            for prod in subgroup.get("products", []):
                prod_name = prod.get("name", "")
                norm_name = normalize_text(prod_name)
                index[norm_name] = (brand, true_subgroup)
    return index

with open("product_groupings_v2.json", encoding="utf-8") as f:
    data = json.load(f)
exact_product_index = build_exact_product_index(data)

def check_exact_product_match(product_input):
    norm_input = normalize_text(product_input)
    return exact_product_index.get(norm_input, (None, None))

###############################################
# SUBGROUP MATCHING (PER-PRODUCT VARIANT)
###############################################
def match_subgroup(product_name, brand_info):
    """
    For each candidate subgroup in the brand, extract all product names (and include the subgroup display name),
    normalize them, and deduplicate the results. Then, for each unique variant, compute:
         score = (# tokens in candidate that are in the product)
                 - (# tokens in candidate missing from the product)
    Return the subgroup display name and the maximum variant score found.
    """
    product_norm = normalize_text(product_name)
    product_tokens = set(product_norm.split())
    logging.debug(f"Product tokens: {product_tokens}")

    best_overall_score = -float('inf')
    best_subgroup = None

    for subgroup in brand_info.get("subgroups", []):
        display = subgroup.get("display_name", "")
        # Gather all variant names (from "products") plus the display name.
        variants = {prod.get("name", "") for prod in subgroup.get("products", [])}
        variants.add(display)
        logging.debug(f"Subgroup '{display}' variants: {variants}")

        max_variant_score = -float('inf')
        for variant in variants:
            variant_norm = normalize_text(variant)
            candidate_tokens = set(variant_norm.split())
            common = candidate_tokens & product_tokens
            extra = candidate_tokens - product_tokens
            score = len(common) - len(extra)
            logging.debug(f"Variant '{variant}': tokens = {candidate_tokens}, common = {common}, extra = {extra}, score = {score}")
            if score > max_variant_score:
                max_variant_score = score
        logging.debug(f"Subgroup '{display}': best variant score = {max_variant_score}")
        if max_variant_score > best_overall_score:
            best_overall_score = max_variant_score
            best_subgroup = display
    return best_subgroup, best_overall_score

###############################################
# CANDIDATE OVERALL MATCHES
###############################################
def get_candidate_overall_matches(product_input, data):
    candidate_brands = match_brand(product_input)
    overall_matches = []
    for candidate_brand, brand_score in candidate_brands.items():
        if candidate_brand in data:
            brand_info = data[candidate_brand]
            if brand_info.get("subgroups"):
                subgroup, subgroup_score = match_subgroup(product_input, brand_info)
            else:
                subgroup, subgroup_score = None, 0
            overall_score = brand_score + subgroup_score
            overall_matches.append((candidate_brand, subgroup, brand_score, subgroup_score, overall_score))
        else:
            overall_matches.append((candidate_brand, None, brand_score, 0, brand_score))
    
    overall_matches.sort(key=lambda x: x[4], reverse=True)
    return overall_matches

###############################################
# COSINE SIMILARITY RE-RANKING
###############################################
def compute_cosine_similarity(product_input, candidate_brand, candidate_subgroup):
    candidate_string = candidate_brand
    if candidate_subgroup:
        candidate_string += " " + candidate_subgroup
    product_norm = normalize_text(product_input)
    candidate_norm = normalize_text(candidate_string)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([product_norm, candidate_norm])
    sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return sim

###############################################
# REFINEMENT STAGE: Remove Shared Words
###############################################
def refine_candidate_subgroups(product_input, overall_matches):
    """
    Refines candidate subgroup matches by:
      1. Computing the normalized token set for each candidate subgroup.
      2. Finding the common tokens shared across all candidates.
      3. For each candidate, subtract these common tokens to get its distinctive tokens.
      4. Compute a refined score as the count of distinctive tokens that appear in the product.
         (If a candidate's subgroup becomes empty after removing common tokens, refined score = 0.)
    Returns a list of tuples:
       (candidate_brand, candidate_subgroup, brand_score, subgroup_score, overall_score, refined_score, distinctive_tokens)
    sorted in descending order of refined_score.
    """
    candidate_token_sets = []
    for candidate in overall_matches:
        candidate_brand, candidate_subgroup, _, _, _ = candidate
        if candidate_subgroup:
            tokens = set(normalize_text(candidate_subgroup).split())
        else:
            tokens = set()
        candidate_token_sets.append(tokens)
        logging.debug(f"Candidate '{candidate_brand} - {candidate_subgroup}': tokens = {tokens}")
    
    if candidate_token_sets:
        common_tokens = set.intersection(*candidate_token_sets)
    else:
        common_tokens = set()
    logging.debug(f"Common tokens across all candidates: {common_tokens}")
    
    product_tokens = set(normalize_text(product_input).split())
    logging.debug(f"Product tokens: {product_tokens}")
    
    refined_matches = []
    for candidate, tokens in zip(overall_matches, candidate_token_sets):
        candidate_brand, candidate_subgroup, brand_score, subgroup_score, overall_score = candidate
        distinctive_tokens = tokens - common_tokens
        refined_score = len(distinctive_tokens & product_tokens)
        logging.debug(f"Candidate '{candidate_brand} - {candidate_subgroup}': distinctive tokens = {distinctive_tokens}, refined score = {refined_score}")
        refined_matches.append((candidate_brand, candidate_subgroup, brand_score, subgroup_score, overall_score, refined_score, distinctive_tokens))
    
    refined_matches.sort(key=lambda x: x[5], reverse=True)
    logging.debug(f"Refined matches sorted by refined score: {refined_matches}")
    return refined_matches

###############################################
# NEW FLAVOR DETECTION LOGIC
###############################################
def detect_new_flavor(product_input, candidate_subgroup):
    """
    Extracts flavor tokens from the product input and candidate subgroup.
    If the product contains a flavor word that is not present in the candidate subgroup,
    return True (indicating a new flavor subgroup should be created), otherwise False.
    """
    product_flavors = extract_flavor_words(product_input)
    candidate_flavors = extract_flavor_words(candidate_subgroup) if candidate_subgroup else set()
    logging.debug(f"Product flavors: {product_flavors}")
    logging.debug(f"Candidate flavors: {candidate_flavors}")
    # If there is at least one flavor in the product that is not in the candidate, flag as new.
    if product_flavors and not (product_flavors & candidate_flavors):
        return True
    return False

###############################################
# DEDUPLICATION PIPELINE
###############################################
def deduplicate_product(product_input, data, use_exact_match, new_brand_threshold=10, advanced_threshold=2, use_refinement=True):
    # 1. Exact match check.
    if use_exact_match:
        exact_brand, exact_subgroup = check_exact_product_match(product_input)
        if exact_brand is not None:
            return {
                "brand": exact_brand,
                "subgroup": exact_subgroup,
                "brand_score": None,
                "subgroup_score": None,
                "total_score": None,
                "overall_matches": [("Exact Match", exact_brand, exact_subgroup)],
                "new_flavor": False
            }
    
    # 2. Get candidate overall matches.
    overall_matches = get_candidate_overall_matches(product_input, data)
    if not overall_matches:
        return {"brand": None, "subgroup": None, "overall_matches": [], "new_flavor": False}
    
    # 3. Re-rank with cosine similarity if the top two candidates are close.
    if len(overall_matches) > 1 and (overall_matches[0][4] - overall_matches[1][4]) < advanced_threshold:
        advanced_candidates = []
        for candidate in overall_matches:
            candidate_brand, candidate_subgroup, brand_score, subgroup_score, overall_score = candidate
            cosine_sim = compute_cosine_similarity(product_input, candidate_brand, candidate_subgroup)
            combined_score = overall_score + (cosine_sim * 10)
            advanced_candidates.append((candidate_brand, candidate_subgroup, brand_score, subgroup_score, overall_score, combined_score))
        advanced_candidates.sort(key=lambda x: x[5], reverse=True)
        best_candidate = advanced_candidates[0]
        best_match = best_candidate[:5]
    else:
        best_match = overall_matches[0]
    
    # 4. Optionally refine candidate subgroup matches.
    if use_refinement:
        refined_candidates = refine_candidate_subgroups(product_input, overall_matches)
        if refined_candidates and refined_candidates[0][5] > 0:
            combined_candidates = []
            for cand in refined_candidates:
                candidate_brand, candidate_subgroup, brand_score, subgroup_score, overall_score, refined_score, distinct_tokens = cand
                combined = overall_score + (5 * refined_score)
                combined_candidates.append((candidate_brand, candidate_subgroup, brand_score, subgroup_score, overall_score, refined_score, combined, distinct_tokens))
            combined_candidates.sort(key=lambda x: x[6], reverse=True)
            best_candidate = combined_candidates[0]
            best_match = best_candidate[:5]
    
    # 5. Flavor detection: if the best candidate's subgroup does not contain any flavor word
    #    that is present in the product, flag new flavor by setting subgroup to None.
    if best_match[1]:
        if detect_new_flavor(product_input, best_match[1]):
            best_match = (best_match[0], None, best_match[2], best_match[3], best_match[4])
            new_flavor = True
        else:
            new_flavor = False
    else:
        new_flavor = False
    
    # 6. Threshold check.
    if best_match[4] < new_brand_threshold:
        return {"brand": None, "subgroup": None, "overall_matches": overall_matches, "new_flavor": new_flavor}
    else:
        return {
            "brand": best_match[0],
            "subgroup": best_match[1],
            "brand_score": best_match[2],
            "subgroup_score": best_match[3],
            "total_score": best_match[4],
            "overall_matches": overall_matches,
            "new_flavor": new_flavor
        }

###############################################
# MAIN
###############################################
if __name__ == "__main__":
    product_input = input("Enter the drink product name to match: ")
    
    overall_matches = get_candidate_overall_matches(product_input, data)
    best_prediction = deduplicate_product(product_input, data, use_exact_match)
    
    print("\nOverall candidate matches (Brand, Subgroup, Brand Score, Subgroup Score, Total Score):")
    for candidate in overall_matches:
        print(candidate)
    
    print("\nBest overall match:")
    if best_prediction["brand"] is None:
        print("No strong match found. This product might belong to a new brand or subgroup.")
    else:
        print(f"Brand: {best_prediction['brand']}")
        print(f"Subgroup: {best_prediction['subgroup']}")
        print(f"Brand Score: {best_prediction['brand_score']}, "
              f"Subgroup Score: {best_prediction['subgroup_score']}, "
              f"Total Score: {best_prediction['total_score']}")
        if best_prediction["new_flavor"]:
            print("New flavor detected: the product has flavor words that none of the candidate subgroups have.")
