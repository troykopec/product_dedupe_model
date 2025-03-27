import json
from subtype_guess import deduplicate_product  # Ensure deduplicate_product is exported from subtype_guess.py

use_exact_match = False # Turn this off in testing against dataset, 
                       # this should only be used in prod as we want to 
                       # test the models search abilities without exact matching
                       # with the data we just created and are testing against

def test_accuracy(data, new_brand_threshold=10):
    total_products = 0
    false_count = 0
    correct_brand = 0
    correct_subgroup = 0

    # Iterate over each brand (the key is the true brand)
    for true_brand, brand_info in data.items():
        for subgroup in brand_info.get("subgroups", []):
            true_subgroup = subgroup.get("display_name", "")
            for product in subgroup.get("products", []):
                total_products += 1
                product_name = product.get("name", "")
                prediction = deduplicate_product(product_name, data, use_exact_match, new_brand_threshold)
                pred_brand = prediction.get("brand")
                pred_subgroup = prediction.get("subgroup")
                
                # Determine correctness.
                brand_correct = (pred_brand == true_brand)
                subgroup_correct = (pred_subgroup == true_subgroup)
                
                if brand_correct and subgroup_correct:
                    correct_brand += 1
                    correct_subgroup += 1
                else:
                    false_count += 1
                    print("Product:", product_name)
                    print("True -> Brand:", true_brand, ", Subgroup:", true_subgroup)
                    print("Predicted -> Brand:", pred_brand, ", Subgroup:", pred_subgroup)
                    print("-" * 50)
    
    brand_accuracy = (correct_brand / total_products) * 100 if total_products else 0
    subgroup_accuracy = (correct_subgroup / total_products) * 100 if total_products else 0
    print("Total products tested:", total_products)
    print("False predictions:", false_count)
    print("Brand Accuracy: {:.2f}%".format(brand_accuracy))
    print("Subgroup Accuracy: {:.2f}%".format(subgroup_accuracy))

if __name__ == "__main__":
    # Load your ground truth JSON data.
    with open("product_groupings_v2.json", encoding="utf-8") as f:
        data = json.load(f)
    
    # You can adjust the new_brand_threshold if needed.
    test_accuracy(data, new_brand_threshold=10)
