"""Analysis using CPU only."""
import os
import json
import numpy as np
from apyori import apriori, RelationRecord

# Limit how many product items should be processed
USER_RECORDS_LIMIT = 100000  # TODO include in filenames to reduce necessary number of preprocessing
PREPROCESSED_MATRICES = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "..", "..", "data", "preproccessed_" + str(USER_RECORDS_LIMIT) + ".npy")
)
PREPROCESSED_TFIDF_MATRICES = os.path.abspath(
    os.path.join(
        os.path.realpath(__file__), "..", "..", "data", "preproccessed_tfidf_" + str(USER_RECORDS_LIMIT) + ".npy"
    )
)
PREPROCESSED_PCA_MATRICES = os.path.abspath(
    os.path.join(
        os.path.realpath(__file__), "..", "..", "data", "preproccessed_pca_" + str(USER_RECORDS_LIMIT) + ".npy"
    )
)
ASSIGNED_LABELS = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "..", "..", "data", "assigned_labels_" + str(USER_RECORDS_LIMIT) + ".npy")
)


def rule_to_json_obj(rule: RelationRecord) -> str:
    ruleset = {"items": list(rule.items), "support": rule.support, "rules": []}
    for stat in rule.ordered_statistics:
        x = {
            "antecedent": list(stat.items_base),
            "consequent": list(stat.items_add),
            "lift": stat.lift,
            "confidence": stat.confidence,
        }
        ruleset["rules"].append(x)
    return json.dumps(ruleset, indent=2)


if not os.path.exists(PREPROCESSED_MATRICES) or not os.path.exists(PREPROCESSED_TFIDF_MATRICES):
    print("Preprocessed files unavailable. Do preprocessing first.")

print("Preprocessed files found, loading...")
with open(PREPROCESSED_MATRICES, "rb") as f:
    print("Loading products users...")
    products_users = np.load(f)
    print("Loading products...")
    products = np.load(f)
    print("Loading products BoW...")
    products_bow = np.load(f)
    print("Loading categories users...")
    categories_users = np.load(f)
    print("Loading categories...")
    categories = np.load(f)
    print("Loading categories BoW..")
    categories_bow = np.load(f)

with open(PREPROCESSED_TFIDF_MATRICES, "rb") as f:
    print("Loading products TF-IDF...")
    products_tfidf = np.load(f)
    print("Loading categories TF-IDF...")
    categories_tfidf = np.load(f)


with open(ASSIGNED_LABELS, "rb") as f:
    print("Loading products labels...")
    products_labels = np.load(f)
    print("Loading categories labels...")
    categories_labels = np.load(f)

print("Loading DONE.")

for label in np.unique(products_labels)[1:]:
    print("")
    print("Processing product cluster " + str(label))
    print("------------------------------")
    cluster_products = products_bow[products_labels == label]
    cluster_user_products = (cluster_products > 0) * products
    cluster_user_product_sets = set(
        [
            frozenset(
                [
                    str(cluster_user_products[i, j])
                    for j in range(cluster_user_products.shape[1])
                    if cluster_user_products[i, j] > 0
                ]
            )
            for i in range(cluster_user_products.shape[0])
        ]
    )
    for cs in cluster_user_product_sets:
        print(cs)
    cluster_user_products_lists = [
        [
            str(cluster_user_products[i, j])
            for j in range(cluster_user_products.shape[1])
            if cluster_user_products[i, j] > 0
        ]
        for i in range(cluster_user_products.shape[0])
    ]
    # results = apriori(cluster_user_products_lists, min_support=0.1, min_lift=1.0, min_confidence=0.7)
    apriori_settings = {"min_support": 0.01, "min_lift": 1.001, "min_confidence": 0.5}
    results = apriori(
        cluster_user_products_lists,
        min_support=apriori_settings["min_support"],
        min_lift=apriori_settings["min_lift"],
        min_confidence=apriori_settings["min_confidence"],
    )

    print("Saving users of product cluster " + str(label))
    product_cluster_users_file = os.path.abspath(
        os.path.join(os.path.realpath(__file__), "..", "..", "rules", "product_cluster_" + str(label) + "_info.json")
    )
    with open(product_cluster_users_file, "wt", encoding="utf-8") as f:
        pcu = {
            "users": [str(x) for x in products_users[products_labels == label]],
            "settings": apriori_settings,
            "user_products": [list(x) for x in cluster_user_product_sets],
        }
        json_object = json.dumps(pcu, indent=2)
        f.write(json_object)

    result_list = list(results)
    print("Number of found rules: " + str(len(result_list)))
    for i in range(len(result_list)):  # pylint: disable=consider-using-enumerate
        print("---------------------------")
        print("Product Result " + str(label) + "-" + str(i))
        print("Items: " + str(result_list[i].items))
        print("Support: " + str(result_list[i].support))
        for j in range(len(result_list[i].ordered_statistics)):  # pylint: disable=consider-using-enumerate
            # if result_list[i].ordered_statistics[j].lift == 1:
            #     continue
            print("Statistics " + str(label) + "-" + str(i) + "-" + str(j))
            print("Items base> " + str(result_list[i].ordered_statistics[j].items_base))
            print("Items add> " + str(result_list[i].ordered_statistics[j].items_add))
            print("Items confidence> " + str(result_list[i].ordered_statistics[j].confidence))
            print("Items lift> " + str(result_list[i].ordered_statistics[j].lift))
        # Save the rule
        # if all([x.lift == 1 for x in result_list[i].ordered_statistics]):
        #     continue
        with open(
            os.path.abspath(
                os.path.join(
                    os.path.realpath(__file__),
                    "..",
                    "..",
                    "rules",
                    "product_cluster_" + str(label) + "_ruleset_" + str(i) + ".json",
                )
            ),
            "wt",
            encoding="utf-8",
        ) as f:
            f.write(rule_to_json_obj(result_list[i]))


for label in np.unique(categories_labels)[1:]:
    print("")
    print("Processing category cluster " + str(label))
    print("------------------------------")
    cluster_categories = categories_bow[categories_labels == label]
    cluster_user_categories = (cluster_categories > 0) * categories
    cluster_user_category_sets = set(
        [
            frozenset(
                [
                    str(cluster_user_categories[i, j])
                    for j in range(cluster_user_categories.shape[1])
                    if cluster_user_categories[i, j] > 0
                ]
            )
            for i in range(cluster_user_categories.shape[0])
        ]
    )
    for cs in cluster_user_category_sets:
        print(cs)
    cluster_user_categories_lists = [
        [
            str(cluster_user_categories[i, j])
            for j in range(cluster_user_categories.shape[1])
            if cluster_user_categories[i, j] > 0
        ]
        for i in range(cluster_user_categories.shape[0])
    ]
    # results = apriori(cluster_user_categories_lists, min_support=0.1, min_lift=1.0, min_confidence=0.7)
    apriori_settings = {"min_support": 0.015, "min_lift": 1.001, "min_confidence": 0.5}
    results = apriori(
        cluster_user_categories_lists,
        min_support=apriori_settings["min_support"],
        min_lift=apriori_settings["min_lift"],
        min_confidence=apriori_settings["min_confidence"],
    )

    print("Saving users of category cluster " + str(label))
    category_cluster_users_file = os.path.abspath(
        os.path.join(os.path.realpath(__file__), "..", "..", "rules", "category_cluster_" + str(label) + "_info.json")
    )
    with open(category_cluster_users_file, "wt", encoding="utf-8") as f:
        pcu = {
            "users": [str(x) for x in categories_users[categories_labels == label]],
            "settings": apriori_settings,
            "user_categories": [list(x) for x in cluster_user_category_sets],
        }
        json_object = json.dumps(pcu, indent=2)
        f.write(json_object)

    result_list = list(results)
    print("Number of found rules: " + str(len(result_list)))
    for i in range(len(result_list)):  # pylint: disable=consider-using-enumerate
        print("---------------------------")
        print("Category Result " + str(label) + "-" + str(i))
        print("Items: " + str(result_list[i].items))
        print("Support: " + str(result_list[i].support))
        for j in range(len(result_list[i].ordered_statistics)):  # pylint: disable=consider-using-enumerate
            # if result_list[i].ordered_statistics[j].lift == 1:
            #     continue
            print("Statistics " + str(label) + "-" + str(i) + "-" + str(j))
            print("Items base> " + str(result_list[i].ordered_statistics[j].items_base))
            print("Items add> " + str(result_list[i].ordered_statistics[j].items_add))
            print("Items confidence> " + str(result_list[i].ordered_statistics[j].confidence))
            print("Items lift> " + str(result_list[i].ordered_statistics[j].lift))
        # if all([x.lift == 1 for x in result_list[i].ordered_statistics]):
        #     continue
        with open(
            os.path.abspath(
                os.path.join(
                    os.path.realpath(__file__),
                    "..",
                    "..",
                    "rules",
                    "category_cluster_" + str(label) + "_ruleset_" + str(i) + ".json",
                )
            ),
            "wt",
            encoding="utf-8",
        ) as f:
            f.write(rule_to_json_obj(result_list[i]))

print("Finished.")
