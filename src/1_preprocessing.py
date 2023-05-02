"""Analysis using CPU only."""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
import os
import statistics as stats
from tqdm import tqdm
import numpy as np

# Limit how many product items should be processed
USER_RECORDS_LIMIT = 100000  # TODO include in filenames to reduce necessary number of preprocessing
PREPROCESSED_MATRICES = os.path.join(os.curdir, "data", "preproccessed_" + str(USER_RECORDS_LIMIT) + ".npy")
PREPROCESSED_TFIDF_MATRICES = os.path.join(
    os.curdir, "data", "preproccessed_tfidf_" + str(USER_RECORDS_LIMIT) + ".npy"
)
USERDATA = os.path.join(os.curdir, "data", "preprocessed_users_" + str(USER_RECORDS_LIMIT) + ".json")
RAW_DATA = os.path.join(os.curdir, "data", "kz.csv")

# Load raw data
print("Loading raw CSV data")
file = open(RAW_DATA, "r", encoding="utf-8")
data: list[list[str]] = list(csv.reader(file, delimiter=","))
file.close()

# Get structure from header row, remove header
STRUCTURE: dict[str, int] = {data[0][i]: i for i in range(len(data[0]))}
data = data[1:]

# Fill empty strings for missing category_id & category_code
print("Fixing missing cells in CSV data & converting formats")
for x in tqdm(range(len(data))):
    while len(data[x]) < 8:
        data[x].insert(3, "")
    data[x][STRUCTURE["event_time"]] = (
        datetime.strptime(data[x][STRUCTURE["event_time"]], "%Y-%m-%d %H:%M:%S %Z")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    data[x][STRUCTURE["order_id"]] = int(data[x][STRUCTURE["order_id"]])
    data[x][STRUCTURE["product_id"]] = int(data[x][STRUCTURE["product_id"]])
    if data[x][STRUCTURE["category_id"]].isdigit():
        data[x][STRUCTURE["category_id"]] = int(data[x][STRUCTURE["category_id"]])
    else:
        data[x][STRUCTURE["category_id"]] = -1
    data[x][STRUCTURE["price"]] = float(data[x][STRUCTURE["price"]])
    if data[x][STRUCTURE["user_id"]].isdigit():
        data[x][STRUCTURE["user_id"]] = int(data[x][STRUCTURE["user_id"]])
    else:
        data[x][STRUCTURE["user_id"]] = -1

# Show examples
print("Data example:")
for i in range(10):
    print(data[i])

# Convert to numpy array and sort by userId to speed up processing
dataM = np.array(data, dtype=object)
dataM = dataM[dataM[:, STRUCTURE["user_id"]] != -1]
ind = np.lexsort((dataM[:, STRUCTURE["order_id"]], dataM[:, STRUCTURE["user_id"]]))
dataM = dataM[ind]
dataM = dataM[: min([len(dataM), USER_RECORDS_LIMIT])]

# Load matrices or start creating new ones
dataByUser: defaultdict[int, dict] = defaultdict()
print("Loading created matrices/creating new ones")
# Unique categories and products for BoW / TF-IDF
users = np.array(list(set([x[STRUCTURE["user_id"]] for x in dataM[1:]])))
users.sort()
categories = np.array(list(set([x[STRUCTURE["category_id"]] for x in dataM[1:]])))
categories.sort()
products = np.array(list(set([x[STRUCTURE["product_id"]] for x in dataM[1:]])))
products.sort()
# BoW with shape (n_samples,n_features)
categories_bow = np.zeros([len(users), len(categories)], dtype=np.int16)
products_bow = np.zeros([len(users), len(products)], dtype=np.int16)
print("Sorting by user: ")
lastUserId: int | None = None
lastOrderId: int | None = None
for x in tqdm(range(len(dataM))):
    userId = dataM[x][STRUCTURE["user_id"]]
    if userId != lastUserId:
        lastUserId = userId
        user = {}
        orders = {}
        uind = np.searchsorted(users, userId)
    orderId = dataM[x][STRUCTURE["order_id"]]
    if orderId != lastOrderId:
        lastOrderId = orderId
        order = {}
        order["event_time"] = dataM[x][STRUCTURE["event_time"]]
        content: list[dict[str, int | str]] = []
    content.append(
        {
            "product_id": dataM[x][STRUCTURE["product_id"]],
            "category_id": dataM[x][STRUCTURE["category_id"]],
            "category_code": dataM[x][STRUCTURE["category_code"]],
            "brand": dataM[x][STRUCTURE["brand"]],
            "price": dataM[x][STRUCTURE["price"]],
        }
    )
    order["content"] = content
    order["total_price"] = sum(map(lambda x: x["price"], content))
    orders[dataM[x][STRUCTURE["order_id"]]] = order
    user["orders"] = orders
    user["first_purchase"] = min(map(lambda x: x["event_time"], list(orders.values())))
    user["last_purchase"] = max(map(lambda x: x["event_time"], list(orders.values())))
    user["total_spent"] = sum(map(lambda x: x["total_price"], list(orders.values())))
    user["purchase_count"] = len(orders.items())
    user["avg_purchase"] = stats.mean(map(lambda x: x["total_price"], list(orders.values())))
    dataByUser[userId] = user
    # Fill in BoW
    if userId != -1:
        pind = np.searchsorted(products, content[-1]["product_id"])
        products_bow[uind, pind] += 1
        if content[-1]["category_id"] != -1:
            cind = np.searchsorted(categories, content[-1]["category_id"])
            categories_bow[uind, cind] += 1

# Remove products & categories that do not belong to any user
pinds = np.where(np.sum(products_bow, 0) == 0)[0]  # find zero sum product columns
products_bow = np.delete(products_bow, pinds, 1)  # delete product column
products = np.delete(products, pinds)
cinds = np.where(np.sum(categories_bow, 0) == 0)[0]
categories_bow = np.delete(categories_bow, cinds, 1)  # delete category column
categories = np.delete(categories, cinds)

# Remove users without items
uinds = np.where(np.sum(products_bow, 1) == 0)[0]  # find zero sum product columns
products_bow = np.delete(products_bow, uinds, 0)  # delete product column
products_users = np.delete(users, uinds)
uinds = np.where(np.sum(categories_bow, 1) == 0)[0]
categories_bow = np.delete(categories_bow, uinds, 0)
categories_users = np.delete(users, uinds)


# Save created matrices to save time next time
print("Saving BOW matrices")
with open(PREPROCESSED_MATRICES, "wb") as f:
    np.save(f, products_users)
    np.save(f, products)
    np.save(f, products_bow)
    np.save(f, categories_users)
    np.save(f, categories)
    np.save(f, categories_bow)

# Compoute TF-IDF
products_tf = products_bow / np.sum(products_bow, 1).reshape([products_bow.shape[0], 1])
products_idf = np.log10(products_bow.shape[0] / np.sum(products_bow > 0, 0)).reshape([1, products_bow.shape[1]])
products_tfidf = products_tf * products_idf
categories_tf = categories_bow / np.sum(categories_bow, 1).reshape([categories_bow.shape[0], 1])
categories_idf = np.log10(categories_bow.shape[0] / np.sum(categories_bow > 0, 0)).reshape(
    [1, categories_bow.shape[1]]
)
categories_tfidf = categories_tf * categories_idf

# Save created matrices to save time next time
print("Saving TF-IDF matrices")
with open(PREPROCESSED_TFIDF_MATRICES, "wb") as f:
    np.save(f, products_tfidf)
    np.save(f, categories_tfidf)

print("Sorted data example:")
usersIDs: list[int] = list(dataByUser.keys())
for i in range(3):
    print(json.dumps(dataByUser.get(usersIDs[i]), indent=2))

print("Saving user records")
with open(USERDATA, "wt", encoding="utf-8") as f:
    dataByUserJSON = json.dumps(dataByUser, indent=2)
    for l in tqdm(dataByUserJSON.split("\n")):
        f.write(l + "\n")

print("Finished.")
