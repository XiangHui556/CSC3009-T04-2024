import os


def get_categories(data_dir):
    categories = os.listdir(data_dir)
    print("Categories:", categories)

    # for category in categories:
    #     print(
    #         f"Number of images in {category}: {len(os.listdir(os.path.join(data_dir, category)))}"
    #     )
    return categories


# data_dir = "./dataset_4"
# get_categories(data_dir)
