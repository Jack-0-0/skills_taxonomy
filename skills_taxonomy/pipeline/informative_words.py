"""These functions can be used to help name the skill clusters and
sub clusters as the clustering process does not name them.

To help name the classes we can use class-TF-IDF to find the most
informative words in each class and subclass.

TF-IDF is usually used to find informative words inside a document.
But in our case we want to find informative words across a class or sub class.
To do this we form a 'document' containing all texts from a class
and then perform TF-IDF across on that 'document'.

c_tf_idf and extract_top_n_words_per_topic below are from this
implementation of class-TD-IDF https://github.com/MaartenGr/cTFIDF
"""
from sklearn.feature_extraction.text import CountVectorizer
from skills_taxonomy.utils.dir_management import make_dir_if_not_exist
import json
import numpy as np
from skills_taxonomy import PROJECT_DIR


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(
        documents
    )
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, classes, n=20):
    words = count.get_feature_names()
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {
        label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1]
        for i, label in enumerate(classes)
    }
    return top_n_words


def view_top_n_words(skills, class_level):
    skills_classid = skills[["description", class_level]]
    skills_classid["skill_id"] = range(len(skills_classid))
    skills_per_class = skills_classid.groupby([class_level], as_index=False).agg(
        {"description": " ".join}
    )
    classes = skills_per_class[class_level]
    tf_idf_class, count_class = c_tf_idf(
        skills_per_class.description.values, m=len(skills)
    )
    top_n_words = extract_top_n_words_per_topic(tf_idf_class, count_class, classes)
    return top_n_words


def save_informative_words(
    skills, save_dir=f"{PROJECT_DIR}/outputs/most_informative_words/"
):
    make_dir_if_not_exist(save_dir)

    # save most informative words for classes
    with open(f"{save_dir}/class.json", "w") as fp:
        json.dump(
            view_top_n_words(skills, class_level="class_id"),
            fp,
            sort_keys=True,
            indent=4,
        )

    # save most informative words for subclasses
    with open(f"{save_dir}/subclass.json", "w") as fp:
        json.dump(
            view_top_n_words(skills, class_level="subclass_id"),
            fp,
            sort_keys=True,
            indent=4,
        )
