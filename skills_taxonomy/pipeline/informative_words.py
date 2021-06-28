"""These functions can be used to help name the skill clusters and
sub clusters as the clustering process does not name them.

To help name the classes we can use class-TF-IDF to find the most
informative words in each class and subclass.

TF-IDF is usually used to find informative words inside a document.
But in our case we want to find informative words across a class or sub class.
To do this we form a 'document' containing all texts from a class
and then perform TF-IDF across on that 'document'.

The top 3 functions below are from this
implementation of class-TD-IDF https://github.com/MaartenGr/cTFIDF
"""
from sklearn.feature_extraction.text import CountVectorizer
from skills_taxonomy.utils.dir_management import make_dir_if_not_exist
import json
import numpy as np
from skills_taxonomy import PROJECT_DIR


def c_tf_idf(documents, doc_length, ngram_range=(1, 1)):
    """Using list of documents, calculate and
    return class TF-IDF matrix and counts""

    Args:
        documents (list of strs): documents to perform class TF-IDF on
        doc_length (int): length of documents
        ngram_range (tuple): lower and upper boundary of the range of n-values
                        for different word n-grams to be extracted.
                        Defaults to (1, 1) i.e unigrams.

    Returns:
        array: matrix of tf_idf values
        CountVectorizer: CountVectorizer which has been fit to documents
    """
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(
        documents
    )
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(doc_length, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, classes, n=20):
    """Extract the top N words for each topic

    Args:
        tf_idf (array): matrix of tf_idf values
        count (CountVectorizer): CountVectorizer which has been fit to documents
        classes (list of ints): list of class ids
        n (int): number of most informative words to find, defaults to 20

    Returns:
        dict: top N words for each topic
    """
    words = count.get_feature_names()
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {
        label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1]
        for i, label in enumerate(classes)
    }
    return top_n_words


def view_top_n_words(skills, class_level):
    """View top N most informative words for skill class or subclass

    Args:
        skills (df): dataframe of skills with descriptions
        class_level (str): what class level to use ('class_id' or 'subclass_id')

    Returns:
        dict: top N words for each class or subclass
    """
    skills_classid = skills[["description", class_level]]
    skills_classid["skill_id"] = range(len(skills_classid))
    skills_per_class = skills_classid.groupby([class_level], as_index=False).agg(
        {"description": " ".join}
    )
    classes = skills_per_class[class_level]
    tf_idf_class, count_class = c_tf_idf(
        skills_per_class.description.values, doc_length=len(skills)
    )
    top_n_words = extract_top_n_words_per_topic(tf_idf_class, count_class, classes)
    return top_n_words


def save_json(save_dir, skills, file_name, class_level):
    """Save dictionary of top most informative words as json

    Args:
        save_dir (str): path to directory to save json
        skills (df): dataframe of skills with descriptions
        file_name (str): file name to save json
        class_level (str): what class level to use ('class_id' or 'subclass_id')
    """
    print(f"{save_dir}{file_name}")
    with open(f"{save_dir}{file_name}", "w") as fp:
        json.dump(
            view_top_n_words(skills, class_level),
            fp,
            sort_keys=True,
            indent=4,
        )


def save_informative_words(
    skills, save_dir=f"{PROJECT_DIR}/outputs/most_informative_words/"
):
    """Save most informative words for each class and subclass

    Args:
        skills (df): dataframe of skills with descriptions
        save_dir (str): path to directory to save json,
            defaults to f"{PROJECT_DIR}/outputs/most_informative_words/".
    """
    make_dir_if_not_exist(save_dir)
    # save most informative words for classes
    save_json(save_dir, skills, file_name="class.json", class_level="class_id")
    # save most informative words for subclasses
    save_json(
        save_dir,
        skills,
        file_name="subclass.json",
        class_level="subclass_id",
    )
