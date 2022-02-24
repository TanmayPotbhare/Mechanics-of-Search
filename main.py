import math
import os
import re
from collections import Counter, defaultdict, OrderedDict
from datetime import datetime

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

COLLECTION_DIR = 'collection'
TOPICS_DIR = 'topics'


class Indexer:
    def __init__(self, path):
        self.path = path
        self.corpus = defaultdict(str)
        self.tf = defaultdict(dict)
        self.df = defaultdict(int)
        self.idf = defaultdict(int)
        self.doc_len = defaultdict(int)
        self.avg_doc_len = 0

    def index_files(self):
        """

        :return:
        """
        # print('indexing files')
        for file in os.listdir(self.path):
            abs_path = os.path.join(self.path, file)
            with open(abs_path) as f:
                file_data = f.read()
                docid, txt = self._get_document_text(file_data)
                self.corpus[docid] = txt
                self.doc_len[docid] = len(txt)
                self.tf[docid] = self._calc_tf(txt)

        for term, freq in self.df.items():
            self.idf[term] = math.log(1 + (len(self.corpus) - freq + 0.5) / (freq + 0.5))

        self.avg_doc_len = sum(self.doc_len.values()) / len(os.listdir(self.path))

    def _calc_tf(self, txt):
        """

        :param txt:
        :return:
        """
        tf = {}
        total_no_of_words = len(txt)
        for word, count in Counter(txt).items():
            if total_no_of_words:
                tf[word] = count / total_no_of_words

            self.df[word] = self.df.get(word, 0) + count

        return tf

    @staticmethod
    def _get_document_text(file_data):
        """

        :param file_data:
        :return:
        """
        soup = BeautifulSoup(file_data.lower(), 'lxml')
        headline = re.sub(r'\W ', '', soup.find('headline').text.strip()).split()
        docid = re.sub(r'\W ', '', soup.find('docid').text.strip())
        txt = re.sub(r'\W ', '', soup.find('text').text.strip()).split()
        return docid, [word for word in txt + headline if word not in stop_words]


class PageRanking:
    """

    """
    def __init__(self, indexer):
        self.indexer = indexer

    def search(self, query, limit_results=100):
        """

        :param query:
        :param limit_results:
        :return:
        """
        query = [word for word in query.lower().split() if word not in stop_words]
        scores = self.get_scores(query)
        return self.sort(scores, limit_results=limit_results)

    def get_scores(self, query):
        """

        :param query:
        :return:
        """
        return {}

    def sort(self, scores, limit_results=20):
        """

        :param scores:
        :param limit_results:
        :return:
        """
        sorted_value = OrderedDict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        return {k: sorted_value[k] for k in list(sorted_value)[:limit_results]}


class VectorSpaceModel(PageRanking):
    """

    """
    def get_scores(self, query):
        """

        :param query:
        :return:
        """
        query_wc = {word: query.count(word) for word in query}
        relevance_scores = {}
        for doc_id, tfs in self.indexer.tf.items():
            score = 0
            for word in query:
                if word in tfs:
                    score += query_wc[word] * tfs.get(word) * self.indexer.idf[word]
            relevance_scores[doc_id] = score
        return relevance_scores


class BM25(PageRanking):
    """

    """
    def __init__(self, indexer, k1=1.5, b=0.75):
        self.indexer = indexer
        self.k1 = k1
        self.b = b

    def get_scores(self, query):
        """

        :param query:
        :return:
        """
        scores_dict = {}
        for docid, tfs in self.indexer.tf.items():
            score = 0
            for word in query:
                if word in tfs:
                    numerator = self.indexer.idf[word] * tfs.get(word) * (self.k1 + 1)
                    denominator = tfs.get(word) + self.k1 * (
                            1 - self.b + self.b * self.indexer.doc_len[docid] / self.indexer.avg_doc_len)
                    score += (numerator / denominator)
            scores_dict[docid] = score
        return scores_dict


def dump_results_to_file(model_name, query, results):
    """

    :param model_name:
    :param query:
    :param results:
    :return:
    """
    file_name = f'{model_name}.txt'
    file_path = os.path.join(os.getcwd(), 'output', file_name)
    file_mode = 'a' if os.path.exists(file_path) else 'w'
    with open(file_path, file_mode) as file:
        for index, (docid, score) in enumerate(results.items()):
            file.write(f'{query["query_id"]} iter {docid} {index} {score} run_id\n')


def fetch_topics(topics_dir):
    """

    :param topics_dir:
    :return:
    """
    queries = {}
    for file in os.listdir(topics_dir):
        abs_path = os.path.join(topics_dir, file)
        with open(abs_path) as f:
            soup = BeautifulSoup(f.read().lower(), 'lxml')
            query_id = re.sub(r'\W ', '', soup.find('queryid').text.strip())
            title = re.sub(r'\W ', '', soup.find('title').text.strip())
            desc = re.sub(r'\W ', '', soup.find('desc').text.strip())
            narr = re.sub(r'\W ', '', soup.find('narr').text.strip())
            queries[query_id] = {'query_id': query_id, 'title': title, 'desc': desc, 'narr': narr}
    return queries


def run_models(indexer, query):
    """

    :param indexer:
    :param query:
    :return:
    """
    vector_spc_model = VectorSpaceModel(indexer=indexer)
    vsm_results = vector_spc_model.search(query['title'])
    dump_results_to_file('VectorSpaceModel', query, vsm_results)
    # print("Vector Space Model", vsm_results)
    bm_25_search = BM25(indexer=indexer)
    bm25_results = bm_25_search.search(query['title'])
    dump_results_to_file('BM25', query, bm25_results)
    # print("BM search", bm25_results)


def cleanup_output_folder():
    """

    :return:
    """
    print('Cleaning up output folder')
    path = os.path.join(os.getcwd(), 'output')
    for file in os.listdir(path):
        if os.path.exists(os.path.join(path, file)):
            os.remove(os.path.join(path, file))


def main():
    """

    :return:
    """
    cleanup_output_folder()

    d1 = datetime.now()
    print(d1)

    cwd = os.getcwd()
    topics_dir = os.path.join(cwd, TOPICS_DIR)
    collections_dir = os.path.join(cwd, COLLECTION_DIR)

    queries = fetch_topics(topics_dir)

    print('Indexing Files')
    indexer = Indexer(path=collections_dir)
    indexer.index_files()
    print('Indexing Complete')

    print('Running Queries')
    for query_id, query in queries.items():
        print(f'Running query {query_id} - {query["title"]}')
        run_models(indexer, query)
    print('Queries Complete')
    print(f'Output files are present in {os.path.join(os.getcwd(), "output")}')
    d2 = datetime.now()
    print(d2)
    print('time delta', d2 - d1)


if __name__ == '__main__':
    main()
