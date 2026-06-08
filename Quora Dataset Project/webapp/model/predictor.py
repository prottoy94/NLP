import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "rf_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "count_vectorizer.pkl"
SAFE_DIV = 0.0001

CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "o'clock": "of the clock",
    "shan't": "shall not",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "that'd": "that would",
    "that'd've": "that would have",
    "that'll": "that will",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there're": "there are",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'd": "what did",
    "what're": "what are",
    "what's": "what is",
    "when'd": "when did",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "who'd": "who would",
    "who'd've": "who would have",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "why'd": "why did",
    "why's": "why is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}


def preprocess(question):
    question = str(question).lower().strip()
    question = (
        question.replace("%", " percent ")
        .replace("$", " dollar ")
        .replace("\u20b9", " rupee ")
        .replace("\u20ac", " euro ")
        .replace("\u00a3", " pound ")
        .replace("\u00a5", " yen ")
        .replace("[math]", "")
        .replace("[/math]", "")
        .replace("000,000,000", "b ")
        .replace("000,000", "m")
        .replace("000", "k")
    )
    question = re.sub(r"([0-9]+)000000000", r" \1b ", question)
    question = re.sub(r"([0-9]+)000000", r" \1m ", question)
    question = re.sub(r"([0-9]+)000", r" \1k ", question)

    question = " ".join(CONTRACTIONS.get(word, word) for word in question.split())
    question = (
        question.replace("'ve", " have")
        .replace("'re", " are")
        .replace("'ll", " will")
        .replace("'d", " would")
        .replace("'s", " is")
        .replace("n't", " not")
    )

    question = BeautifulSoup(question, "html.parser").get_text()
    return re.sub(r"\W", " ", question).strip()


def common_words(q1, q2):
    q1_words = set(map(lambda word: word.lower().strip(), q1.split()))
    q2_words = set(map(lambda word: word.lower().strip(), q2.split()))
    return len(q1_words.intersection(q2_words))


def total_words(q1, q2):
    q1_words = set(map(lambda word: word.lower().strip(), q1.split()))
    q2_words = set(map(lambda word: word.lower().strip(), q2.split()))
    return len(q1_words.union(q2_words))


def fetch_token_features(q1, q2):
    token_features = [0.0] * 8
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set(q1_tokens)
    q2_words = set(q2_tokens)
    common_word_count = len(q1_words.intersection(q2_words))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_word_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[3] = common_word_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[4] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[5] = int(q1_tokens[0] == q2_tokens[0])
    token_features[6] = abs(len(q1_tokens) - len(q2_tokens))
    token_features[7] = (len(q1_tokens) + len(q2_tokens)) / 2
    return token_features


def longest_common_substring(left, right):
    if not left or not right:
        return ""

    lengths = [0] * (len(right) + 1)
    best_length = 0
    best_end = 0

    for i, left_char in enumerate(left, start=1):
        previous = 0
        for j, right_char in enumerate(right, start=1):
            current = lengths[j]
            if left_char == right_char:
                lengths[j] = previous + 1
                if lengths[j] > best_length:
                    best_length = lengths[j]
                    best_end = i
            else:
                lengths[j] = 0
            previous = current

    return left[best_end - best_length : best_end]


def fetch_length_features(q1, q2):
    length_features = [0.0] * 3
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    lcs = longest_common_substring(q1, q2)
    length_features[2] = len(lcs) / (min(len(q1), len(q2)) + 1)
    return length_features


def _ratio(left, right):
    return round(SequenceMatcher(None, left, right).ratio() * 100)


def _partial_ratio(left, right):
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if not shorter:
        return 0

    best = 0
    window = len(shorter)
    for index in range(0, len(longer) - window + 1):
        best = max(best, _ratio(shorter, longer[index : index + window]))
    return best


def _token_sort_ratio(left, right):
    return _ratio(" ".join(sorted(left.split())), " ".join(sorted(right.split())))


def _token_set_ratio(left, right):
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    intersection = " ".join(sorted(left_tokens.intersection(right_tokens)))
    left_diff = " ".join(sorted(left_tokens.difference(right_tokens)))
    right_diff = " ".join(sorted(right_tokens.difference(left_tokens)))
    return max(_ratio(intersection, f"{intersection} {left_diff}".strip()), _ratio(intersection, f"{intersection} {right_diff}".strip()))


def fetch_fuzzy_features(q1, q2):
    if fuzz is not None:
        return [
            fuzz.QRatio(q1, q2),
            fuzz.partial_ratio(q1, q2),
            fuzz.token_sort_ratio(q1, q2),
            fuzz.token_set_ratio(q1, q2),
        ]

    return [_ratio(q1, q2), _partial_ratio(q1, q2), _token_sort_ratio(q1, q2), _token_set_ratio(q1, q2)]


class DuplicateQuestionPredictor:
    def __init__(self, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.model = self._load_pickle(self.model_path)
        self.vectorizer = self._load_pickle(self.vectorizer_path)

    @property
    def is_ready(self):
        return self.model is not None and self.vectorizer is not None

    def _load_pickle(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

        with path.open("rb") as file:
            return pickle.load(file)

    def create_query_point(self, question1, question2):
        q1 = preprocess(question1)
        q2 = preprocess(question2)

        if not q1 or not q2:
            raise ValueError("Questions must contain at least one word after preprocessing.")

        total = total_words(q1, q2)
        word_share = round(common_words(q1, q2) / total, 2) if total else 0.0

        input_query = [
            len(q1),
            len(q2),
            len(q1.split(" ")),
            len(q2.split(" ")),
            common_words(q1, q2),
            total,
            word_share,
        ]
        input_query.extend(fetch_token_features(q1, q2))
        input_query.extend(fetch_length_features(q1, q2))
        input_query.extend(fetch_fuzzy_features(q1, q2))

        q1_bow = self.vectorizer.transform([q1]).toarray()
        q2_bow = self.vectorizer.transform([q2]).toarray()
        query_point = np.hstack((np.array(input_query).reshape(1, 22), q1_bow, q2_bow))

        expected_features = getattr(self.model, "n_features_in_", query_point.shape[1])
        if query_point.shape[1] != expected_features:
            raise ValueError(f"Feature shape mismatch: created {query_point.shape[1]}, expected {expected_features}.")

        return query_point, q1, q2

    def predict(self, question1, question2):
        query_point, clean_q1, clean_q2 = self.create_query_point(question1, question2)
        prediction = int(self.model.predict(query_point)[0])
        duplicate_probability = None
        confidence = None

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(query_point)[0]
            classes = list(getattr(self.model, "classes_", []))
            if 1 in classes:
                duplicate_probability = float(probabilities[classes.index(1)])
            confidence = float(np.max(probabilities))

        return {
            "prediction": prediction,
            "label": "Duplicate" if prediction == 1 else "Not duplicate",
            "duplicate_probability": duplicate_probability,
            "confidence": confidence,
            "cleaned_questions": {"question1": clean_q1, "question2": clean_q2},
        }
