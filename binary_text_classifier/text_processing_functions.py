# Standard libraries
import re
import string
import pandas as pd

# Scikit-Learn
from sklearn.feature_extraction.text import CountVectorizer

# NLP tools
import urlextract
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

nltk.download("stopwords")
from nltk.corpus import stopwords
from spellchecker import SpellChecker

# Assign the list of NLTK stopwords to a constant
STOPWORDS = stopwords.words("english")


def get_top_n_words(
    data: pd.Series | list[str],
    top_n: int | None = None,
    exclude_list: list[str] | None = None,
) -> list[tuple]:
    """Get the top_n words and their frequencies from a corpus of text.
    Scikit-Learn's CountVectorizer is used to create the matrix of vocabulary and counts.
    A list of word and count tuples is returned.
    Words provided in the optional exclude_list are excluded.

    Parameters
    ----------
    data : pd.Series | list[str]
        Text data
    top_n : int | None, optional
        Get the n most frequently occuring words in the given data, by default None
    exclude_list : list[str] | None
        Optional list of words to exclude in data returned, by default True

    Returns
    -------
    list[tuple]
        List of word and count tuples for the top_n words.
    """
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(data)
    sum_words = matrix.sum(axis=0)
    if exclude_list is None:
        words_freq = [
            (word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()
        ]
    else:
        words_freq = [
            (word, sum_words[0, idx])
            for word, idx in vectorizer.vocabulary_.items()
            if word not in exclude_list
        ]

    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    if top_n is not None:
        return words_freq[:top_n]
    else:
        return words_freq


def remove_urls_by_tld(
    data: pd.Series | list[str], tld_list: list[str], replace_str: str | None
) -> list[str]:
    """Remove urls in the data if the url contains a top-level domain matching
    one in tld_list. URLs are removed or optionally replaced with a provided string.

    Parameters
    ----------
    data : pd.Series | list[str]
        Text data
    tld_list : list[str]
        List of top-level domains to consider
    replace_str : str | None
        Optional string replacement for URL

    Returns
    -------
    list[str]
        Text data with URLs removed or replaced.
    """
    url_extractor = urlextract.URLExtract()
    processed_text = []
    for text in data:
        text = text.lower()
        urls = list(set(url_extractor.find_urls(text)))
        urls = [s for s in urls if any(tld in s for tld in tld_list)]
        if replace_str is None:
            for url in urls:
                text = text.replace(url, "")
        else:
            for url in urls:
                text = text.replace(url, replace_str)
        processed_text.append(text)
    return processed_text


def replace_word_containing(
    data: pd.Series | list[str],
    str_contains_list: list[str],
    replacement: str,
) -> list[str]:
    """Replace words which contain one or more of the strings provided in str_contains_list.
    Words are replaced with a provided string.

    Parameters
    ----------
    data : pd.Series | list[str]
        Text data
    str_contains_list : list[str]
        List of strings to check for in each word
    replacement : str
        String replacement

    Returns
    -------
    list[str]
        Text data with words replaced.
    """
    processed_text = []
    for text in data:
        text = text.lower()
        words = text.split()
        words_to_replace = [
            word for word in words if any(s in word for s in str_contains_list)
        ]
        for word in words_to_replace:
            text = text.replace(word, replacement)
        processed_text.append(text)
    return processed_text


def remove_numbers_from_text(
    data: pd.Series | list[str],
    min_digits: int,
    replace_str: str | None,
    max_word_length: int,
) -> list[str]:
    """Remove or replace substrings which contain a minimum number of consecutive digits (and do not exceed a maximum length).

    Parameters
    ----------
    data : pd.Series | list[str]
        Text data
    min_digits : int
        Minimum number of consecutive digits in substring for replacement to be applied
    replace_str : str | None
        Optional string replacement
    max_word_length : int
        Maximum length of substring for replacement to be applied

    Returns
    -------
    list[str]
        Text data with string replacement applied.
    """
    processed_text = []
    for text in data:
        text = text.lower()
        words = text.split()
        numbers_to_replace = [
            word
            for word in words
            if re.search(rf"\d{{{min_digits},}}", word) and len(word) <= max_word_length
        ]
        if replace_str is None:
            for num in numbers_to_replace:
                text = text.replace(num, "")
        else:
            for num in numbers_to_replace:
                text = text.replace(num, replace_str)
        processed_text.append(text)
    return processed_text


def remove_currency_from_text(
    data: pd.Series | list[str], replace_str: str | None
) -> list[str]:
    """Remove or replace substrings which contain a pound, dollar or euro currency symbol.
    Any spaces between a currency symbol and following digits are first removed.

    Parameters
    ----------
    data : pd.Series | list[str]
        Text data
    replace_str : str | None
        Replacement string

    Returns
    -------
    list[str]
        Text data with currency symbols and values replaced by given string.
    """
    processed_text = []
    currency_pattern = r"([£€$])\s*(\d)"  # match for currency symbol followed by optional whitespace and digit
    for text in data:
        text = text.lower()
        # First, remove any spaces between currency symbol and following digit
        text = re.sub(currency_pattern, r"\1\2", text)
        words = text.split()
        currency_to_replace = [
            word for word in words if re.search(currency_pattern, word)
        ]
        if replace_str is None:
            for cur in currency_to_replace:
                text = text.replace(cur, "")
        else:
            for cur in currency_to_replace:
                text = text.replace(cur, replace_str)
        processed_text.append(text)
    return processed_text


def find_incorrect_spellings(
    data: pd.Series | list[str], distance: int = 2
) -> list[dict]:
    """Checks text against words in the SpellChecker corpus and returns a
    list containing a dictionary for each element of the input data,
    where the keys are the potentially mis-spelled words and the values are
    the suggested correction.

    Parameters
    ----------
    data : pd.Series | list[str]
        Text data
    distance : int, optional
        The maximum edit distance (how many alterations
        permitted for a correction from the original word), by default 2

    Returns
    -------
    list[dict]
        List of dictionaries of mis-spelled words and their corrections
    """
    # Initialize the SpellChecker
    spell_checker = SpellChecker(distance=distance)
    # Create ordinal number regex (such tokens will be ignored)
    ordinal_number_pattern = re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE)

    processed_text = []
    # Cache suggestions to avoid re-computing corrections
    suggestions_dict = {}
    # Process each string in the series
    for text in data:
        # Tokenize the string
        tokens = word_tokenize(text)
        # Reduce to list of unknown words
        tokens = spell_checker.unknown(tokens)
        # Store the unusual spellings and potential corrections in a dict
        unusual_spellings = {}
        for token in tokens:
            if (
                token.startswith("@")
                or token.startswith("#")
                or not token.isascii()
                or ordinal_number_pattern.match(token)
            ):
                continue
            elif token in suggestions_dict:
                unusual_spellings[token] = suggestions_dict[token]
            else:
                spell_checked = spell_checker.correction(token)
                if token != spell_checked:
                    unusual_spellings[token] = spell_checked
                    suggestions_dict[token] = spell_checked

        processed_text.append(unusual_spellings)

    return processed_text


def expand_contractions(
    data: pd.Series | list[str], contractions_dict: dict[str], do_expansion: bool = True
) -> list[str]:
    """Replace contractions found in the given dictionary key by the mapped value string.

    Parameters
    ----------
    data : pd.Series | list[str]
        Text data
    contractions_dict : dict[str]
        Mapping from contraction to expansion
    do_expansion : bool, optional
        Whether to do the expansion or remove the string, by default True

    Returns
    -------
    list[str]
        Text data with contractions expanded or removed.
    """
    processed_text = []
    for text in data:
        text = text.lower()
        words = text.split()
        words_to_replace = [word for word in words if word in contractions_dict]
        if not do_expansion:
            for word in words_to_replace:
                text = text.replace(word, "")
        else:
            for word in words_to_replace:
                text = text.replace(word, contractions_dict[word])
        processed_text.append(text)
    return processed_text


def remove_punctuation(text: str, replace_with_space: bool = True) -> str:
    """Remove punctuation from given string.

    Parameters
    ----------
    text : str
        Text data
    replace_with_space : bool, optional
        Whether to replace the punctuation with a space or not, by default True

    Returns
    -------
    str
        Text data with punctuation removed.
    """
    if replace_with_space:
        return text.translate(
            str.maketrans(string.punctuation, " " * len(string.punctuation))
        )
    else:
        return text.translate(str.maketrans("", "", string.punctuation))


def remove_non_alphabetic(text: str) -> str:
    """Remove non alphabetic characters from string, replacing with a space.

    Parameters
    ----------
    text : str
        Text data

    Returns
    -------
    str
        Text data with non-alphabetic characters removed.
    """
    return re.sub("[^a-zA-Z]", " ", text)


def snowball_stem(text: str, exclude_words: list[str]) -> str:
    """Apply stemming to input text using NLTK's SnowballStemmer.

    Parameters
    ----------
    text : str
        Text data
    exclude_words : list[str]
        List of words to exclude from the stemming

    Returns
    -------
    str
        Text data with stemming applied.
    """
    text = text.lower().split()
    stemmer = SnowballStemmer("english")
    text = [stemmer.stem(word) for word in text if not word in set(exclude_words)]
    return " ".join(text)


def stem_and_tidy(
    data: pd.Series | list[str],
    exclude_words: list[str] | None,
    no_punct: bool = True,
    only_alpha: bool = True,
    stemming: bool = True,
) -> list[str]:
    """Helper function to combine calls to remove_punctuation, remove_non_alphabetic and snowball_stem.

    Parameters
    ----------
    data : pd.Series | list[str]
        Text data
    exclude_words : list[str] | None
        List of words to exclude from the stemming
    no_punct : bool, optional
        Whether to remove punctuation, by default True
    only_alpha : bool, optional
        Whether to remove non-alphabetic characters, by default True
    stemming : bool, optional
        whether to apply stemming, by default True

    Returns
    -------
    list[str]
        Text data after requested processing.
    """
    processed_text = []
    for text in data:
        if no_punct:
            text = remove_punctuation(text)
        if only_alpha:
            text = remove_non_alphabetic(text)
        if stemming:
            text = snowball_stem(text, exclude_words=exclude_words)
        processed_text.append(text)
    return processed_text
