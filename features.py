import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.probability import FreqDist
from nltk.util import ngrams
import gensim.downloader as api
from redditparser import Reddit_Parser
from sklearn.feature_extraction.text import TfidfVectorizer
import math

# Around 4 minutes

class StylometryExtractor():
    def __init__(self, corpus):
        self.corpus = corpus
        self.embedding_vocab = api.load("glove-wiki-gigaword-300")
        self.top_100_common_words = None
        self.top_10_common_adjectives = None
        self.top_10_common_conjunctions = None
        self.top_10_common_interrogatives = None
        self.top_10_common_nouns = None
        self.top_10_common_verbs = None
        self.top_30_common_three_grams = None
        self.top_30_common_five_grams = None
        self.top_100_common_oov_embeddings = None
        self.top_100_common_tri_grams = None
        self.top_100_common_five_grams = None
    
    def is_consonant(self, char):
        consonant_pattern = re.compile(r'[bcdfghjklmnpqrstvwxyz]', re.IGNORECASE)
        return bool(consonant_pattern.match(char))

    def is_vowel(self, char):
        vowel_pattern = re.compile(r'[aeiou]', re.IGNORECASE)
        return bool(vowel_pattern.match(char))
    
    def preprocess_corpus(self):
        
        corpus = self.corpus
        print("Saved corpus variable")

        # Preprocessing step

        corpus_lower = corpus.lower()
        print("lowered")
        tokens = word_tokenize(corpus_lower)
        print("tokenized")
        tagged_tokens = pos_tag(tokens)
        print("pos_tagged")
        freq_dist = FreqDist(tagged_tokens)
        freq_dist_no_tag = FreqDist(tokens)
        print("general freq dist done")
        vocabulary = freq_dist_no_tag.keys()
        print("vocabulary initialized")
        adjectives = [word for word, tag in tagged_tokens if tag.startswith('JJ')]
        print("adjectives done")
        conjunctions = [word for word, tag in tagged_tokens if tag.startswith('CC')]
        print("conjunctions done")
        interrogatives = [word for word, tag in tagged_tokens if tag in ['WP', 'WRB']]
        print("interrogatives done")
        nouns = [word for word, tag in tagged_tokens if tag in ['NN', 'NNS', 'NNPS', 'NNP']]
        print("nouns done")
        verbs = [word for word, tag in tagged_tokens if tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
        print("verbs done")
        three_grams = list(ngrams(tokens, 3))
        print("3grams done")
        five_grams = list(ngrams(tokens, 5))
        print("5grams done")
        oov_embeddings = [word for word in vocabulary if word not in self.embedding_vocab]
        print("oov done")

        ''''''
        freq_dist_adjectives = FreqDist(adjectives)
        print("collected freq_dists for adjectives")
        freq_dist_conjunctions = FreqDist(conjunctions)
        print("collected freq_dists for conjunctions")
        freq_dist_interrogatives = FreqDist(interrogatives)
        print("collected freq_dists for interrogatives")
        freq_dist_nouns = FreqDist(nouns)
        print("collected freq_dists for nouns")
        freq_dist_verbs = FreqDist(verbs)
        print("collected freq_dists for verbs")
        freq_dist_oov_embeddings = FreqDist(oov_embeddings)
        print("collected freq_dists for oovs")
        freq_dist_three_grams = FreqDist(three_grams)
        print("collected freq_dists for 3grams")
        freq_dist_five_grams = FreqDist(five_grams)
        print("collected freq_dists for 5grams")

        ''''''
        print("starting collecting most commons")
        self.top_100_common_words = [x[0] for x in freq_dist.most_common(100)]
        self.top_10_common_adjectives = [x[0] for x in freq_dist_adjectives.most_common(10)]
        self.top_10_common_conjunctions = [x[0] for x in freq_dist_conjunctions.most_common(10)]
        self.top_10_common_interrogatives = [x[0] for x in freq_dist_interrogatives.most_common(10)]
        self.top_10_common_nouns = [x[0] for x in freq_dist_nouns.most_common(10)]
        self.top_10_common_verbs = [x[0] for x in freq_dist_verbs.most_common(10)]
        self.top_30_common_three_grams = [x[0] for x in freq_dist_three_grams.most_common(30)]
        self.top_30_common_five_grams = [x[0] for x in freq_dist_five_grams.most_common(30)]
        self.top_100_common_oov_embeddings = [x[0] for x in freq_dist_oov_embeddings.most_common(100)]
        self.top_100_common_tri_grams = [x[0] for x in freq_dist_five_grams.most_common(100)]
        self.top_100_common_five_grams = [x[0] for x in freq_dist_three_grams.most_common(100)]

        # Update init to save info about the corpus

    def get_values(self, value):

        if value == 'words':
            return self.top_100_common_words
        elif value == 'adjectives':
            return self.top_10_common_adjectives
        elif value == 'conjunctions':
            return self.top_10_common_conjunctions
        elif value == 'interrogatives':
            return self.top_10_common_interrogatives
        elif value == 'nouns':
            return self.top_10_common_nouns
        elif value == 'verbs':
            return self.top_10_common_verbs
        elif value == 'three_grams':
            return self.top_30_common_three_grams
        elif value == 'five_grams':
            return self.top_30_common_five_grams
        elif value == 'oov':
            return self.top_100_common_oov_embeddings
        else:
            ValueError('Set value')

    def stylometry_extractor(self, paragraph, character_level=True, word_level=True, sentence_level=True):

        # Some init
        F = np.array([])
        F = F.reshape(-1)
        F_chars = np.zeros(104)
        F_words = np.zeros(60)
        F_sents = np.zeros(220)

        epsilon = 1e-10

        # Preprocess paragraph
        sentences = sent_tokenize(paragraph)
        tokens = word_tokenize(paragraph)
        paragraph_lower = paragraph.lower()
        tokens_lower = word_tokenize(paragraph_lower)
        pos_tagged_tokens = pos_tag(tokens)
        parts_of_speech = [x[1] for x in pos_tagged_tokens]

        # Set regexs

        punctuation = r'[\W_]+'
        other_things = r'[^a-zA-Z0-9\s\\p{P}]'
        words = len(tokens)

        # Character level

        chars_list = list(paragraph_lower)
        chars = len(list(paragraph_lower))
        alphas = len([char for char in chars_list if char.isalpha()])
        uppers = len([char for char in list(paragraph) if char.isupper()])
        digits = len([char for char in chars_list if char.isdigit()])
        whitespaces = len([char for char in chars_list if char == ' '])
        vowels = len([char for char in paragraph if char.lower() in 'aeiou'])
        char_two_grams = len([''.join(gram) for gram in list(ngrams(chars_list, 2))])
        consonant_vowel_twograms = len([''.join(gram) for gram in ngrams(chars_list, 2) if re.match(r'[bcdfghjklmnpqrstvwxyz][aeiou]', ''.join(gram))])
        vowel_consonant_twograms = len([''.join(gram) for gram in ngrams(chars_list, 2) if re.match(r'[aeiou][bcdfghjklmnpqrstvwxyz]', ''.join(gram))])
        consonant_consonant_twograms = len([''.join(gram) for gram in ngrams(chars_list, 2) if re.match(r'[bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrstvwxyz]', ''.join(gram))])
        vowel_vowel_twograms = len([''.join(gram) for gram in ngrams(chars_list, 2) if re.match(r'[aeiou][aeiou]', ''.join(gram))])

        # Word level

        word_three_grams = list(ngrams(tokens, 3))
        word_five_grams = list(ngrams(tokens, 5))

        # Sentence level 

        puncts = len(re.findall(punctuation, paragraph))
        others = len(re.findall(other_things, paragraph)) 
        pp = len([pos_tag for pos_tag in parts_of_speech if pos_tag.startswith('PRP')])
        adj = len([pos_tag for pos_tag in parts_of_speech if pos_tag.startswith('JJ')])
        conj = len([pos_tag for pos_tag in parts_of_speech if pos_tag.startswith('CC')])
        aux = len([pos_tag for pos_tag in parts_of_speech if pos_tag.startswith('MD')])
        interr = len([pos_tag for pos_tag in parts_of_speech if pos_tag in ['WP', 'WRB']])
        nouns = len([pos_tag for pos_tag in parts_of_speech if pos_tag in ['NN', 'NNS', 'NNPS', 'NNP']])
        verbs = len([pos_tag for pos_tag in parts_of_speech if pos_tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']])
        dets = len([pos_tag for pos_tag in parts_of_speech if pos_tag.startswith('DT')])
        articles = len([token for token in tokens_lower if token in ["a", "an", "the"]])
        sents = len(nltk.sent_tokenize(paragraph)) 
        top100_common_words = [word for word, _ in self.top_100_common_words]

        # Build feature vector

        freq_cons_0 = ["t", "n", "s", "r", "h"]
        count_freq_cons_0 = sum(paragraph_lower.count(char) for char in freq_cons_0)
        freq_cons_1 = ["l", "d", "c", "p", "f"]
        count_freq_cons_1 = sum(paragraph_lower.count(char) for char in freq_cons_1)
        freq_cons_2 = ["m", "w", "y", "b", "g"]
        count_freq_cons_2 = sum(paragraph_lower.count(char) for char in freq_cons_2)
        freq_cons_3 = ["j", "k", "q", "v", "x", "z"]
        count_freq_cons_3 = sum(paragraph_lower.count(char) for char in freq_cons_3)
        to_be_verbs = [r'\b(am)\b', r'\b(are)\b', r'\b(be)\b', r'\b(been)\b', r'\b(being)\b', r'\b(is)\b', r'\b(was)\b', r'\b(were)\b']
        count_to_be_verbs = sum(paragraph_lower.count(word) for word in to_be_verbs)

        def count_missed_uppercase(tokens):
            missed_uppercase_count = 0

            for i in range(len(tokens) - 1):
                if tokens[i] == '.' and tokens[i+1][0].islower():
                    missed_uppercase_count += 1

            return missed_uppercase_count
        
        def count_missing_periods(sentences):
            missing_period_count = 0

            for sentence in sentences:
                if sentence[-1] != '.':
                    missing_period_count += 1

            return missing_period_count

        if character_level:
            F_chars = [
                alphas/(chars + epsilon),
                uppers/(chars + epsilon),
                digits/(chars + epsilon),
                whitespaces/(chars + epsilon),
                vowels/(chars + epsilon),
                paragraph_lower.count("a")/(vowels + epsilon),
                paragraph_lower.count("e")/(vowels + epsilon),
                paragraph_lower.count("i")/(vowels + epsilon),
                paragraph_lower.count("o")/(vowels + epsilon),
                paragraph_lower.count("u")/(vowels + epsilon),
                paragraph_lower.count("a")/(chars + epsilon),
                count_freq_cons_0/(alphas + epsilon),
                count_freq_cons_1/(alphas + epsilon),
                count_freq_cons_2/(alphas + epsilon),
                count_freq_cons_3/(alphas + epsilon),
                paragraph_lower.count("t")/(count_freq_cons_0 + epsilon),
                paragraph_lower.count("n")/(count_freq_cons_0 + epsilon),
                paragraph_lower.count("s")/(count_freq_cons_0 + epsilon),
                paragraph_lower.count("r")/(count_freq_cons_0 + epsilon),
                paragraph_lower.count("h")/(count_freq_cons_0 + epsilon),
                paragraph_lower.count("l")/(count_freq_cons_1 + epsilon),
                paragraph_lower.count("d")/(count_freq_cons_1 + epsilon),
                paragraph_lower.count("c")/(count_freq_cons_1 + epsilon),
                paragraph_lower.count("p")/(count_freq_cons_1 + epsilon),
                paragraph_lower.count("f")/(count_freq_cons_1 + epsilon),
                paragraph_lower.count("m")/(count_freq_cons_2 + epsilon),
                paragraph_lower.count("w")/(count_freq_cons_2 + epsilon),
                paragraph_lower.count("y")/(count_freq_cons_2 + epsilon),
                paragraph_lower.count("b")/(count_freq_cons_2 + epsilon),
                paragraph_lower.count("g")/(count_freq_cons_2 + epsilon),
                paragraph_lower.count("j")/(count_freq_cons_3 + epsilon),
                paragraph_lower.count("k")/(count_freq_cons_3 + epsilon),
                paragraph_lower.count("q")/(count_freq_cons_3 + epsilon),
                paragraph_lower.count("v")/(count_freq_cons_3 + epsilon),
                paragraph_lower.count("x")/(count_freq_cons_3 + epsilon),
                paragraph_lower.count("z")/(count_freq_cons_3 + epsilon),
                consonant_consonant_twograms/(char_two_grams + epsilon),
                vowel_consonant_twograms/(char_two_grams + epsilon),
                consonant_vowel_twograms/(char_two_grams + epsilon),
                vowel_vowel_twograms/(char_two_grams + epsilon),
                paragraph_lower.count("st")/(consonant_consonant_twograms + epsilon),
                paragraph_lower.count("nd")/(consonant_consonant_twograms + epsilon),
                paragraph_lower.count("th")/(consonant_consonant_twograms + epsilon),
                paragraph_lower.count("an")/(vowel_consonant_twograms + epsilon),
                paragraph_lower.count("in")/(vowel_consonant_twograms + epsilon),
                paragraph_lower.count("er")/(vowel_consonant_twograms + epsilon),
                paragraph_lower.count("es")/(vowel_consonant_twograms + epsilon),
                paragraph_lower.count("on")/(vowel_consonant_twograms + epsilon),
                paragraph_lower.count("at")/(vowel_consonant_twograms + epsilon),
                paragraph_lower.count("en")/(vowel_consonant_twograms + epsilon),
                paragraph_lower.count("or")/(vowel_consonant_twograms + epsilon),
                paragraph_lower.count("he")/(consonant_vowel_twograms + epsilon),
                paragraph_lower.count("re")/(consonant_vowel_twograms + epsilon),
                paragraph_lower.count("ti")/(consonant_vowel_twograms + epsilon),
                paragraph_lower.count("ea")/(vowel_vowel_twograms + epsilon),
                sum(1 for i in range(len(paragraph_lower) - 1) if paragraph_lower[i].isalpha() and paragraph_lower[i] == paragraph_lower[i + 1])/(char_two_grams + epsilon),
                paragraph_lower.count("a"),
                paragraph_lower.count("b"),
                paragraph_lower.count("c"),
                paragraph_lower.count("d"),
                paragraph_lower.count("e"),
                paragraph_lower.count("f"),
                paragraph_lower.count("g"),
                paragraph_lower.count("h"),
                paragraph_lower.count("i"),
                paragraph_lower.count("j"),
                paragraph_lower.count("k"),
                paragraph_lower.count("l"),
                paragraph_lower.count("m"),
                paragraph_lower.count("n"),
                paragraph_lower.count("o"),
                paragraph_lower.count("p"),
                paragraph_lower.count("q"),
                paragraph_lower.count("r"),
                paragraph_lower.count("s"),
                paragraph_lower.count("t"),
                paragraph_lower.count("u"),
                paragraph_lower.count("v"),
                paragraph_lower.count("w"),
                paragraph_lower.count("x"),
                paragraph_lower.count("y"),
                paragraph_lower.count("z"),
                paragraph_lower.count("~"),
                paragraph_lower.count("@"),
                paragraph_lower.count("#"),
                paragraph_lower.count("$"),
                paragraph_lower.count("%"),
                paragraph_lower.count("^"),
                paragraph_lower.count("&"),
                paragraph_lower.count("*"),
                paragraph_lower.count("-"),
                paragraph_lower.count("="),
                paragraph_lower.count("+"),
                paragraph_lower.count(">"),
                paragraph_lower.count("<"),
                paragraph_lower.count("["),
                paragraph_lower.count("]"),
                paragraph_lower.count("{"),
                paragraph_lower.count("}"),
                paragraph_lower.count("/"),
                paragraph_lower.count("\\"),
                paragraph_lower.count("|"),
            ]
            F = np.concatenate((F, F_chars))

        if word_level:
            F_words = [
                len([token for token in tokens if len(token) == 1])/(words + epsilon),
                len([token for token in tokens if len(token) == 2])/(words + epsilon),
                len([token for token in tokens if len(token) == 3])/(words + epsilon),
                len([token for token in tokens if len(token) == 4])/(words + epsilon),
                len([token for token in tokens if len(token) == 5])/(words + epsilon),
                len([token for token in tokens if len(token) == 6])/(words + epsilon),
                len([token for token in tokens if len(token) == 7])/(words + epsilon),
                len([token for token in tokens if len(token) >= 8])/(words + epsilon),
                len([token for token in tokens if len(token) <= 3])/(words + epsilon),
                chars/(words + epsilon),
                len(set(tokens))/(words + epsilon),
                words, 
                len([token for token in tokens if len(token) <= 3]),
                len([token for token in tokens if len(token) == 1]),
                len([token for token in tokens if len(token) == 2]),
                len([token for token in tokens if len(token) == 3]),
                len([token for token in tokens if len(token) == 4]),
                len([token for token in tokens if len(token) == 5]),
                len([token for token in tokens if len(token) == 6]),
                len([token for token in tokens if len(token) == 7]),
                len([token for token in tokens if len(token) == 8]),
                len([token for token in tokens if len(token) == 9]),
                len([token for token in tokens if len(token) == 10]),
                len([token for token in tokens if len(token) == 11]),
                len([token for token in tokens if len(token) == 12]),
                len([token for token in tokens if len(token) >= 13]),
                paragraph_lower.count(":)"),
                paragraph_lower.count(":("),
                paragraph_lower.count(r'\b(lol)\b'),
                paragraph_lower.count(";)"),
                paragraph_lower.count("..."),
                paragraph_lower.count("cmv"),
                paragraph_lower.count("eli5"),
                paragraph_lower.count("iirc"),
                paragraph_lower.count("imo"),
                paragraph_lower.count("imho"),
                paragraph_lower.count("irl"),
                paragraph_lower.count("mrw"),
                paragraph_lower.count("mfw"),
                paragraph_lower.count("nsfl"),
                paragraph_lower.count("nsfw"),
                paragraph_lower.count(r'\b(op)\b'),
                paragraph_lower.count(r'\b(oc)\b'),
                paragraph_lower.count("psa"),
                paragraph_lower.count("tldr"),
                paragraph_lower.count("tl;dr"),
                paragraph_lower.count(r'\b(til)\b'),
                paragraph_lower.count("wip"),
                paragraph_lower.count("ysk"),
                paragraph_lower.count("aka"),
                paragraph_lower.count("goat"),
                paragraph_lower.count(r'\b(ffs)\b'),
                paragraph_lower.count("fyi"),
                paragraph_lower.count("tbh"),
                paragraph_lower.count("ikr"),
                count_missed_uppercase(tokens),
                count_missing_periods(sentences),
                sum(1 for token in tokens if token in self.top_100_common_oov_embeddings)/(words + epsilon),
                sum(1 for trigram in word_three_grams if trigram in self.top_100_common_tri_grams)/(len(word_three_grams) + epsilon),
                sum(1 for fivegram in word_five_grams if fivegram in self.top_100_common_five_grams)/(len(word_five_grams) + epsilon),
            ]
            F = np.concatenate((F, F_words))

        if sentence_level:
            F_sents = [
                sents,
                puncts/chars,
                paragraph_lower.count('.')/(puncts + epsilon),
                paragraph_lower.count(',')/(puncts + epsilon),
                paragraph_lower.count('?')/(puncts + epsilon),
                paragraph_lower.count('!')/(puncts + epsilon),
                paragraph_lower.count(';')/(puncts + epsilon),
                paragraph_lower.count(':')/(puncts + epsilon),
                paragraph_lower.count('\'')/(puncts + epsilon),
                paragraph_lower.count('\"')/(puncts + epsilon),
                others/(chars + epsilon),
                digits/(others + epsilon),
                conj/(words + epsilon),
                interr/(words + epsilon),
                pp/(words + epsilon),
                nouns/(words + epsilon),
                verbs/(words + epsilon),
                adj/(words + epsilon),
                articles/(words + epsilon),
                articles/(adj + epsilon),
                dets/(words + epsilon),
                aux/(words + epsilon),
                aux/(verbs + epsilon),
                chars/(sents + epsilon),
                words/(sents + epsilon),
                paragraph_lower.count("can")/(aux + epsilon),
                paragraph_lower.count("did")/(aux + epsilon),
                paragraph_lower.count(r'\b(do)\b')/(aux + epsilon),
                paragraph_lower.count("does")/(aux + epsilon),
                paragraph_lower.count(r'\b(had)\b')/(aux + epsilon),
                paragraph_lower.count(r'\b(has)\b')/(aux + epsilon),
                paragraph_lower.count("have")/(aux + epsilon),
                paragraph_lower.count("could")/(aux + epsilon),
                paragraph_lower.count("should")/(aux + epsilon),
                paragraph_lower.count("would")/(aux + epsilon),
                paragraph_lower.count(r'\b(will)\b')/(aux + epsilon),
                count_to_be_verbs/(words + epsilon),
                count_to_be_verbs/(verbs + epsilon),
                paragraph_lower.count(r'\b(am)\b')/(count_to_be_verbs + epsilon),
                paragraph_lower.count(r'\b(are)\b')/(count_to_be_verbs + epsilon),
                paragraph_lower.count(r'\b(be)\b')/(count_to_be_verbs + epsilon),
                paragraph_lower.count(r'\b(is)\b')/(count_to_be_verbs + epsilon),
                paragraph_lower.count(r'\b(was)\b')/(count_to_be_verbs + epsilon),
                paragraph_lower.count(r'\b(were)\b')/(count_to_be_verbs + epsilon),
                paragraph_lower.count(r'\b(the)\b')/(articles + epsilon),
                paragraph_lower.count(r'\b(a)\b')/(articles + epsilon),
                paragraph_lower.count(r'\b(an)\b')/(articles + epsilon),
                paragraph_lower.count(top100_common_words[0])/(words + epsilon),
                paragraph_lower.count(top100_common_words[1])/(words + epsilon),
                paragraph_lower.count(top100_common_words[2])/(words + epsilon),
                paragraph_lower.count(top100_common_words[3])/(words + epsilon),
                paragraph_lower.count(top100_common_words[4])/(words + epsilon),
                paragraph_lower.count(top100_common_words[5])/(words + epsilon),
                paragraph_lower.count(top100_common_words[6])/(words + epsilon),
                paragraph_lower.count(top100_common_words[7])/(words + epsilon),
                paragraph_lower.count(top100_common_words[8])/(words + epsilon),
                paragraph_lower.count(top100_common_words[9])/(words + epsilon),
                paragraph_lower.count(top100_common_words[10])/(words + epsilon),
                paragraph_lower.count(top100_common_words[11])/(words + epsilon),
                paragraph_lower.count(top100_common_words[12])/(words + epsilon),
                paragraph_lower.count(top100_common_words[13])/(words + epsilon),
                paragraph_lower.count(top100_common_words[14])/(words + epsilon),
                paragraph_lower.count(top100_common_words[15])/(words + epsilon),
                paragraph_lower.count(top100_common_words[16])/(words + epsilon),
                paragraph_lower.count(top100_common_words[17])/(words + epsilon),
                paragraph_lower.count(top100_common_words[18])/(words + epsilon),
                paragraph_lower.count(top100_common_words[19])/(words + epsilon),
                paragraph_lower.count(top100_common_words[20])/(words + epsilon),
                paragraph_lower.count(top100_common_words[21])/(words + epsilon),
                paragraph_lower.count(top100_common_words[22])/(words + epsilon),
                paragraph_lower.count(top100_common_words[23])/(words + epsilon),
                paragraph_lower.count(top100_common_words[24])/(words + epsilon),
                paragraph_lower.count(top100_common_words[25])/(words + epsilon),
                paragraph_lower.count(top100_common_words[26])/(words + epsilon),
                paragraph_lower.count(top100_common_words[27])/(words + epsilon),
                paragraph_lower.count(top100_common_words[28])/(words + epsilon),
                paragraph_lower.count(top100_common_words[29])/(words + epsilon),
                paragraph_lower.count(top100_common_words[30])/(words + epsilon),
                paragraph_lower.count(top100_common_words[31])/(words + epsilon),
                paragraph_lower.count(top100_common_words[32])/(words + epsilon),
                paragraph_lower.count(top100_common_words[33])/(words + epsilon),
                paragraph_lower.count(top100_common_words[34])/(words + epsilon),
                paragraph_lower.count(top100_common_words[35])/(words + epsilon),
                paragraph_lower.count(top100_common_words[36])/(words + epsilon),
                paragraph_lower.count(top100_common_words[37])/(words + epsilon),
                paragraph_lower.count(top100_common_words[38])/(words + epsilon),
                paragraph_lower.count(top100_common_words[39])/(words + epsilon),
                paragraph_lower.count(top100_common_words[40])/(words + epsilon),
                paragraph_lower.count(top100_common_words[41])/(words + epsilon),
                paragraph_lower.count(top100_common_words[42])/(words + epsilon),
                paragraph_lower.count(top100_common_words[43])/(words + epsilon),
                paragraph_lower.count(top100_common_words[44])/(words + epsilon),
                paragraph_lower.count(top100_common_words[45])/(words + epsilon),
                paragraph_lower.count(top100_common_words[46])/(words + epsilon),
                paragraph_lower.count(top100_common_words[47])/(words + epsilon),
                paragraph_lower.count(top100_common_words[48])/(words + epsilon),
                paragraph_lower.count(top100_common_words[49])/(words + epsilon),
                paragraph_lower.count(r'\b(a)\b'),
                paragraph_lower.count('about'),
                paragraph_lower.count('above'),
                paragraph_lower.count('after'),
                paragraph_lower.count(r'\b(all)\b'),
                paragraph_lower.count('although'),
                paragraph_lower.count(r'\b(am)\b'),
                paragraph_lower.count('among'),
                paragraph_lower.count(r'\b(an)\b'),
                paragraph_lower.count(r'\b(and)\b'),
                paragraph_lower.count('another'),
                paragraph_lower.count(r'\b(any)\b'),
                paragraph_lower.count('anybody'),
                paragraph_lower.count('anyone'),
                paragraph_lower.count('anything'),
                paragraph_lower.count(r'\b(are)\b'),
                paragraph_lower.count('around'),
                paragraph_lower.count(r'\b(as)\b'),
                paragraph_lower.count(r'\b(at)\b'),
                paragraph_lower.count(r'\b(be)\b'),
                paragraph_lower.count('because'),
                paragraph_lower.count('before'),
                paragraph_lower.count('behind'),
                paragraph_lower.count('below'),
                paragraph_lower.count('beside'),
                paragraph_lower.count('between'),
                paragraph_lower.count('both'),
                paragraph_lower.count(r'\b(but)\b'),
                paragraph_lower.count(r'\b(by)\b'),
                paragraph_lower.count(r'\b(can)\b'),
                paragraph_lower.count(r'\b(do)\b'),
                paragraph_lower.count('down'),
                paragraph_lower.count('each'),
                paragraph_lower.count('either'),
                paragraph_lower.count('enough'),
                paragraph_lower.count('every'),
                paragraph_lower.count('everybody'),
                paragraph_lower.count('everyone'),
                paragraph_lower.count('everything'),
                paragraph_lower.count('few'),
                paragraph_lower.count('following'),
                paragraph_lower.count(r'\b(for)\b'),
                paragraph_lower.count('from'),
                paragraph_lower.count('have'),
                paragraph_lower.count(r'\b(he)\b'),
                paragraph_lower.count(r'\b(her)\b'),
                paragraph_lower.count(r'\b(him)\b'),
                paragraph_lower.count(r'\b(i)\b'),
                paragraph_lower.count(r'\b(if)\b'),
                paragraph_lower.count(r'\b(in)\b'),
                paragraph_lower.count('including'),
                paragraph_lower.count('inside'),
                paragraph_lower.count(r'\b(into)\b'),
                paragraph_lower.count(r'\b(is)\b'),
                paragraph_lower.count(r'\b(it)\b'),
                paragraph_lower.count(r'\b(its)\b'),
                paragraph_lower.count('latter'),
                paragraph_lower.count(r'\b(less)\b'),
                paragraph_lower.count(r'\b(like)\b'),
                paragraph_lower.count('little'),
                paragraph_lower.count(r'\b(lots)\b'),
                paragraph_lower.count('many'),
                paragraph_lower.count(r'\b(me)\b'),
                paragraph_lower.count(r'\b(more)\b'),
                paragraph_lower.count(r'\b(most)\b'),
                paragraph_lower.count('much'),
                paragraph_lower.count(r'\b(my)\b'),
                paragraph_lower.count('need'),
                paragraph_lower.count('neither'),
                paragraph_lower.count(r'\b(no)\b'),
                paragraph_lower.count('nobody'),
                paragraph_lower.count(r'\b(none)\b'),
                paragraph_lower.count(r'\b(nor)\b'),
                paragraph_lower.count('nothing'),
                paragraph_lower.count(r'\b(of)\b'),
                paragraph_lower.count(r'\b(off)\b'),
                paragraph_lower.count(r'\b(on)\b'),
                paragraph_lower.count('once'),
                paragraph_lower.count(r'\b(one)\b'),
                paragraph_lower.count('onto'),
                paragraph_lower.count('opposite'),
                paragraph_lower.count(r'\b(or)\b'),
                paragraph_lower.count(r'\b(our)\b'),
                paragraph_lower.count('outside'),
                paragraph_lower.count('over'),
                paragraph_lower.count(r'\b(some)\b'),
                paragraph_lower.count('somebody'),
                paragraph_lower.count('someone'),
                paragraph_lower.count('something'),
                paragraph_lower.count('such'),
                paragraph_lower.count('than'),
                paragraph_lower.count('that'),
                paragraph_lower.count(r'\b(the)\b'),
                paragraph_lower.count(r'\b(their)\b'),
                paragraph_lower.count(r'\b(them)\b'),
                paragraph_lower.count(r'\b(these)\b'),
                paragraph_lower.count(r'\b(they)\b'),
                paragraph_lower.count(r'\b(this)\b'),
                paragraph_lower.count(r'\b(those)\b'),
                paragraph_lower.count('though'),
                paragraph_lower.count('through'),
                paragraph_lower.count(r'\b(till)\b'),
                paragraph_lower.count(r'\b(to)\b'),
                paragraph_lower.count('toward '),
                paragraph_lower.count('towards'),
                paragraph_lower.count('under'),
                paragraph_lower.count('unless'),
                paragraph_lower.count('whether'),
                paragraph_lower.count('which'),
                paragraph_lower.count('while'),
                paragraph_lower.count(r'\b(who)\b'),
                paragraph_lower.count('whoever'),
                paragraph_lower.count('whom'),
                paragraph_lower.count('whose'),
                paragraph_lower.count(r'\b(will)\b'),
                paragraph_lower.count(r'\b(with)\b'),
                paragraph_lower.count('within'),
                paragraph_lower.count('without'),
                paragraph_lower.count('worth'),
                paragraph_lower.count('would'),
                paragraph_lower.count(r'\b(yes)\b'),
                paragraph_lower.count(r'\b(you)\b'),
                paragraph_lower.count(r'\b(your)\b'),
                paragraph_lower.count(r'([^\w\s])\1')/(char_two_grams + epsilon),
            ]
            F = np.concatenate((F, F_sents))

        return F
