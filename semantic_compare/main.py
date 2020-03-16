import numpy as np
import tensorflow_hub as hub
from collections import deque
import spacy


class SemanticComparator:
    def __init__(self, merge_entities=True, sentencizer=True, spacy_model='en_core_web_lg'):
        self.nlp = spacy.load(spacy_model)
        self.activated = spacy.prefer_gpu()
        # merge tokens for entities
        if merge_entities:
            merge_ents = self.nlp.create_pipe("merge_entities")
            self.nlp.add_pipe(merge_ents)
        # merge tockes for noun chunks
        merge_nps = self.nlp.create_pipe("merge_noun_chunks")
        self.nlp.add_pipe(merge_nps)
        # split by sents
        if sentencizer:
            split_by_sents = self.nlp.create_pipe('sentencizer')
            self.nlp.add_pipe(split_by_sents, before='parser')

        self.semantic_model = None

    def split_by_sents(self, doc, spacy_model='en_core_web_lg'):
        nlp = spacy.load(spacy_model)
        split_by_sents = nlp.create_pipe('sentencizer')    
        nlp.add_pipe(split_by_sents, before='parser')
        return nlp(doc)

    def add_custom_pipe(self, custom_pipe, *args, **kwargs):
        self.nlp.add_pipe(custom_pipe, **kwargs)

    def find_heads(self, token):
        """ Function finds all words that are situated before and are connected to a given word.
            How to use:
                Pass a word, recieve its heads.
        """
        head = token
        heads = []
        is_break = False
        while head.dep_ != "ROOT":
            head = head.head
            if head.pos_ == 'VERB':
                for c in head.children:
                    # if verb has connection with another nouns it means that it has not connection with given word
                    if c != token and c.pos_ == 'NOUN' and c not in heads:
                        is_break = True
                        break
                if is_break == True:
                    break
            heads.append(head)

        return heads

    def find_children(self, token):
        """ Function finds all words that are situated after and are connected to a given word.
            How to use:
                Pass a word, recieve its children.
        """
        # depth first algorithm
        search_queue = deque()
        search_queue += list(token.children)
        searched = set()
        while search_queue:
            child = search_queue.popleft()
            if not child in searched:
                search_queue += child.children
                searched.add(child)
        return searched

    def extract_phrases(self, doc):
        """ Function finds all sentences and then simplifies them.
            How the function works:
            1. iterates through all noun chunks
            2. finds connections between nouns and other words(heads, children)
            3. joins all words in right order
            How to use:
                Pass a document, recieve all simplified sentences.
        """
        doc = self.nlp(doc)

        simplified = set()
        for sent in doc.sents:
            sent_pos = {}
            # save positions of words
            for i, token in enumerate(sent):
                sent_pos[token.text] = i
            # iterate through each noun_chunk
            for noun_chunk in sent.noun_chunks:
                if noun_chunk.root.dep_ == 'pobj':
                    continue
                heads = self.find_heads(noun_chunk.root)
                children = self.find_children(noun_chunk.root)
                if heads:
                    # container where we will add all words of simplified sentence
                    container = []
                    # iterate through each head element
                    for head in heads:
                        verbs = []
                        # add verbs to verbs list
                        if head.pos_ == 'VERB':
                            verbs.append(head)
                        else:
                            # add only words to container that are correctly connected to noun_chunk
                            if head.dep_ != 'nmod' and head.dep_ != 'conj' and head.dep_ != 'compound':
                                splited_by_cc = False
                                for head_child in head.children:
                                    if head_child.pos_ == 'PUNCT' or head_child.pos_ == 'cc':
                                        splited_by_cc = True
                                        break
                                if splited_by_cc == False:
                                    container.append(head.text)
                        # sort container by word position in sentence
                        container = sorted(
                            container, key=lambda el: sent_pos[el])
                        if verbs:
                            for verb in verbs:
                                # copy container for each verb
                                joined_container = container[:]
                                # add verb
                                joined_container.append(verb.text)

                                # add noun_chunk
                                joined_container.append(noun_chunk.text)
                                # sort by word position
                                joined_container = sorted(
                                    joined_container, key=lambda el: sent_pos[el])
                                # if children exist, we add text of each children filtered by word positions after noun_chunk
                                if children:
                                    children = sorted(
                                        [c for c in children], key=lambda el: sent_pos[el.text])
                                    for c_index, c in enumerate(children):
                                        if c.pos_ == 'PUNCT' or c.dep_ == 'cc' or c.dep_ == 'nsubj':
                                            children = children[:c_index]
                                            break
                                    joined_container += children

                                # convert to string
                                joined_container = " ".join(
                                    map(str, joined_container))
                                # add to simplified set
                                simplified.add(joined_container)
                else:
                    # if have no heads, just add noun_chunks with children if exist
                    children = self.find_children(noun_chunk.root)
                    joined_container = [noun_chunk.text, ]
                    # sort by word positions in sentence
                    joined_container = sorted(
                        joined_container, key=lambda el: sent_pos[el])
                    # if children exist, we add text of each children filtered by word positions after noun_chunk
                    if children:
                        children = sorted([c for c in children],
                                          key=lambda el: sent_pos[el.text])
                        for c_index, c in enumerate(children):
                            # if here is cc or PUNCT we stop iterating
                            if c.pos_ == 'PUNCT' or c.dep_ == 'cc' or c.dep_ == 'nsubj':
                                children = children[:c_index]
                                break
                        joined_container += children
                    simplified.add(" ".join(map(str, joined_container)))

        return list(simplified)

    def load_semantic_model(self, path='https://tfhub.dev/google/universal-sentence-encoder-large/5'):
        # Loading pretrained model for extracting featrures from sentences
        module_url = path
        self.semantic_model = hub.load(module_url)

    def embed(self, input):
        return self.semantic_model(input)

    def compare_phrases(self, phrase1, phrase2):
        if not self.semantic_model:
            self.load_semantic_model()
        features1 = self.embed((phrase1,))
        features2 = self.embed((phrase2,))
        simularity = np.ndarray.tolist(np.inner(features1, features2))[0][0]
        return simularity

    def build_similarity_matrix(self, document1, document2):
        """
            Function extracts features from senteces and finds how simular they are
        """
        if not self.semantic_model:
            self.load_semantic_model()

        #   Extracting features from docs
        features1 = self.embed(document1)
        features2 = self.embed(document2)
        similarity = np.inner(features1, features2)
        return similarity

        # find index of element in list
    def _find_index(self, lst, val):
        index = -1
        for j, el in enumerate(lst):
            if el == val:
                index = j
                break
        return index

    def bucket_sorting(self, document1, document2, similarity, cut_off=0.3):
        in_doc1 = set()
        in_doc2 = set()
        in_both = set()
        for i in range(len(similarity)):
            max_corr = max(similarity[i])
            # Cut off small values
            if max_corr > cut_off:
                # Find index of the most similar combination
                max_i = self._find_index(similarity[i], max_corr)
                # similar combination
                comb = document2[max_i]
                in_both.add((document1[i], comb))
            else:
                in_doc1.add(document1[i])
        for sent in document2:
            if sent not in in_both:
                in_doc2.add(sent)
        return in_both, in_doc1, in_doc2

    def transform_phrase(self, phrase1, phrase2):
        """
            Find all allign phrases
        """
        phrases = []
        if phrase1 != phrase2:
            possible_combs = []
            combs2 = self.nlp(phrase2)
            combs1 = self.nlp(phrase1)
            role_nouns = set(combs2.noun_chunks)
            combs2_verbs = [el for el in combs2 if el.pos_ == 'VERB']
            combs1_verbs = set(el for el in combs1 if el.pos_ == 'VERB')
            phrases.append(phrase1)
            for combs1_verb in combs1_verbs:
                noun_chunk = " ".join(list(map(str, combs1_verb.subtree))[1:])
                for verb in combs2_verbs:
                    phrases.append(" ".join((verb.text, noun_chunk)))

            phrases.append(phrase2)
        else:
            return [phrase1]
        return phrases
