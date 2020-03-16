import unittest2 as unittest
import sys
sys.path.append('../../')
from semantic_compare import SemanticComparator as sc

class ExtractingPhrasesTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ExtractingPhrasesTests, self).__init__(*args, **kwargs)
        self.comparator = sc()
    
    def test_basic(self):
        comparator = self.comparator
        doc = "Create, promote and develop a business"
        phrases = comparator.extract_phrases(doc)
        expected_phrases = ['Create a business',
                            'promote a business', 'develop a business']
        self.assertCountEqual(phrases, expected_phrases)

    def test_split_by_senteces(self):
        comparator = self.comparator
        doc = '''Experience shaping the BI strategy from C-Level to Technical developers. Extensive delivery of platform within a Business Intelligence and Analytics function. Communication with stakeholders on all levels.'''
        doc = comparator.split_by_sents(doc)
        count_sents = len(list(doc.sents))
        print([s for s in doc.sents])
        self.assertEqual(count_sents, 3)


class ComparingPhrasesTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ComparingPhrasesTests, self).__init__(*args, **kwargs)
        self.comparator = sc()

    def test_basic(self):
        comparator = self.comparator

        phrase1 = 'Understand customer needs'
        phrase2 = 'Capture business requirements'
        self.assertEqual(
            round(comparator.compare_phrases(phrase1, phrase2), 3), 0.386)

    def test_similarity_matrix(self):
        comparator = self.comparator
        doc1 = "Create, promote and develop a business"
        doc2 = '''
        * Experience shaping the BI strategy from C-Level to Technical developers.
        * Extensive delivery of platform within a Business Intelligence and Analytics function.
        * Communication with stakeholders on all levels.
        '''
        phrases_1 = comparator.extract_phrases(doc1)
        phrases_2 = comparator.extract_phrases(doc2)
        similarity = comparator.build_similarity_matrix(phrases_1, phrases_2)
        return self.assertEqual(len(phrases_1), len(similarity))

    def test_align_phrases(self):
        comparator = self.comparator
        phrase1 = 'Understand customer needs'
        phrase2 = 'Capture business requirements'
        transformed = comparator.transform_phrase(phrase1, phrase2)
        return self.assertIn('Capture customer needs', transformed)

    def test_bucket_sorting(self):
        comparator = self.comparator
        doc1 = "Create, promote and develop a business"
        doc2 = '''
        * Experience shaping the BI strategy from C-Level to Technical developers.
        * Extensive delivery of platform within a Business Intelligence and Analytics function.
        * Communication with stakeholders on all levels.
        '''
        phrases_1 = comparator.extract_phrases(doc1)
        phrases_2 = comparator.extract_phrases(doc2)
        similarity = comparator.build_similarity_matrix(phrases_1, phrases_2)
        in_both, in_doc1, in_doc2 = comparator.bucket_sorting(
            phrases_1, phrases_2, similarity, cut_off=0.5)

