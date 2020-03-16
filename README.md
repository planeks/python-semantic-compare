# python-semantic-сompare
Exctracts, compares, transforms and sorts with buckets phrases.
### Installation
Project requires spacy model for natural language processing. If you want to use english, please run this command
```sh
$ python -m spacy download en_core_web_lg
```
## Usage
#### Extract phrases
**Simple Usage**
```sh
from semantic_compare import SemanticComparator as sc
comparator = sc(sentencizer=True)
phrases = comparator.extract_phrases("Create, promote and develop a business.")
```
**Output:**
```sh
['Create a business','promote a business','develop a business']
```
```sentencizer``` - splitter of sentences by punctuation(dot, question mark, exclamation mark).

**Advanced Usage**
```sh
from semantic_compare import SemanticComparator as sc

# Sentence splitter
def our_sentencizer(doc):
    """
    Sentence splitter function that allows splitting document on sentences
    by different punctuations and new line
    """
    for i, token in enumerate(doc[:-2]):
        if token.text == "•" or "•" in token.text:
            doc[i].is_sent_start = True
        elif (token.text == "." or token.text == '...' \ 
            or token.text == '?' or token.text == '!' or token.text == '\n') \
            and doc[i+1].is_title:
            doc[i+1].is_sent_start = True
        else:
            doc[i+1].is_sent_start = False
    return doc


# Merge enteties and buld noun chunks
comparator = sc(merge_entities=False, spacy_model='en_core_web_sm')
    
# Add custom pipe for text preprocessing
comparator.add_custom_pipe(our_sentencizer, before='parser')

phrases = comparator.extract_phrases('''
Must Have:
* Experience shaping the BI strategy from C-Level to Technical developers.
* Extensive delivery of platform within a Business Intelligence and Analytics function.
* Communication with stakeholders on all levels.
''')
print('\n'.join(phrases))
```
If you need to merge named entities or noun chunks, check out the built-in merge_entities and ```merge_noun_chunks```, ```merge_noun_chunks```.
Using ```add_custom_pipe``` you can add your custom pipe for text processing in spacy.
### Compare phrases (Semantic similarity)
Get the similarity of phrases against each other.
**Example 1:**
```sh
phrase1 = 'Understand customer needs'
phrase2 = 'Capture business requirements'
similarity = comparator.compare_phrases(phrase1, phrase2)
print(similarity)
```
**Output:**
```
0.38569751
```
**Example 2:**
Get a two dimensional matrix that clusters the similarity of phrases against each other.
```sh
phrases_1 = [
    'Communication with stakeholders',
    'Understand customer needs',
    'Experience shaping the BI strategy',
    'shaping the BI strategy',
    'Delivery of platform Analytics function',
]

phrases_2 = [
    'Extensive delivery of platform within a Business Intelligence and Analytics function',
    'shaping the BI strategy',
    'Experience shaping the BI strategy from C-Level to Technical developers',
    'Communication with stakeholders on all levels',
    'Capture business requirements',
    'Play computer games',
]
similarity = comparator.build_similarity_matrix(phrases_1, phrases_2)
print(similarity)
```
**Output:**
```
[[-0.03689054  0.0372301   0.17840812  0.09079809  0.65748763]
 [ 0.18079719  0.12055688  0.77624094  1.          0.22749564]
 [ 0.08472343  0.11505745  0.7030021   0.48876476  0.13252231]
 [ 0.7132235   0.07449755  0.178031    0.15712512  0.0676512 ]
 [ 0.11637229  0.38569745  0.23005028  0.25646406  0.26493344]
 [ 0.17955953  0.15243992  0.11233422  0.16087453  0.03144675]]
```
## Bucket sorting
When you compare two documents you can see which phrases present in both or only in a specific document.
```sh
phrases_1 = [
    'Communication with stakeholders',
    'Understand customer needs',
    'Experience shaping the BI strategy',
    'shaping the BI strategy',
    'Delivery of platform Analytics function',
]

phrases_2 = [
    'Extensive delivery of platform within a Business Intelligence and Analytics function',
    'shaping the BI strategy',
    'Experience shaping the BI strategy from C-Level to Technical developers',
    'Communication with stakeholders on all levels',
    'Capture business requirements',
    'Play computer games',
]
# cut_off - a percentage of similarity should be bigger than it so that we consider that phrases are similar(default=0.3)
in_both, in_doc1, in_doc2 = comparator.bucket_sorting(
    phrases_1, phrases_2, similarity, cut_off=0.5)
```
## Transfrom phrases
Get all steps of transformation from one phrase to another.
**Example:**
```sh
print(comparator.transform_phrase(
    'Understand customer needs',
    'Capture business requirements',
))
```
**Output**
```sh
["Understand customer needs", "Capture customer needs", "Capture business requirements"]
```
