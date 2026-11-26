"""WordNet, spaCy and NER annotations of definienda and definition tokens.

For a WordNet synset, the definiendum gets its WordNet id, name and part of
speech, its Brown-corpus and WordNet frequency, its lexname category, and
abstraction and generality scores. Each token of the cleaned definition gets
its spaCy part of speech (mapped to WordNet tags) and dependency label, its
NER tag, and, when WordNet knows the token, the same WordNet attributes.

``WordNetAnnotator.annotate_wordnet()`` builds ``data/WordNet_filtered_data.json``
over every synset with an alphabetic name. ``annotate_records`` adds the same
annotations to definition records that carry a ``wordnet_id``.

Requires spaCy with ``en_core_web_sm``, NLTK with the ``wordnet`` and
``brown`` corpora, and a token-classification model for NER. Models are
loaded when an annotator is created, not on import.
"""

import json
import random
import re
from pathlib import Path

import numpy as np

# spaCy POS tags -> WordNet POS tags ('0' when unmapped).
SPACY_TO_WORDNET_POS = {
    'ADJ': 'a',   # Adjective
    'ADP': 'r',   # Adposition
    'ADV': 'r',   # Adverb
    'AUX': 'v',   # Auxiliary verb
    'CCONJ': 'r',  # Coordinating conjunction
    'DET': 'r',   # Determiner
    'INTJ': 'r',   # Interjection
    'NOUN': 'n',  # Noun
    'NUM': 'r',   # Numeral
    'PART': 'r',  # Particle
    'PRON': 'n',  # Pronoun
    'PROPN': 'n',  # Proper noun
    'PUNCT': 'r',  # Punctuation
    'SCONJ': 'r',  # Subordinating conjunction
    'SYM': 'r',   # Symbol
    'VERB': 'v',  # Verb
    'X': 'r'      # Other
}


def custom_tokenizer(nlp):
    """spaCy tokenizer that splits on in-word punctuation but keeps hyphenated words."""
    from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, LIST_ELLIPSES, LIST_ICONS
    from spacy.lang.de.punctuation import _quotes
    from spacy.tokenizer import Tokenizer
    from spacy.util import compile_infix_regex

    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[{al}])\.(?=[{au}])".format(al=ALPHA_LOWER, au=ALPHA_UPPER),
            r"(?<=[{a}])[,!?](?=[{a}])".format(a=ALPHA),
            r'(?<=[{a}])[:<>=](?=[{a}])'.format(a=ALPHA),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            r"(?<=[{a}])([{q}\]\[])(?=[{a}])".format(a=ALPHA, q=_quotes),
            r"(?<=[0-9])-(?= [0-9])",
        ]
    )
    infix_re = compile_infix_regex(infixes)
    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                     suffix_search=nlp.tokenizer.suffix_search,
                     infix_finditer=infix_re.finditer,
                     token_match=nlp.tokenizer.token_match,
                     rules=nlp.Defaults.tokenizer_exceptions)


def clean_definition(sentence):
    """A WordNet gloss with special characters removed.

    A leading parenthesised part, e.g. ``"(botany) ..."``, is moved to the end
    after a semicolon; only that part is stripped of special characters.
    Otherwise everything but letters and spaces is removed (hyphens included).
    """
    match = re.match(r'\(([^)]+)\)\s*(.*)', sentence)
    if match:
        enclosed_part = match.group(1)
        rest_of_definition = match.group(2)
        cleaned_enclosed_part = re.sub(r'[^a-zA-Z\s]', '', enclosed_part)
        cleaned_enclosed_part = cleaned_enclosed_part.replace('-', '_')
        cleaned_def = f"{rest_of_definition.strip()}; {cleaned_enclosed_part}"
    else:
        cleaned_def = re.sub(r'[^a-zA-Z\s]', '', sentence)
        cleaned_def = ' '.join(cleaned_def.split())
    return cleaned_def


def choose_synset(token, token_pos, synsets, choose_first=False):
    """Synset with the token's part of speech and morphological base form."""
    from nltk.corpus import wordnet

    if choose_first:
        return synsets[0]
    selected = [s for s in synsets if s.name().split('.')[1] == token_pos]
    selected = [s for s in selected if s.name().split('.')[0] == wordnet.morphy(token, token_pos)]
    if len(selected) > 0:
        return selected[0]
    return None


def freq_lemma_count(synset):
    """Summed corpus counts of the synset's lemmas."""
    freq = 0
    for l in synset.lemmas():
        freq += l.count()
    return freq


def get_hypernyms(synset):
    """Direct hypernyms followed by the (nested) hypernyms of each."""
    hypernyms = synset.hypernyms()
    if not hypernyms:
        return []
    return hypernyms + [h for h in map(get_hypernyms, hypernyms)]


def closure_score(synset):
    """Size of the hypernym closure plus the hyponym closure."""
    hypernym_level = len(list(synset.closure(lambda s: s.hypernyms())))
    hyponym_level = len(list(synset.closure(lambda s: s.hyponyms())))
    return hypernym_level + hyponym_level


def calculate_generality_score_from_synset(synset):
    """Number of hyponyms plus number of topic domains."""
    if not synset:
        return None
    generality_score = 0
    generality_score += len(synset.hyponyms())
    generality_score += len(synset.topic_domains())
    return generality_score


def calculate_abstraction_score_from_synset(synset, hypernyms=True, part_whole=True,
                                            attributes=False, frames=False, mst=False,
                                            closures=False):
    """Abstraction score from hypernym depth, part-whole relations and,
    optionally, attributes, verb frames, also-see trees and closures."""
    if not synset:
        return None
    abstraction_score = 0
    if hypernyms:
        all_hypernyms = [hypernym for hypernym in get_hypernyms(synset)]
        if all_hypernyms:
            abstraction_level = max(len(path) if isinstance(path, list) else 1
                                    for path in all_hypernyms)
            abstraction_score += abstraction_level
    if part_whole:
        abstraction_score += len(synset.part_meronyms()) + len(synset.part_holonyms())
    if attributes:
        abstraction_score += len(synset.attributes())
    if frames and synset.pos() == 'v':
        abstraction_score += len(synset.frame_ids())
    if mst:
        abstraction_score += len(synset.mst(lambda s: s.also_sees()))
    if closures:
        abstraction_score += closure_score(synset)
    return abstraction_score


class WordNetAnnotator:
    """Holds the spaCy pipeline, Brown frequencies and NER model.

    ``ner`` can be any callable with the output format of a Hugging Face
    token-classification pipeline; by default the pinned model is loaded.
    """

    def __init__(self, spacy_model="en_core_web_sm",
                 ner_model="dbmdz/bert-large-cased-finetuned-conll03-english",
                 ner_revision="4c53496", ner=None, seed=42):
        import nltk
        import spacy
        import torch
        from nltk.corpus import brown

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.nlp = spacy.load(spacy_model)
        self.nlp.tokenizer = custom_tokenizer(self.nlp)
        self.brown_word_frequencies = nltk.FreqDist(brown.words())
        if ner is None:
            from transformers import pipeline

            ner = pipeline("ner", model=ner_model, revision=ner_revision)
        self.ner_pipe = ner

    def ner_classification(self, sentence):
        """Named entities of a sentence (entity, score, index, word, start, end)."""
        preds = self.ner_pipe(f'"""{sentence}"""')
        return [
            {
                "entity": pred["entity"],
                "score": round(pred["score"], 4),
                "index": pred["index"],
                "word": pred["word"],
                "start": pred["start"],
                "end": pred["end"],
            }
            for pred in preds
        ]

    def label_words_with_pos_and_syntax(self, sentence):
        """``(token, WordNet POS, dependency label)`` for every token."""
        doc = self.nlp(sentence)
        return [(token.text, SPACY_TO_WORDNET_POS.get(token.pos_, '0'), token.dep_)
                for token in doc]

    def annotate_token(self, token, pos, syntax, definition_ner):
        from nltk.corpus import wordnet

        attribute = dict()
        token_synset = wordnet.synsets(token)
        attribute['name'] = token
        attribute['brown_frequency'] = self.brown_word_frequencies[token]
        attribute['pos'] = pos
        attribute['wordnet_org'] = False
        attribute['syntax'] = syntax
        attribute['ner'] = None
        for entity in definition_ner:
            if entity['word'] == token:
                attribute['ner'] = entity['entity']
                attribute['ner_score'] = str(entity['score'])
        if token_synset:
            if len(token_synset) == 1:
                token_synset = token_synset[0]
            elif len(token_synset) > 1:
                token_synset = choose_synset(token, pos, token_synset)
            if token_synset:
                attribute['wordnet_frequency'] = freq_lemma_count(token_synset)
                attribute['category'] = token_synset.lexname().split('.')[1]
                attribute['abstraction_level'] = calculate_abstraction_score_from_synset(token_synset)
                attribute['generalization_level'] = calculate_generality_score_from_synset(token_synset)
                attribute['definition'] = clean_definition(token_synset.definition())
                attribute['wordnet_org'] = True
        if not attribute['wordnet_org']:
            attribute['wordnet_frequency'] = None
            attribute['lemmas'] = None
            attribute['category'] = None
            attribute['abstraction_level'] = None
            attribute['generalization_level'] = None
            attribute['definition'] = None
        return attribute

    def annotate_synset(self, synset, definition=None):
        """Annotations of one synset; ``definition`` replaces its cleaned gloss."""
        synset_id = synset.name()
        name, pos = synset_id.split('.')[0:2]
        out = dict()
        out['id'] = synset_id
        out['name'] = name
        out['pos'] = pos
        out['brown_frequency'] = self.brown_word_frequencies[name]
        out['wordnet_frequency'] = freq_lemma_count(synset)
        out['category'] = synset.lexname().split('.')[1]
        out['abstraction_level'] = calculate_abstraction_score_from_synset(synset)
        out['generalization_level'] = calculate_generality_score_from_synset(synset)
        out['definition'] = (clean_definition(synset.definition())
                             if definition is None else definition)
        definition_ner = self.ner_classification(out['definition'])
        out['definition_attributes'] = [
            self.annotate_token(token, tpos, syntax, definition_ner)
            for token, tpos, syntax in self.label_words_with_pos_and_syntax(out['definition'])
        ]
        return out

    def annotate_wordnet(self, out=None, limit=None):
        """Annotate every WordNet synset with an alphabetic name.

        Writes ``out`` (e.g. ``data/WordNet_filtered_data.json``) when given.
        """
        from nltk.corpus import wordnet

        data = []
        for synset in wordnet.all_synsets():
            if limit is not None and len(data) >= limit:
                break
            if synset.name().split('.')[0].isalpha():
                data.append(self.annotate_synset(synset))
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            with open(out, 'w') as f:
                json.dump(data, f, indent=4)
            print(f'WordNet data has been saved to {out} with {len(data)} entries.')
        return data

    def annotate_records(self, records, id_field="wordnet_id", definition_field="definition"):
        """Add ``annotation`` to each record, from its synset and its own definition."""
        from nltk.corpus import wordnet

        out = []
        for rec in records:
            try:
                synset = wordnet.synset(rec[id_field])
            except Exception:
                out.append({**rec, "annotation": None})
                continue
            out.append({**rec, "annotation": self.annotate_synset(synset, rec[definition_field])})
        return out
