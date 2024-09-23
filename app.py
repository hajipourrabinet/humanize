import nltk
from nltk.corpus import wordnet
import random
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
import string

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load spaCy's English model for POS tagging
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def get_synonyms(word, pos_tag):
    """Get a list of synonyms for a given word based on its POS tag."""
    synonyms = set()
    pos = get_wordnet_pos(pos_tag)
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return synonyms

def get_wordnet_pos(treebank_tag):
    """Map POS tag to first character lemmatize() accepts."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def common_phrase_replacement(text):
    """Replace common phrases with more human-like phrases."""
    phrase_map = {
        "as soon as possible": "at your earliest convenience",
        "I am writing to you": "I'm reaching out to you",
        "due to the fact that": "because",
        "in order to": "to",
        "the quick brown fox": "a fast, brown animal",
        "over the lazy dog": "over a sluggish canine",
        "please find attached": "attached is",
        "thank you for your time": "thanks for your time",
        "should you have any questions": "if you have any questions",
        "in the event that": "if",
        "prior to": "before",
        "with regard to": "regarding",
        "at this point in time": "currently",
        "on a daily basis": "daily",
        "a large number of": "many",
        "the reason why is because": "because",
        "in the near future": "soon"
    }
    for phrase, replacement in phrase_map.items():
        # Use case-insensitive replacement
        text = replace_case_insensitive(text, phrase, replacement)
    return text

def replace_case_insensitive(text, old, new):
    """Replace all case-insensitive occurrences of old with new in text."""
    import re
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    return pattern.sub(new, text)

def humanize_text(text):
    """Humanize text by replacing words with synonyms and applying randomness."""
    doc = nlp(text)
    new_tokens = []

    for token in doc:
        if token.is_punct or token.is_space:
            new_tokens.append(token.text)
            continue

        # Get synonyms based on POS tag
        synonyms = get_synonyms(token.text.lower(), token.tag_)
        if synonyms:
            # Decide whether to replace
            if random.random() < 0.3:  # 30% chance to replace
                synonym = random.choice(list(synonyms))
                # Preserve capitalization
                if token.text[0].isupper():
                    synonym = synonym.capitalize()
                new_tokens.append(synonym)
            else:
                new_tokens.append(token.text)
        else:
            new_tokens.append(token.text)

    return ''.join(new_tokens)

def sentence_restructuring(text):
    """Restructure sentences to improve flow."""
    sentences = sent_tokenize(text)
    
    # Simple restructuring: Shuffle sentence order with constraints
    if len(sentences) > 1:
        # To maintain some logical flow, avoid shuffling the first and last sentences
        middle_sentences = sentences[1:-1]
        random.shuffle(middle_sentences)
        sentences = [sentences[0]] + middle_sentences + [sentences[-1]]
    
    return ' '.join(sentences)

# Read input text from file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Save humanized text to a file
def save_text_to_file(file_path, text):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

# Specify the input and output file paths
input_file = 'input_text.txt'
output_file = 'humanized_text.txt'

# Read the input text from the file
input_text = read_text_from_file(input_file)

# Apply phrase replacement
text_after_phrase_replacement = common_phrase_replacement(input_text)

# Humanize the text with synonyms
humanized_text = humanize_text(text_after_phrase_replacement)

# Restructure sentences
humanized_text = sentence_restructuring(humanized_text)

# Optionally, you can perform additional passes
# For example, another round of phrase replacement or synonym replacement

# Save the humanized text to a file
save_text_to_file(output_file, humanized_text)

print("Original text after phrase replacement:", text_after_phrase_replacement)
print("Highly humanized text saved to:", output_file)
