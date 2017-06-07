import os
import pickle
import re
from nltk import sent_tokenize, word_tokenize
from document_class import Document
from data import *

def extract_metadata(string):
    '''
    Pulls the metadata content out of a Helsinki corpus metadata tag.
    '''
    return " ".join(string.strip(">").split()[1:])

def char_replace(text):
    '''Replace special character encodings with the actual characters.'''
    text = text.replace("+A", "Æ") # capital ash Æ
    text = text.replace("+a", "æ") # lowecase ash æ
    text = text.replace("+D", "Ý") # capital eth Ý
    text = text.replace("+d", "ð") # lowercase eth ð
    text = text.replace("+G", "Ȝ") # capital yogh Ȝ
    text = text.replace("+g", "ȝ") # lowercase yogh ȝ
    text = text.replace("+TT", "Ꝥ") # capital crossed thorn Ꝥ
    text = text.replace("+Tt", "Ꝥ") # capital crossed thorn Ꝥ
    text = text.replace("+tt", "ꝥ") # lowercase crossed thorn ꝥ
    text = text.replace("+T", "Þ") # capital thorn Þ
    text = text.replace("+t", "þ") # lowercase thorn þ
    text = text.replace("+L", "£") # pound sign £
    text = text.replace("+e", "ę") # lowercase e caudata ę
    return text

newline = re.compile(r"#\n") # finds line continuation chars
font = re.compile(r"\(\^(.*?)\^\)", re.DOTALL) # font other than basic font
foreign = re.compile(r"\(\\(.*?)\\\)", re.DOTALL) # foreign language
runes = re.compile(r"\(}(.*?)}\)", re.DOTALL) # runes
emendation = re.compile(r"\[{(.*?){\]", re.DOTALL) # emendations
editor = re.compile(r"\[\\.*?\\\]", re.DOTALL) # editor's comment
heading = re.compile(r"\[}(.*?)}\]", re.DOTALL) # heading
helsinki = re.compile(r"\[\^.*?\^\]", re.DOTALL) # helsinki corpus comments
empty_heading = re.compile(r"\[}\s*?}\]", re.DOTALL) # empty heading

def clean_text(text):
    '''
    Removes Helsiniki corpus markup from the document text and returns
    the cleaned text.
    '''
    # clean newline markers
    text = re.sub(newline, "\n", text)
    # clean emendation markers
    text = re.sub(emendation, "\g<1>", text)
    # clean editor and corpus comments
    text = re.sub(editor, "", text)
    text = re.sub(helsinki, "", text)
    # clean formatting markers
    text = re.sub(font, "\g<1>", text)
    text = re.sub(heading, "\g<1>", text)
    # remove already empty headings
    text = re.sub(empty_heading, "", text)
    # remove foreign text and runes on no_foreign varieties
    no_foreign = re.sub(foreign, "", text)
    no_foreign = re.sub(runes, "", text)
    # clean foreign text and rune markers
    text = re.sub(foreign, "\g<1>", text)
    text = re.sub(runes, "\g<1>", text)

    # run appropriate things through char_replace
    text = char_replace(text)
    no_foreign = char_replace(no_foreign)

    return text, no_foreign

train_docs = []
devtest_docs = []
test_docs = []
for file in os.listdir(CORPUS_DIR):
    if file[0] != "c":
        continue
    full_text = open(CORPUS_DIR + file).read()
    doc = Document(file)
    raw_text = ""
    for line in full_text.split("\n"):
        if line.startswith("<N "):
            name = extract_metadata(line)
            # if we're on a new document in the same file
            if raw_text.strip() != "" and name != doc.name:
                # save raw text
                doc.raw_text = raw_text
                # clean text
                cleaned, cleaned_foreign = clean_text(raw_text)
                # replace newlines with a space
                cleaned = re.sub(r"\n\n*", " ", cleaned)
                cleaned_foreign = re.sub(r"\n\n*", " ", cleaned_foreign)
                # save the cleaned forms
                doc.cleaned_text = cleaned
                doc.cleaned_text_no_foreign = cleaned_foreign
                # save char lists
                doc.chars = list(cleaned)
                doc.chars_no_foreign = list(cleaned_foreign)
                # add doc to list
                if doc.filename in TRAIN:
                    train_docs.append(doc)
                elif doc.filename in DEVTEST:
                    devtest_docs.append(doc)
                else:
                    test_docs.append(doc)
                # create new doc and reset raw text
                doc = Document(file)
                raw_text = ""
            doc.name = name
        elif line.startswith("<C "):
            doc.corpus_section = extract_metadata(line)
        elif line.startswith("<O "):
            doc.o_date = extract_metadata(line)
        elif line.startswith("<M "):
            doc.m_date = extract_metadata(line)
        elif line.startswith("<D "):
            doc.dialect = extract_metadata(line)
        elif line.startswith("<V "):
            doc.style = extract_metadata(line)
        elif line.startswith("<G "):
            doc.translate_relation = extract_metadata(line)
        elif line.startswith("<F "):
            doc.original_language = extract_metadata(line)
        elif line.startswith("<"):
            continue
        else:
            raw_text += line + "\n"

    # save final part of the document
    # save raw text
    doc.raw_text = raw_text
    # clean text
    cleaned, cleaned_foreign = clean_text(raw_text)
    # replace newlines with a space
    cleaned = re.sub(r"\n\n*", " ", cleaned)
    cleaned_foreign = re.sub(r"\n\n*", " ", cleaned_foreign)
    # save the cleaned forms
    doc.cleaned_text = cleaned
    doc.cleaned_text_no_foreign = cleaned_foreign
    # save char lists
    doc.chars = list(cleaned)
    doc.chars_no_foreign = list(cleaned_foreign)
    # add doc to list
    if doc.filename in TRAIN:
        train_docs.append(doc)
    elif doc.filename in DEVTEST:
        devtest_docs.append(doc)
    else:
        test_docs.append(doc)

os.makedirs("cached_data", exist_ok=True)
with open("cached_data/train_docs.pkl", 'wb') as f:
    pickle.dump(train_docs, f)
with open("cached_data/devtest_docs.pkl", 'wb') as f:
    pickle.dump(devtest_docs, f)
with open("cached_data/test_docs.pkl", 'wb') as f:
    pickle.dump(test_docs, f)
print("Done!")
