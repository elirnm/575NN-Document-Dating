import pickle
import sys

data = pickle.load(open("data/documents.pkl", 'rb'))

def count(lang_id):
    file_dict = {}
    for file in data:
        if file.filename[1] == lang_id:
            if file.filename in file_dict:
                file_dict[file.filename] += len(file.cleaned_text_no_format.split())
            else:
                file_dict[file.filename] = len(file.cleaned_text_no_format.split())
    return file_dict

def split(lang, files):
    fnames = sorted(list(files.keys()))
    print("{0} total {1} files".format(len(fnames), lang))
    total_words =sum(files.values())
    print("{0} total {1} words".format(total_words, lang))
    train = fnames[:int(len(fnames) * 0.8)]
    test = fnames[int(len(fnames) * 0.8):]
    print("{0} training {1} files".format(len(train), lang))
    train_words = sum([files[x] for x in train])
    print("{0} training {1} words".format(train_words, lang))
    print("{0} testing {1} files".format(len(test), lang))
    test_words = sum([files[x] for x in test])
    print("{0} testing {1} words".format(test_words, lang))
    print("{0}/{1} {2} split".format((train_words / total_words) * 100, (test_words / total_words) * 100, lang))
    print()
    print(lang + "\n" + str(train) + "\n" + str(test) + "\n\n", file=sys.stderr)

# get files
oe_files = count('o')
me_files = count('m')
emode_files = count('e')

split("OE", oe_files)
split("ME", me_files)
split("EModE", emode_files)
