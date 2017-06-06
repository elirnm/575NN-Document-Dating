class Document:
    def __init__(self, filename):
        ########################################
        # metadata
        ########################################
        # name of the documents: <N *>
        self.name = ""
        # date of the original: <O *>
        self.o_date = ""
        # date of the manuscript: <M *>
        self.m_date = ""
        # dialect: <D *>
        self.dialect = ""
        # section of the corpus: <C *>
        self.corpus_section = ""
        # text style: <V *>
        self.style = ""
        # relationship to translated work
        self.translate_relation = ""
        # original lanugage of foreign work
        self.original_language = ""
        # filename
        self.filename = filename

        ########################################
        # text
        ########################################
        # the text as in the corpus
        self.raw_text = ""
        # cleaned text; see preprocess.clean_text()
        self.cleaned_text = ""
        # cleaned text with no foreign languages; see preprocess.clean_text()
        self.cleaned_text_no_foreign = ""
        # the document as characters
        self.chars = []
        # chars with no foreign languages
        self.chars_no_foreign = []
