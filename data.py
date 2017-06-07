# The alphabet for character vectors
ALPHABET = [' ', 'Ꝥ', 'ę', 'I', '/', '2', 'V', 'Z', '1', 'o', 'P', 'j', ')',
            't', ':', 'K', 'A', 'ꝥ', 'f', '6', '-', '=', 'L', '8', 'Ȝ', 'c',
            'k', 'R', 'l', 'þ', 'M', 'q', '4', 'G', '~', '(', 'ȝ', 'æ', 'r',
            "'", 'D', 'g', ';', 'n', 'u', '{', 'ð', 'C', 'b', '"', 'i', 'F',
            '7', '5', 's', 'z', 'X', 'm', 'U', '&', '.', 'J', '!', '`', '£',
            'B', '3', 'E', 'p', 'w', 'T', '0', 'a', '9', '[', 'Æ', 'S', 'N',
            'Þ', 'y', 'Q', 'h', 'Y', 'e', '?', 'H', 'W', 'Ý', ']', 'd', 'v',
            'O', 'x', ',']

CASE_I_ALPHABET = [' ', 'ę', '/', '2', '1', 'o', 'j', ')',
            't', ':', 'ꝥ', 'f', '6', '-', '=', '8', 'c',
            'k', 'l', 'þ', 'q', '4', '~', '(', 'ȝ', 'æ', 'r',
            "'", 'g', ';', 'n', 'u', '{', 'ð', 'b', '"', 'i',
            '7', '5', 's', 'z', 'm', '&', '.', '!', '`', '£',
            '3', 'p', 'w', '0', 'a', '9', '[',
            'y', 'h', 'e', '?', 'ý', ']', 'd', 'v',
            'x', ',']

DATA_DIR = "./cached_data/"

CORPUS_DIR = "/corpora/ICAME/texts/helsinki/"

TRAIN = ['coalex', 'coandrea', 'coapollo', 'cobede', 'cobenrul', 'cobeowul', 'coblick', 'coboeth',
         'cobrunan', 'cobyrhtf', 'cochad', 'cochrist', 'cochroa2', 'cochroa3', 'cochroe4', 'cocura',
         'cocynew', 'codicts', 'codocu1', 'codocu2', 'codocu3', 'codocu4', 'codream', 'codurham',
         'coepihom', 'coexeter', 'coexodus', 'cogenesi', 'cogregd3', 'cogregd4', 'coinspol',
         'cokentis', 'colacnu', 'colaece', 'colaw2', 'colaw3', 'colaw4', 'coleofri', 'colindis',
         'comarga', 'comartyr', 'comarvel', 'cometboe', 'cometrps', 'conorthu', 'coohtwu2',
         'coohtwu3', 'coorosiu', 'cootest', 'cmaelr3', 'cmaelr4', 'cmalisau', 'cmancre', 'cmastro', 
         'cmbrut3', 'cmcapchr', 'cmcapser', 'cmcaxpro', 'cmchauli', 'cmcloud', 'cmctpros',
         'cmctvers', 'cmcursor', 'cmdigby', 'cmdocu2', 'cmdocu3', 'cmdocu4', 'cmearlps', 'cmedmund',
         'cmequato', 'cmfitzja', 'cmfoxwo', 'cmgaytry', 'cmgower', 'cmgregor', 'cmhali', 'cmhansyn',
         'cmhavelo', 'cmhilton', 'cmhorn', 'cmhorses', 'cminnoce', 'cmjulia', 'cmjulnor', 'cmkathe',
         'cmkempe', 'cmkentse', 'cmlambet', 'cmlaw', 'cmludus', 'cmmalory', 'cmmandev', 'cmmankin',
         'cmmarga', 'cmmetham', 'cmmirk', 'cmmoon', 'cmnorhom', 'cmntest', 'cmoffic3', 'cmoffic4',
         'cmorm', 'cmotest', 'cmperidi', 'cmpeterb', 'cmphlebo', 'cmpoemh', 'cmpoems', 'cmpolych',
         'cmprick', 'cmpriv', 'cmpurvey', 'cmreynar', 'cmreynes', 'ceauto1', 'ceauto2', 'ceauto3',
         'cediar2a', 'cediar2b', 'cediar3a', 'cediar3b', 'ceeduc1a', 'ceeduc1b', 'ceeduc2a',
         'ceeduc2b', 'ceeduc3a', 'ceeduc3b', 'cefict1a', 'cefict1b', 'cefict2a', 'cefict2b',
         'cefict3a', 'cefict3b', 'cehand1a', 'cehand1b', 'cehand2a', 'cehand2b', 'cehand3a',
         'cehand3b', 'cehist1a', 'cehist1b', 'cehist2a', 'cehist2b', 'cehist3a', 'cehist3b',
         'celaw1', 'celaw2', 'celaw3', 'centest1', 'centest2', 'ceoffic1', 'ceoffic2', 'ceoffic3',
         'ceotest1', 'ceotest2', 'ceplay1a', 'ceplay1b', 'ceplay2a', 'ceplay2b', 'ceplay3a',
         'ceplay3b', 'cepriv1', 'cepriv2', 'cepriv3', 'cescie1a', 'cescie1b', 'cescie2a', 'cescie2b',
         'cescie3a', 'cescie3b']

DEVTEST = ['coadrian', 'coaelet3', 'coaelet4', 'coaelhom', 'coaelive', 'coaepref', 'coaepreg',
           'cmayenbi', 'cmbenrul', 'cmbestia', 'cmbevis', 'cmbodley', 'cmboeth', 'cmbrut1',
           'cebio1', 'cebio2', 'cebio3', 'ceboeth1', 'ceboeth2', 'ceboeth3', 'cediar1a', 'cediar1b']

TEST = ['coparips', 'cophoeni', 'coprefcp', 'coprefso', 'coprogno', 'coquadru', 'coriddle',
        'corushw', 'cosolomo', 'cotempo', 'covesps', 'cowsgosp', 'cowulf3', 'cowulf4', 'cmrobglo',
        'cmrollbe', 'cmrollps', 'cmrolltr', 'cmrood', 'cmroyal', 'cmsawles', 'cmseleg', 'cmsiege',
        'cmsirith', 'cmthorn', 'cmthrush', 'cmtownel', 'cmtrinit', 'cmveshom', 'cmvices1',
        'cmvices4', 'cmwycser', 'cmyork', 'ceserm1a', 'ceserm1b', 'ceserm2a', 'ceserm2b', 'ceserm3a',
        'ceserm3b', 'cetrav1a', 'cetrav1b', 'cetrav2a', 'cetrav2b', 'cetrav3a', 'cetrav3b', 
        'cetri1', 'cetri2a', 'cetri2b', 'cetri3a', 'cetri3b']
