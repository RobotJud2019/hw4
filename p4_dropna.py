# This is part of the submission to the problemset 4 in the course building a robot judge in spring 2019

# this script is used to generate a pickle file containing 1000 of the 5000+ cases
# in the cases corpus. the 1000 were randomly selected without replacement. 
# Cases with NA in the metadata (judgeid, party of judge, log_citation, ) are skipped

import pickle
import numpy as np
import csv
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from txt_utils import *

from datetime import datetime
from datetime import timedelta

# td takes care of the time difference between my laptop and realtime
td = timedelta (days = 0, hours =  0, minutes = 0 )
Anfang = datetime.now()
Start = Anfang + td


#############

from datetime import datetime

from os import listdir
from os.path import isfile, join

import spacy	#  initializing data structures used later
from spacy.lang.en import English
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

nlp2 = spacy.load('en_core_web_sm')

fpath = "/home/xhta/Robot/cases/"
ldir = listdir(fpath)

p4read_md = pd.read_csv("case_metadata.csv")
p4read_md = p4read_md.dropna()  #  
#p4read_md.to_pickle("p4_md_1k_dropna.20190611_1743.pkl")

choose_howmany = 1000

selected = np.random.choice(np.arange(len(p4read_md)), choose_howmany, replace = False)
p4_md = p4read_md.iloc[selected]

p4_md = p4_md.set_index("caseid")
p4_df = p4_md
i = 0
for fname in ldir:
    lae = len(fname)
    cname = fname[5:(lae-4)]
    year = fname[0:4]
    if ( not (cname in p4_df.index)): continue
    fna2 = join(fpath, year + '_' + cname + '.txt')
    rawtext =   open(fna2).read()
    doc = nlp(rawtext)
    p4_df.at[cname, 'doc'] =   rawtext
#
    if ( i% 119 ==0):
        print("i:" + str(i) + " " + datetime.now().strftime('%Y%m%d_%H:%M:%S'))	 # serves as a "progress bar" when reading in a large # of docs
    p4_df.at[cname, 'jahr'] =  year
    i = i + 1
#
    p4_df.at[cname, 'nlets'] =  len(rawtext)
    sentences = [sent.string.strip() for sent in doc.sents]
    p4_df.at[cname, 'nsents'] = len(sentences)
    p4_df.at[cname, 'nwords'] = len([token for token in doc if not token.is_punct])
    doc2 = nlp2(rawtext)
    p4_df.at[cname, 'nnouns'] = len([w for w in list(doc2) if w.tag_.startswith('N')])	# PTB convention
    p4_df.at[cname, 'nverbs'] = len([w for w in list(doc2) if w.tag_.startswith('V')])
    p4_df.at[cname, 'nadjes'] = len([w for w in list(doc2) if w.tag_.startswith('J')])

je = datetime.now() + td
pkl_fname = 'p4_df_1k.' + je.strftime('%Y%m%d_%H%M%S' + ".pkl")
p4_df.to_pickle(pkl_fname)

