import re
import io
import math
import os,sys
import bisect
import pickle

# ML dependency
import torch

# system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# data path
_localDir=os.path.dirname(__file__)
_curpath=os.path.normpath(os.path.join(os.getcwd(),_localDir))

torch_path=os.path.normpath(os.path.join(os.getcwd(),_localDir,"data/partspeech.pkl"))
dict_path=os.path.normpath(os.path.join(os.getcwd(),_localDir,"data/extra_dict.pkl"))

# Hidden Markov
from HMMmodel import HMM
from FMMmodel import FMM
from LSTMmodel import LSTMTagger

hmm = HMM()
fmm = FMM()

with open(dict_path, 'rb') as inp:
    word_to_ix = pickle.load(inp)
    tag_to_ix = pickle.load(inp)
    ix_to_tag = pickle.load(inp)

def cut(text):
    blocks = re.split("([^\u4E00-\u9FA5]+)",text)
    result = []
    for block in blocks:
        if re.match("[\u4E00-\u9FA5]+",block):
            result.extend(hmm.cut(block))
        else:
            # tmp = re.split("[^a-zA-Z0-9+#]",block)
            result.extend([x for x in block if x.strip()!=""])
    return result

def load_corpus(file_name):
    hmm.train(file_name)
    fmm.train(file_name)

def cut_fmm(text):
    blocks = re.split("([^\u4E00-\u9FA5]+)",text)
    result = []
    for block in blocks:
        if re.match("[\u4E00-\u9FA5]+",block):
            result.extend(fmm.cut(block))
        else:
            # tmp = re.split("[^a-zA-Z0-9+#]",block)
            result.extend([x for x in block if x.strip()!=""])
    return result

def prepare_sequence(seq,to_ix):
    idxs=[]
    for w in seq:
        if(w in to_ix.keys()):
            idxs.append(to_ix[w])
        else:
            idxs.append(to_ix["None"])
    return torch.tensor(idxs,dtype=torch.long)

def cut_mark(text):
    lstm_model = torch.load(torch_path)
    cut_list = []
    mark_list = []
    sentence = cut(text)
    with torch.no_grad():
        inputs=prepare_sequence(sentence,word_to_ix)
        tag_scores=lstm_model(inputs)
        tem=tag_scores.argmax(dim=1).numpy().tolist()
    for idx in range(len(sentence)):
        cut_list.append(sentence[idx])
        mark_list.append(ix_to_tag[tem[idx]])
    return cut_list,mark_list

if __name__ == '__main__':
    # load_corpus("trainCorpus.txt_utf8")
    res1 = cut('这个程序不能准确的分割出喜欢，这也就是概率模型的问题所在。')
    res2 = cut_fmm('中国特色社会主义，邓小平改革开放')
    print(res1)
    print(res2)
    print(cut_mark('江泽民主席来到北京负责燃料工业部的指导建设工作。'))