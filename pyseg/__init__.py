import re
import io
import math
import os,sys
import bisect
import pickle

# system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# data path
_localDir=os.path.dirname(__file__)
_curpath=os.path.normpath(os.path.join(os.getcwd(),_localDir))

# Hidden Markov
from HMMmodel import HMM
from FMMmodel import FMM

hmm = HMM()
fmm = FMM()

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

if __name__ == '__main__':
    # load_corpus("trainCorpus.txt_utf8")
    res1 = cut('这个程序不能准确的分割出喜欢，这也就是概率模型的问题所在。')
    res2 = cut_fmm('中国特色社会主义，邓小平改革开放')
    print(res1)
    print(res2)