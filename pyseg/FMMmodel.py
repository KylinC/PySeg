import os
import io
import bisect
import pickle
import collections

class RunError(Exception):
    def __init__(self):
        Exception.__init__(self)
    def __str__(self):
        return repr('RunError')

class FMM(object):
    def __init__(self,train_switch=False):
        _localDir=os.path.dirname(__file__)
        self._curpath=os.path.normpath(os.path.join(os.getcwd(),_localDir))
        self.model_file = os.path.join(self._curpath,"data/fmm_model.pkl")
        self.word_list = []
        self.max_length = 0
        if not train_switch:
            try:
                with open(self.model_file, 'rb') as f:
                    self.word_list = pickle.load(f)
            except:
                pass
            num_list = [len(one) for one in self.word_list]
            self.max_length = max(num_list)
            # print(self.max_length)

    def flatten(self,x):
        result = []
        for el in x:
            if isinstance(x, collections.Iterable) and not isinstance(el, str):
                result.extend(self.flatten(el))
            else:
                result.append(el)
        return result

    def train(self, file_name):
        path = os.path.join(self._curpath,"data/"+file_name)
        with io.open(path, encoding='utf8') as f:
            self.word_list = [line.strip().split() for line in f]
            self.word_list = self.flatten(self.word_list)
            self.word_list.sort()
        num_list = [len(one) for one in self.word_list]
        self.max_length = max(num_list)
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.word_list, f)

    def check_prefix(self,text):
        idx = bisect.bisect_right(self.word_list,text)
        if(text.startswith(self.word_list[idx-1])):
            return len(self.word_list[idx-1])
        else:
            return 1

    def cut(self,text):
        begin = 0
        while(begin<len(text)):
            if(begin+self.max_length<=len(text)):
                res_offset = self.check_prefix(text[begin:begin+self.max_length])
                # print(text[begin:begin+res_offset])
                yield text[begin:begin+res_offset]
                begin=begin+res_offset
            else:
                res_offset = self.check_prefix(text[begin:])
                yield text[begin:begin+res_offset]
                begin=begin+res_offset

if __name__ == '__main__':
    fmm = FMM()
    fmm.train("test.utf8")
    text = '在建设有中国特色的社会主义道路上'
    res = fmm.cut(text)
    print(list(res))