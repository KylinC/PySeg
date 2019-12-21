import os
import io
import pickle

class RunError(Exception):
    def __init__(self):
        Exception.__init__(self)
    def __str__(self):
        return repr('RunError')

class HMM(object):
    def __init__(self,train_switch=False):
        _localDir=os.path.dirname(__file__)
        self._curpath=os.path.normpath(os.path.join(os.getcwd(),_localDir))
        self.model_file = os.path.join(self._curpath,"data/hmm_model.pkl")
        self.state_list = ['B', 'M', 'E', 'S']
        self.trans_p = {}
        self.emit_p = {}
        self.start_p = {}
        self.load_para = False
        if not train_switch:
            try:
                with open(self.model_file, 'rb') as f:
                    self.trans_p = pickle.load(f)
                    self.emit_p = pickle.load(f)
                    self.start_p = pickle.load(f)
                    self.load_para = True
            except:
                pass

    def train(self, file_name):
        path = os.path.join(self._curpath,"data/"+file_name)
        Count_dic = {}
        def init_parameters():
            for state in self.state_list:
                self.trans_p[state] = {s: 0.0 for s in self.state_list}
                self.start_p[state] = 0.0
                self.emit_p[state] = {}
                Count_dic[state] = 0

        def makeLabel(text):
            out_text = []
            if len(text) == 1:
                out_text.append('S')
            else:
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']
            return out_text

        init_parameters()
        line_num = -1
        words = set()
        with io.open(path, encoding='utf8') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                word_list = [i for i in line if i != ' ']
                words |= set(word_list)
                linelist = line.split()
                line_state = []
                for w in linelist:
                    line_state.extend(makeLabel(w))

                assert(len(word_list) == len(line_state))

                for k, v in enumerate(line_state):
                    Count_dic[v] += 1
                    if k == 0:
                        self.start_p[v] += 1
                    else:
                        self.trans_p[line_state[k-1]][v] += 1
                        self.emit_p[line_state[k]][word_list[k]] = \
                            self.emit_p[line_state[k]].get(word_list[k], 0) + 1.0
        self.start_p = {k: v * 1.0 / line_num for k, v in self.start_p.items()}
        self.trans_p = {k: {k1: v1 / (Count_dic[k]+1) for k1, v1 in v.items()}
                      for k, v in self.trans_p.items()}
        self.emit_p = {k: {k1: (v1 + 1) / (Count_dic[k]+1) for k1, v1 in v.items()}
                      for k, v in self.emit_p.items()}

        with open(self.model_file, 'wb') as f:
            pickle.dump(self.trans_p, f)
            pickle.dump(self.emit_p, f)
            pickle.dump(self.start_p, f)

        return self


    def viterbi(self, text, states, start_p, trans_p, emit_p):
        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]
        for t in range(1, len(text)):
            V.append({})
            newpath = {}

            neverSeen = text[t] not in emit_p['S'].keys() and \
                text[t] not in emit_p['M'].keys() and \
                text[t] not in emit_p['E'].keys() and \
                text[t] not in emit_p['B'].keys()
            
            for y in states:
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0
                (prob, state) = max(
                    [(V[t-1][y0] * trans_p[y0].get(y, 0) * emitP, y0) for y0 in states if V[t-1][y0] > 0]
                )
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath
        if (emit_p['M'].get(text[-1], 0)):
            (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E', 'M')])
        else:
            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])
        return (prob, path[state])

    def cut(self, text):
        prob, pos_list = self.viterbi(text, self.state_list, self.start_p, self.trans_p, self.emit_p)
        begin, next = 0, 0
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin: i+1]
                next = i + 1
            elif pos == 'S':
                yield char
                next = i + 1
        if next < len(text):
            yield text[next:]


if __name__ == '__main__':
    hmm = HMM()
    # hmm.train('trainCorpus.txt_utf8')
    # hmm.load_model(True)
    text = '这个程序不能准确的分割出喜欢，这也就是概率模型的问题所在。'
    res = hmm.cut(text)
    print(list(res))