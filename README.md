# PySeg
> 中文分词库, 词性标注库
>
> Chinese Words Segmentation and Tagger Library via Python

[![](https://img.shields.io/badge/python-3.5.7-blue.svg)]()

[![](https://img.shields.io/badge/Torch-1.0-orange)]()



## Usage

- 将项目内pyseg放置于工作目录或者site-packages目录
- import snailseg

```python
import pyseg

# 默认使用隐马尔可夫模型（HMM）训练的2013年人民日报语料库进行分词
words = pyseg.cut('发言人强调，该法案涉台内容严重违反一个中国原则，损害中美关系。')
for w in words:
	  print(w)
    
# 使用二分正向最大匹配模型（BFMM）训练的2013年人民日报语料库进行分词 O(T)= nlog(n)
words = pyseg.cut_fmm('中国特色社会主义，邓小平改革开放。')
for w in words:
	  print(w)
    
# 将语料置于pyseg/data内,即可通过文件名进行HMM和BFMM模型的训练参数重载
pyseg.load_corpus("trainCorpus.txt_utf8")

# 12.15日加入词性标注功能，但是由于标注语料缺乏，所以只限于适用！！
# 应用HMM模型进行预处理分词，之后使用LSTM进行词性标注
res5,res5_tag = pyseg.cut_mark('江泽民主席来到北京负责燃料工业部的指导建设工作。')
for tag in res5_tag:
    print(tag)
  
```



## Algorithm

- 默认cut方法使用2013年人民日报语料库训练的隐马尔可夫模型（HMM）进行分词，具体实现参考MCMC模型以及Viterbi算法（DP）
- cut_fmm算法使用2013年人民日报语料库训练的二分正向最大匹配模型（BFMM）进行分词，具体实现在FMM上进行改进，将逐次缩位改为二分查找最大前缀，具体可以参考博客中分词内容：[https://kylinchen.top](https://kylinchen.top)
- HMM+Embedding+LSTM进行词性标注，训练语料还较少，目标使用1998年人民日报语料库进行训练



## Example

>  以HMM模型为例

- Input

```python
res1 = pyseg.cut('这个程序不能准确的分割出喜欢，这也就是概率模型的问题所在。')
res2 = pyseg.cut('中国特色社会主义，邓小平改革开放')
res3 = pyseg.cut('发言人强调，该法案涉台内容严重违反一个中国原则和中美三个联合公报规定，严重损害中美关系和台海和平稳定。')
res4 = pyseg.cut('通知强调，各地要做好统筹安排，按照国务院关于保障义务教育教师工资待遇的工作部署，加大工作力度。')
res5,tag = pyseg.cut_mark('江泽民主席来到北京负责燃料工业部的指导建设工作。')
```

- Output

```pyhton
这个\程序\不能\准确\的\分割出\喜欢\，\这\也\就\是\概率\模型\的\问题\所在\。\

中国\特色\社会\主义\，\邓小平\改革\开放\

发言人\强调\，\该\法案\涉台\内容\严重\违反\一个\中国\原则\和\中美\三个\联合\公报\规定\，\严重\损害\中美\关系\和\台海\和\平\稳定\。\

通知\强调\，\各地\要\做好\统筹\安排\，\按照\国务院\关于\保障\义务\教育\教师\工资待遇\的\工作\部署\，\加大\工作\力度\。\

江泽民主席\n    来到\v    北京\ns    负责\v    燃料\n    工业部\n    的\uj    指导\n    建设\vn    工作\vn    。\x 
```

