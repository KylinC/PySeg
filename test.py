import pyseg
from tkinter import *

# # Hidden Markov Model

res1 = pyseg.cut('这个程序不能准确的分割出喜欢，这也就是概率模型的问题所在。')
res2 = pyseg.cut('中国特色社会主义，邓小平改革开放')
res3 = pyseg.cut('发言人强调，该法案涉台内容严重违反一个中国原则和中美三个联合公报规定，严重损害中美关系和台海和平稳定。')
res4 = pyseg.cut('通知强调，各地要做好统筹安排，按照国务院关于保障义务教育教师工资待遇的工作部署，加大工作力度。')

print(res1)