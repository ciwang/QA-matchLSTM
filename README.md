# Stanford CS224N Final Project

## Reading Comprehension with Deep Learning

A neural network architecture for reading comprehension using the [Stanford Question Answering Dataset (SQuAD)](https://arxiv.org/abs/1606.05250). Given 100K question-answer pairs, the task is defined as predicting an answer span within a given context paragraph.

Question:
> Why was Tesla returned to Gospic?

Paragraph: 
> On 24 March 1879, Tesla was returned to Gospic under police guard for `not having a residence permit`. On 17 April 1879, Milutin Tesla died at the age of 60 after contracting an unspecified illness (although some sources say that he died of a stroke). During that year, Tesla taught a large class of students in his old school, Higher Real Gymnasium, in Gospic.

Answer:
> not having a residence permit

----
## Results

Baseline (validation set)
F1 24.9
EM 16

Match-LSTM (validation set)
F1 61.0
EM 51

Match-LSTM (dev set)
F1 53.7
EM 40

Match-LSTM (test set)
F1 54.0
EM 41

----
## Authors
* Cindy Wang
* Annie Hu
* Brandon Yang

----
## Thanks
* [Machine Comprehension Using Match-LSTM and Answer Pointer (Wang, Jiang)](https://arxiv.org/abs/1608.07905)
* [Dynamic Coattention Networks For Question Answering (Xiong, Zhong, Socher)](https://arxiv.org/abs/1611.01604)
