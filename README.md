# Low Resource Prediction Using Bert Attention Scores

This library is the code-base that accompanies our in-progress workshop paper.
We demonstrate that the attention distributions of trained BERT models
provide strong enough signal to be used as the input themselves to downstream
shallow neural networks. This approach enables us to limit the amount of data we require to train classification models.

To Do:

* Write script to cross-validate results, instead of running inference on a test dataset.
* Implement baselines/experiments listed below
* Different classification models (e.g. transformer)
* Understand BERT attention (keeping 12 heads vs 1) 

Experiments to try running
* Compute average attention scores of 6th layer
* Get the weak labels
  * POS, Glove, Marta (!)
* Test out different averaging techniques
* Move away from windowing --> how do we want to get the attention scores in this case?
=======
Experiments to try running:
https://docs.google.com/document/d/1_CX-DbHrUQOEmhAmAFVW5Et83q-f0s-6CHGFHdrvda0/edit?usp=sharing
