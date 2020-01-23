# Low Resource Prediction Using Bert Attention Scores

This library is the code-base that accompanies our in-progress workshop paper.
We demonstrate that the attention distributions of trained BERT models
provide strong enough signal to be used as the input themselves to downstream
shallow neural networks. We demonstrate that by using attention distributions,
we limit the amount of data we require to train classification models.

To Do:

* Clean up code base 
    * Move Sabri functions into util file 
    * Ensure that all the data is stored in the correct place  
    * Ensure that all notebooks can be run 
* Write script to cross-validate results, instead of running inference on a test dataset.
* Implement baseline using inference on BERT  
* Understand BERT attention (keeping 12 heads vs 1)  
* Moving away from windowing  
* Future concern: Is it an issue that we are always predicting type of bias - what is we predict no bias also  
