Q1: Vision Transformer on CIFAR-10

 Introduction:
 
For this task, I built a Vision Transformer (ViT) model completely from scratch using PyTorch and trained it on the CIFAR-10 dataset. The dataset contains 60,000 small color images of size 32×32 pixels in 10 categories such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The objective was to train my custom built Vision Transformer to classify these images accurately.

How the Model Works:

Unlike convolutional neural networks, the Vision Transformer processes images as sequences of patches. Each image is split into small patches, and each patch is given a positional embedding so the model knows where it belongs in the image. I also added a special CLS token that acts as the overall summary of the image. These patch embeddings and the CLS token are passed through Transformer encoder layers that use self-attention to capture relationships between patches. Finally, the CLS token is used by a linear layer to predict the image’s class.

Implementation:

I implemented the model entirely in PyTorch without using external libraries like timm. The training was carried out in Google Colab with GPU enabled. The notebook file q1.ipynb contains all the code. When executed, the notebook automatically downloads the CIFAR-10 dataset, trains three different Vision Transformer models (Small, Medium, and Large), and reports the best accuracy for each.

How to Run the File:

1.	Open the notebook file q1.ipynb in Google Colab.
2.	Go to Runtime → Change runtime type → GPU to enable GPU acceleration.
3.	Run all the cells one by one from top to bottom.
4.	The CIFAR-10 dataset will be downloaded automatically.
5.	Training will start for three models: Small, Medium, and Large.
6.	At the end, the notebook will display the best accuracy achieved by each model and save the trained weights.

Results:

The Small Vision Transformer, trained for 15 epochs, achieved a best accuracy of 71.82% in about 6 minutes. The Medium model, trained for 20 epochs, achieved 79.14% in around 15 minutes. The Large model, trained for 25 epochs, achieved 79.65% but required about 45 minutes. These results show that the Medium model provides the best balance of accuracy and training efficiency.


Observations:

From my experiments, I found that increasing the model size from Small to Medium gave a significant accuracy improvement of around 7%. However, moving from Medium to Large only improved accuracy slightly, while the training time increased threefold. This proves that for smaller datasets like CIFAR-10, scaling up the model size is not always effective unless we also train for more epochs or use stronger data augmentation methods such as Mixup, CutMix, or RandAugment.

 Visualization:
 
I created a bar chart that compares the best accuracies of the Small, Medium, and Large models. The chart clearly shows that accuracy improves when moving from Small to Medium but then levels off when moving to Large, reinforcing the conclusion that the Medium model is the most efficient for this dataset.

Conclusion:

I successfully built and trained a Vision Transformer from scratch on the CIFAR-10 dataset. My experiments showed that while larger models can sometimes perform better, the gains are not always proportional to the increased training time. On CIFAR-10, the Medium Vision Transformer was the most practical and effective choice, balancing both accuracy and efficiency.






Q2: Sentiment Analysis on IMDB with LSTM

Introduction:

For this task, I built a sentiment classification model using an LSTM (Long Short-Term Memory) network in PyTorch. The dataset used was the IMDB movie review dataset, which contains 50,000 reviews labeled as either positive or negative. The objective was to train the model to correctly classify whether a given review expressed positive or negative sentiment.


How the Model Works:

The model converts each review into a sequence of tokens (words). These tokens are mapped into embeddings that represent each word as a dense vector. The embeddings are then passed through a bidirectional LSTM, which processes the sequence from both directions to capture contextual meaning from the beginning and the end of the review. The final hidden states from both directions are concatenated and fed into a fully connected layer to make the prediction.


Implementation:

I used the Hugging Face datasets library to load the IMDB dataset and wrote a custom tokenizer and vocabulary builder without relying on torchtext. Each review was converted into a sequence of indices and padded to handle varying lengths. The model architecture included an embedding layer, a bidirectional LSTM, and a linear output layer. The notebook file q2.ipynb contains the complete implementation.


How to Run the File:

1.	Open the notebook file q2.ipynb in Google Colab.
2.	Go to Runtime → Change runtime type → GPU to enable GPU acceleration.
3.	Run all cells in sequence from top to bottom.
4.	The IMDB dataset will be downloaded automatically via Hugging Face.
5.	Training will begin, and after each epoch, the notebook will print training and validation losses along with accuracy.
6.	At the end, the notebook will display the best validation accuracy and plot a graph showing accuracy trends across epochs.

Results:

The LSTM model trained for 5 epochs achieved a best accuracy of around 85% on the test set. With additional training (10–15 epochs), the model can reach 88–89% accuracy. The training process is efficient, and the results are strong compared to baseline models for sentiment classification.

Observations:

The model quickly learned to distinguish positive and negative reviews within the first few epochs. Accuracy improved steadily with each epoch, but improvements slowed after around 10 epochs. This indicates that additional performance gains would likely require regularization (e.g., dropout) or more advanced architectures (such as Transformers).

Visualization:

I plotted a line graph comparing the training and validation accuracy across epochs. The graph shows that training and validation accuracy follow similar trends, with validation accuracy peaking around 85% after 5 epochs. This visualization confirms that the model generalizes well without severe overfitting.

Conclusion;

I successfully built and trained an LSTM-based sentiment analysis model from scratch using the IMDB dataset. The experiments showed that even with simple preprocessing and a standard LSTM architecture, we can achieve strong performance on sentiment classification tasks.

