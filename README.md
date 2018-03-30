# Text Classification with CNN and RNN

Using CNN and RNN to classify chinese news

CNNfor sentenve classfication: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

char-CNN：[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)

(https://www.tensorflow.org/api_docs/python/tf/nn/conv1d)

## environment

- Python 2/3 
- TensorFlow 1.3 and above
- numpy
- scikit-learn
- scipy

## database

download from http://thuctc.thunlp.org/

This data set uses 10 difference types and each type has 6500 sentences.

Following types：

```
sports, finance, estate, home design, education, science and technology, mode, politics, games, and entertainment

```

https://drive.google.com/open?id=14tIxvjLI-PrGZHmqDm5dIDm1km2BrFXC

data:

- training set: 5000*10
- validation set: 500*10
- testing set: 1000*10

`helper`has two scripts。Among all，`copy_data.sh`is used to extract 6500sentences from data base，`cnews_group.py`is for merging multiple files into one file。After running those two scripts, we can get

- cnews.train.txt: (50000)
- cnews.val.txt: (5000)
- cnews.test.txt: (10000)

## preprocessing

`data/cnews_loader.py` preprocesses the original data。


After preprocessing：

| Data | Shape | Data | Shape |
| :---------- | :---------- | :---------- | :---------- |
| x_train | [50000, 600] | y_train | [50000, 10] |
| x_val | [5000, 600] | y_val | [5000, 10] |
| x_test | [10000, 600] | y_test | [10000, 10] |

## CNN

### configuration

CNNconfiguration is in `cnn_model.py`.

```python
class TCNNConfig(object):
    """CNN configurations"""

    embedding_dim = 64      # 
    seq_length = 600        # 
    num_classes = 10        # 
    num_filters = 128        # 
    kernel_size = 5         # 
    vocab_size = 5000       # 

    hidden_dim = 128        # fully-connected layer

    dropout_keep_prob = 0.5 # dropout for overfitting
    learning_rate = 1e-3    # 

    batch_size = 64         # 
    num_epochs = 10         # 

    print_per_batch = 100    # 
    save_per_batch = 10      # 
```

### CNNmodel

`cnn_model.py`



![images/cnn_architecture](images/cnn_architecture.png)

### training and validation

run `python run_cnn.py train` to start training process。



```
Configuring CNN model...
Configuring TensorBoard and Saver...
Loading training and validation data...
Time usage: 0:00:14
Training and evaluating...
Epoch: 1
Iter:      0, Train Loss:    2.3, Train Acc:  10.94%, Val Loss:    2.3, Val Acc:   8.92%, Time: 0:00:01 *
Iter:    100, Train Loss:   0.88, Train Acc:  73.44%, Val Loss:    1.2, Val Acc:  68.46%, Time: 0:00:04 *
Iter:    200, Train Loss:   0.38, Train Acc:  92.19%, Val Loss:   0.75, Val Acc:  77.32%, Time: 0:00:07 *
Iter:    300, Train Loss:   0.22, Train Acc:  92.19%, Val Loss:   0.46, Val Acc:  87.08%, Time: 0:00:09 *
Iter:    400, Train Loss:   0.24, Train Acc:  90.62%, Val Loss:    0.4, Val Acc:  88.62%, Time: 0:00:12 *
Iter:    500, Train Loss:   0.16, Train Acc:  96.88%, Val Loss:   0.36, Val Acc:  90.38%, Time: 0:00:15 *
Iter:    600, Train Loss:  0.084, Train Acc:  96.88%, Val Loss:   0.35, Val Acc:  91.36%, Time: 0:00:17 *
Iter:    700, Train Loss:   0.21, Train Acc:  93.75%, Val Loss:   0.26, Val Acc:  92.58%, Time: 0:00:20 *
Epoch: 2
Iter:    800, Train Loss:   0.07, Train Acc:  98.44%, Val Loss:   0.24, Val Acc:  94.12%, Time: 0:00:23 *
Iter:    900, Train Loss:  0.092, Train Acc:  96.88%, Val Loss:   0.27, Val Acc:  92.86%, Time: 0:00:25
Iter:   1000, Train Loss:   0.17, Train Acc:  95.31%, Val Loss:   0.28, Val Acc:  92.82%, Time: 0:00:28
Iter:   1100, Train Loss:    0.2, Train Acc:  93.75%, Val Loss:   0.23, Val Acc:  93.26%, Time: 0:00:31
Iter:   1200, Train Loss:  0.081, Train Acc:  98.44%, Val Loss:   0.25, Val Acc:  92.96%, Time: 0:00:33
Iter:   1300, Train Loss:  0.052, Train Acc: 100.00%, Val Loss:   0.24, Val Acc:  93.58%, Time: 0:00:36
Iter:   1400, Train Loss:    0.1, Train Acc:  95.31%, Val Loss:   0.22, Val Acc:  94.12%, Time: 0:00:39
Iter:   1500, Train Loss:   0.12, Train Acc:  98.44%, Val Loss:   0.23, Val Acc:  93.58%, Time: 0:00:41
Epoch: 3
Iter:   1600, Train Loss:    0.1, Train Acc:  96.88%, Val Loss:   0.26, Val Acc:  92.34%, Time: 0:00:44
Iter:   1700, Train Loss:  0.018, Train Acc: 100.00%, Val Loss:   0.22, Val Acc:  93.46%, Time: 0:00:47
Iter:   1800, Train Loss:  0.036, Train Acc: 100.00%, Val Loss:   0.28, Val Acc:  92.72%, Time: 0:00:50
No optimization for a long time, auto-stopping...
```

Best result in validation data base is 94.12%。

Accuracy and loss：

![images](images/acc_loss.png)


### testing

run `python run_cnn.py test` to test

```
Configuring CNN model...
Loading test data...
Testing...
Test Loss:   0.14, Test Acc:  96.04%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

         体育       0.99      0.99      0.99      1000
         财经       0.96      0.99      0.97      1000
         房产       1.00      1.00      1.00      1000
         家居       0.95      0.91      0.93      1000
         教育       0.95      0.89      0.92      1000
         科技       0.94      0.97      0.95      1000
         时尚       0.95      0.97      0.96      1000
         时政       0.94      0.94      0.94      1000
         游戏       0.97      0.96      0.97      1000
         娱乐       0.95      0.98      0.97      1000

avg / total       0.96      0.96      0.96     10000

Confusion Matrix...
[[991   0   0   0   2   1   0   4   1   1]
 [  0 992   0   0   2   1   0   5   0   0]
 [  0   1 996   0   1   1   0   0   0   1]
 [  0  14   0 912   7  15   9  29   3  11]
 [  2   9   0  12 892  22  18  21  10  14]
 [  0   0   0  10   1 968   4   3  12   2]
 [  1   0   0   9   4   4 971   0   2   9]
 [  1  16   0   4  18  12   1 941   1   6]
 [  2   4   1   5   4   5  10   1 962   6]
 [  1   0   1   6   4   3   5   0   1 979]]
Time usage: 0:00:05
```

Accuracy is over96.04%，precision, recalland f1-scoreare all over 0.9.



## RNN

### configuration

RNN is in `rnn_model.py`

```python
class TRNNConfig(object):
    """RNN"""

    
    embedding_dim = 64      # 
    seq_length = 600        # 
    num_classes = 10        # 
    vocab_size = 5000       # 

    num_layers= 2           # hidden layer
    hidden_dim = 128        # neurons in hidden layer
    rnn = 'gru'             # lstm or gru

    dropout_keep_prob = 0.8 # dropout
    learning_rate = 1e-3    # 

    batch_size = 128         # 
    num_epochs = 10          # 

    print_per_batch = 100    # 
    save_per_batch = 10      # 
```

### RNN model

`rnn_model.py`。



![images/rnn_architecture](images/rnn_architecture.png)

### training and test



run `python run_rnn.py train`



```
Configuring RNN model...
Configuring TensorBoard and Saver...
Loading training and validation data...
Time usage: 0:00:14
Training and evaluating...
Epoch: 1
Iter:      0, Train Loss:    2.3, Train Acc:   8.59%, Val Loss:    2.3, Val Acc:  11.96%, Time: 0:00:08 *
Iter:    100, Train Loss:   0.95, Train Acc:  64.06%, Val Loss:    1.3, Val Acc:  53.06%, Time: 0:01:15 *
Iter:    200, Train Loss:   0.61, Train Acc:  79.69%, Val Loss:   0.94, Val Acc:  69.88%, Time: 0:02:22 *
Iter:    300, Train Loss:   0.49, Train Acc:  85.16%, Val Loss:   0.63, Val Acc:  81.44%, Time: 0:03:29 *
Epoch: 2
Iter:    400, Train Loss:   0.23, Train Acc:  92.97%, Val Loss:    0.6, Val Acc:  82.86%, Time: 0:04:36 *
Iter:    500, Train Loss:   0.27, Train Acc:  92.97%, Val Loss:   0.47, Val Acc:  86.72%, Time: 0:05:43 *
Iter:    600, Train Loss:   0.13, Train Acc:  98.44%, Val Loss:   0.43, Val Acc:  87.46%, Time: 0:06:50 *
Iter:    700, Train Loss:   0.24, Train Acc:  91.41%, Val Loss:   0.46, Val Acc:  87.12%, Time: 0:07:57
Epoch: 3
Iter:    800, Train Loss:   0.11, Train Acc:  96.09%, Val Loss:   0.49, Val Acc:  87.02%, Time: 0:09:03
Iter:    900, Train Loss:   0.15, Train Acc:  96.09%, Val Loss:   0.55, Val Acc:  85.86%, Time: 0:10:10
Iter:   1000, Train Loss:   0.17, Train Acc:  96.09%, Val Loss:   0.43, Val Acc:  89.44%, Time: 0:11:18 *
Iter:   1100, Train Loss:   0.25, Train Acc:  93.75%, Val Loss:   0.42, Val Acc:  88.98%, Time: 0:12:25
Epoch: 4
Iter:   1200, Train Loss:   0.14, Train Acc:  96.09%, Val Loss:   0.39, Val Acc:  89.82%, Time: 0:13:32 *
Iter:   1300, Train Loss:    0.2, Train Acc:  96.09%, Val Loss:   0.43, Val Acc:  88.68%, Time: 0:14:38
Iter:   1400, Train Loss:  0.012, Train Acc: 100.00%, Val Loss:   0.37, Val Acc:  90.58%, Time: 0:15:45 *
Iter:   1500, Train Loss:   0.15, Train Acc:  96.88%, Val Loss:   0.39, Val Acc:  90.58%, Time: 0:16:52
Epoch: 5
Iter:   1600, Train Loss:  0.075, Train Acc:  97.66%, Val Loss:   0.41, Val Acc:  89.90%, Time: 0:17:59
Iter:   1700, Train Loss:  0.042, Train Acc:  98.44%, Val Loss:   0.41, Val Acc:  90.08%, Time: 0:19:06
Iter:   1800, Train Loss:   0.08, Train Acc:  97.66%, Val Loss:   0.38, Val Acc:  91.36%, Time: 0:20:13 *
Iter:   1900, Train Loss:  0.089, Train Acc:  98.44%, Val Loss:   0.39, Val Acc:  90.18%, Time: 0:21:20
Epoch: 6
Iter:   2000, Train Loss:  0.092, Train Acc:  96.88%, Val Loss:   0.36, Val Acc:  91.42%, Time: 0:22:27 *
Iter:   2100, Train Loss:  0.062, Train Acc:  98.44%, Val Loss:   0.39, Val Acc:  90.56%, Time: 0:23:34
Iter:   2200, Train Loss:  0.053, Train Acc:  98.44%, Val Loss:   0.39, Val Acc:  90.02%, Time: 0:24:41
Iter:   2300, Train Loss:   0.12, Train Acc:  96.09%, Val Loss:   0.37, Val Acc:  90.84%, Time: 0:25:48
Epoch: 7
Iter:   2400, Train Loss:  0.014, Train Acc: 100.00%, Val Loss:   0.41, Val Acc:  90.38%, Time: 0:26:55
Iter:   2500, Train Loss:   0.14, Train Acc:  96.88%, Val Loss:   0.37, Val Acc:  91.22%, Time: 0:28:01
Iter:   2600, Train Loss:   0.11, Train Acc:  96.88%, Val Loss:   0.43, Val Acc:  89.76%, Time: 0:29:08
Iter:   2700, Train Loss:  0.089, Train Acc:  97.66%, Val Loss:   0.37, Val Acc:  91.18%, Time: 0:30:15
Epoch: 8
Iter:   2800, Train Loss: 0.0081, Train Acc: 100.00%, Val Loss:   0.44, Val Acc:  90.66%, Time: 0:31:22
Iter:   2900, Train Loss:  0.017, Train Acc: 100.00%, Val Loss:   0.44, Val Acc:  89.62%, Time: 0:32:29
Iter:   3000, Train Loss:  0.061, Train Acc:  96.88%, Val Loss:   0.43, Val Acc:  90.04%, Time: 0:33:36
No optimization for a long time, auto-stopping...
```





![images](images/acc_loss_rnn.png)


### Testing

run `python run_rnn.py test` 

```
Testing...
Test Loss:   0.21, Test Acc:  94.22%
Precision, Recall and F1-Score...
             precision    recall  f1-score   support

         体育       0.99      0.99      0.99      1000
         财经       0.91      0.99      0.95      1000
         房产       1.00      1.00      1.00      1000
         家居       0.97      0.73      0.83      1000
         教育       0.91      0.92      0.91      1000
         科技       0.93      0.96      0.94      1000
         时尚       0.89      0.97      0.93      1000
         时政       0.93      0.93      0.93      1000
         游戏       0.95      0.97      0.96      1000
         娱乐       0.97      0.96      0.97      1000

avg / total       0.94      0.94      0.94     10000

Confusion Matrix...
[[988   0   0   0   4   0   2   0   5   1]
 [  0 990   1   1   1   1   0   6   0   0]
 [  0   2 996   1   1   0   0   0   0   0]
 [  2  71   1 731  51  20  88  28   3   5]
 [  1   3   0   7 918  23   4  31   9   4]
 [  1   3   0   3   0 964   3   5  21   0]
 [  1   0   1   7   1   3 972   0   6   9]
 [  0  16   0   0  22  26   0 931   2   3]
 [  2   3   0   0   2   2  12   0 972   7]
 [  0   3   1   1   7   3  11   5   9 960]]
Time usage: 0:00:33
```


