# ACTRNN  


Stefan Heinrich et al. 2018  
Adaptive and Variational Continuous Time Recurrent Neural Networks  
heinrich@informatik.uni-hamburg.de  


Reference:  
@inproceedings{Heinrich2018AVCTRNN,  
	author       = {Heinrich, Stefan and Alpay, Tayfun and Wermter, Stefan},  
	title        = {Adaptive and Variational Continuous Time Recurrent Neural Networks},  
	booktitle    = {Joint IEEE International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob)},  
	year         = {2018}  
}  


Usage:  
Use the various CTRNN versions just like the other RNN models. For documentation on variable connotation see xctrnn_cell.py.  


Example:  
...  
import tensorflow as tf
import models.tensorflow_extend as tf_extend  
from models.tensorflow_extend import xctrnn_cell, ctrnn_cell, cwrnn_cell  
...  
num_hidden = 60  
num_hidden_v = [30, 20, 10]  
tau_hidden_v = [1, 8, 64]  
connectivity = 'dense'  
initializer_w_tau = tf.glorot_normal_initializer()  
...  
hid_rnn_cell = tf_contrib.BasicLSTMCell(num_hidden)  
\# ->  
hid_rnn_cell = tf_extend.xctrnn_cell.ACTRNNCell(  
                num_hidden_v, tau_hidden_v,  
                connectivity=connectivity,  
                initializer=initializer, initializer_w_tau=initializer_w_tau)  
...  
hiddens, states = tf.nn.dynamic_rnn(hid_rnn_cell, x, dtype=tf.float32)  


Update (Jan 2019):
The CTRNN versions can now be used in tensorflow.keras models as well (tensorflow version 1.12).

Example:  
...  
import tensorflow as tf
from tensorflow import keras  
import models.keras_extend.xctrnn 
...  
num_hidden = 60  
num_hidden_v = [30, 20, 10]  
tau_hidden_v = [1, 8, 64]  
...
model = keras.Sequential()
...  
model.add(keras.layers.SimpleRNN(num_hidden)) 
\# ->  
model.add(xctrnn.ACTRNN(num_hidden_v, tau_vec=tau_hidden_v)) 
...  

