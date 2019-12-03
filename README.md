# Lstar Extraction
Welcome to our public repository, implementing the extraction algorithm from our ICML 2018 paper, [Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples](https://arxiv.org/abs/1711.09576). 

### Google Colaboratory
To use the main notebook here without installing anything, you can go straight to google colaboratory: https://drive.google.com/file/d/1tkJK1rJVEg9e-QcWOxErDb3cQq9UR-yR/view?usp=sharing

### This Repository
Open the `dfa_from_rnn` notebook for a full demonstration and run through of how to use it yourself (we have provided all of the Tomita grammars, but you can also define, train, and extract your own languages!). For the impatient, the `dfa_from_rnn_no_documentation` notebook is exactly like `dfa_from_rnn`, only without all the explanation blocks. And if you want to train and keep track of several RNNs, you can use `dfa_from_rnn_notebook_for_several_rnns`, which is like `dfa_from_rnn_no_documentation` only it keeps all of your RNNs in neat little wrappers with their target languages and then keeps all of those in a list.

### Package Requirements
##### Full Install
Everything here is implemented in Python 3. To use these notebooks, you will also need to install:

>1. [DyNet](http://dynet.readthedocs.io/en/latest/python.html) (for working with our LSTM and GRU networks, which are implemented in DyNet) 
>2. [Graphviz](http://graphviz.readthedocs.io/en/stable/manual.html#installation) (for drawing the extracted DFAs). 
>3. [NumPy and SciPy](https://scipy.org/install.html) (for Scikit-Learn)
>4. [Scikit-Learn](http://scikit-learn.org/stable/install.html) (for the SVM classifier)
>5. [Matplotlib](https://matplotlib.org/users/installing.html) (for plots of our networks' loss during training)
>6. [Jupyter](http://jupyter.readthedocs.io/en/latest/install.html) (for the python notebooks themselves)

If you are on a mac using Homebrew, then NumPy, SciPy, Scikit-Learn, Matplotlib, Graphviz and Jupyter should all hopefully work with `brew install numpy`, `brew install scipy`, etc. 

If you don't have Homebrew, or wherever `brew install` doesn't work, try `pip install` instead. 

For Graphviz you may first need to download and install the package yourself [Graphviz](https://www.graphviz.org/download/), after which you can run `pip install graphviz`. If you're lucky, `brew install graphviz` might take care of all of this for you by itself. On colab, we got Graphviz using `pip install graphviz` and then `apt-get install graphviz`.

DyNet is installed by `pip install dynet` from the command line (for the basic CPU version. For the GPU version, check their [site](http://dynet.readthedocs.io/en/latest/python.html)). 

### Extracting from Existing Networks
You can also apply the code directly to your own networks without most of these packages. The main extraction function is in `Extraction.py` and called `extract`. You can run it on any network that implements the API described in our `dfa_from_rnn` notebook, which is viewable in-browser in git even if you don't have Jupyter, and reiterated here for completeness.
##### Network Extraction API
>1. `classify_word(word)`       returns a True or False classification for a word over the input alphabet
>2. `get_first_RState()`        returns a tuple (v,c) where v is a continuous vector representation of the network's initial state (an RState), and c is a boolean signifying whether it is an accepting state
>3. `get_next_RState(state,char)`    given an RState, returns the next RState the network goes to on input character `char`, in the same format as `get_first_RState` (i.e., a tuple (v,c) of vector + boolean)

##### Partial Install
To run only the extraction code you will only need the NumPy, SciPy, Scikit-Learn, and Graphviz packages. If you want, you can also skip the Graphviz package, at the cost of the ability to visualise your DFAs. Remove the graphviz import from `DFA.py` and set the body of the `draw_nicely` function of the `DFA` class to `pass`. You only need the `DFA`, `Extraction`, `Lstar`, `Helper_Functions`, `Observation_Table`, `Quantisations`, `Teacher`, and `WhiteboxRNNCounterexampleGenerator` modules for extraction.




### Citation
You can cite this work using:

@InProceedings{weiss-goldberg-yahav,  
  title = 	 {Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples},  
  author = 	 {Gail Weiss and Yoav Goldberg and Eran Yahav},  
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},  
  year = 	 {2018},  
  editor = 	 {Jennifer Dy and Andreas Krause},  
  volume = 	 {80},  
  series = 	 {Proceedings of Machine Learning Research},  
  address = 	 {Stockholmsm\\"{a}ssan, Stockholm, Sweden},  
  month = 	 {10--15 Jul},  
  publisher = 	 {PMLR}  
}

