<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/sequence.py#L16)</span>
### TimeseriesGenerator

```python
keras.preprocessing.sequence.TimeseriesGenerator(data, targets, length, sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=False, reverse=False, batch_size=128)
```

Utility class for generating batches of temporal data.

This class takes in a sequence of data-points gathered at
equal intervals, along with time series parameters such as
stride, length of history, etc., to produce batches for
training/validation.

__Arguments__

- __data__: Indexable generator (such as list or Numpy array)
    containing consecutive data points (timesteps).
    The data should be at 2D, and axis 0 is expected
    to be the time dimension.
- __targets__: Targets corresponding to timesteps in `data`.
    It should have same length as `data`.
- __length__: Length of the output sequences (in number of timesteps).
- __sampling_rate__: Period between successive individual timesteps
    within sequences. For rate `r`, timesteps
    `data[i]`, `data[i-r]`, ... `data[i - length]`
    are used for create a sample sequence.
- __stride__: Period between successive output sequences.
    For stride `s`, consecutive output samples would
    be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
- __start_index__: Data points earlier than `start_index` will not be used
    in the output sequences. This is useful to reserve part of the
    data for test or validation.
- __end_index__: Data points later than `end_index` will not be used
    in the output sequences. This is useful to reserve part of the
    data for test or validation.
- __shuffle__: Whether to shuffle output samples,
    or instead draw them in chronological order.
- __reverse__: Boolean: if `true`, timesteps in each output sample will be
    in reverse chronological order.
- __batch_size__: Number of timeseries samples in each batch
    (except maybe the last one).

__Returns__

A [Sequence](/utils/#sequence) instance.

__Examples__


```python
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

data = np.array([[i] for i in range(50)])
targets = np.array([[i] for i in range(50)])

data_gen = TimeseriesGenerator(data, targets,
                               length=10, sampling_rate=2,
                               batch_size=2)
assert len(data_gen) == 20

batch_0 = data_gen[0]
x, y = batch_0
assert np.array_equal(x,
                      np.array([[[0], [2], [4], [6], [8]],
                                [[1], [3], [5], [7], [9]]]))
assert np.array_equal(y,
                      np.array([[10], [11]]))
```
    
----

### pad_sequences


```python
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
```


Pads sequences to the same length.

This function transforms a list of
`num_samples` sequences (lists of integers)
into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
`num_timesteps` is either the `maxlen` argument if provided,
or the length of the longest sequence otherwise.

Sequences that are shorter than `num_timesteps`
are padded with `value` at the end.

Sequences longer than `num_timesteps` are truncated
so that they fit the desired length.
The position where padding or truncation happens is determined by
the arguments `padding` and `truncating`, respectively.

Pre-padding is the default.

__Arguments__

- __sequences__: List of lists, where each element is a sequence.
- __maxlen__: Int, maximum length of all sequences.
- __dtype__: Type of the output sequences.
    To pad sequences with variable length strings, you can use `object`.
- __padding__: String, 'pre' or 'post':
    pad either before or after each sequence.
- __truncating__: String, 'pre' or 'post':
    remove values from sequences larger than
    `maxlen`, either at the beginning or at the end of the sequences.
- __value__: Float or String, padding value.

__Returns__

- __x__: Numpy array with shape `(len(sequences), maxlen)`

__Raises__

- __ValueError__: In case of invalid values for `truncating` or `padding`,
    or in case of invalid shape for a `sequences` entry.
    
----

### skipgrams


```python
keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size, window_size=4, negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None)
```


Generates skipgram word pairs.

This function transforms a sequence of word indexes (list of integers)
into tuples of words of the form:

- (word, word in the same window), with label 1 (positive samples).
- (word, random word from the vocabulary), with label 0 (negative samples).

Read more about Skipgram in this gnomic paper by Mikolov et al.:
[Efficient Estimation of Word Representations in
Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

__Arguments__

- __sequence__: A word sequence (sentence), encoded as a list
    of word indices (integers). If using a `sampling_table`,
    word indices are expected to match the rank
    of the words in a reference dataset (e.g. 10 would encode
    the 10-th most frequently occurring token).
    Note that index 0 is expected to be a non-word and will be skipped.
- __vocabulary_size__: Int, maximum possible word index + 1
- __window_size__: Int, size of sampling windows (technically half-window).
    The window of a word `w_i` will be
    `[i - window_size, i + window_size+1]`.
- __negative_samples__: Float >= 0. 0 for no negative (i.e. random) samples.
    1 for same number as positive samples.
- __shuffle__: Whether to shuffle the word couples before returning them.
- __categorical__: bool. if False, labels will be
    integers (eg. `[0, 1, 1 .. ]`),
    if `True`, labels will be categorical, e.g.
    `[[1,0],[0,1],[0,1] .. ]`.
- __sampling_table__: 1D array of size `vocabulary_size` where the entry i
    encodes the probability to sample a word of rank i.
- __seed__: Random seed.

__Returns__

couples, labels: where `couples` are int pairs and
    `labels` are either 0 or 1.

__Note__

By convention, index 0 in the vocabulary is
a non-word and will be skipped.
    
----

### make_sampling_table


```python
keras.preprocessing.sequence.make_sampling_table(size, sampling_factor=1e-05)
```


Generates a word rank-based probabilistic sampling table.

Used for generating the `sampling_table` argument for `skipgrams`.
`sampling_table[i]` is the probability of sampling
the word i-th most common word in a dataset
(more common words should be sampled less frequently, for balance).

The sampling probabilities are generated according
to the sampling distribution used in word2vec:

```
p(word) = (min(1, sqrt(word_frequency / sampling_factor) /
    (word_frequency / sampling_factor)))
```

We assume that the word frequencies follow Zipf's law (s=1) to derive
a numerical approximation of frequency(rank):

`frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`
where `gamma` is the Euler-Mascheroni constant.

__Arguments__

- __size__: Int, number of possible words to sample.
- __sampling_factor__: The sampling factor in the word2vec formula.

__Returns__

A 1D Numpy array of length `size` where the ith entry
is the probability that a word of rank i should be sampled.
    