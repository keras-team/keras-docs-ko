
### Text Preprocessing

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/text.py#L138)</span>
### Tokenizer

```python
keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
```

Text tokenization utility class.

This class allows to vectorize a text corpus, by turning each
text into either a sequence of integers (each integer being the index
of a token in a dictionary) or into a vector where the coefficient
for each token could be binary, based on word count, based on tf-idf...

__Arguments__

- __num_words__: the maximum number of words to keep, based
    on word frequency. Only the most common `num_words` words will
    be kept.
- __filters__: a string where each element is a character that will be
    filtered from the texts. The default is all punctuation, plus
    tabs and line breaks, minus the `'` character.
- __lower__: boolean. Whether to convert the texts to lowercase.
- __split__: str. Separator for word splitting.
- __char_level__: if True, every character will be treated as a token.
- __oov_token__: if given, it will be added to word_index and used to
    replace out-of-vocabulary words during text_to_sequence calls

By default, all punctuation is removed, turning the texts into
space-separated sequences of words
(words maybe include the `'` character). These sequences are then
split into lists of tokens. They will then be indexed or vectorized.

`0` is a reserved index that won't be assigned to any word.

----

### hashing_trick


```python
keras.preprocessing.text.hashing_trick(text, n, hash_function=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ')
```


Converts a text to a sequence of indexes in a fixed-size hashing space.

__Arguments__

- __text__: Input text (string).
- __n__: Dimension of the hashing space.
- __hash_function__: defaults to python `hash` function, can be 'md5' or
    any function that takes in input a string and returns a int.
    Note that 'hash' is not a stable hashing function, so
    it is not consistent across different runs, while 'md5'
    is a stable hashing function.
- __filters__: list (or concatenation) of characters to filter out, such as
    punctuation. Default: ``!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	

``,
    includes basic punctuation, tabs, and newlines.

- __lower__: boolean. Whether to set the text to lowercase.
- __split__: str. Separator for word splitting.

__Returns__

A list of integer word indices (unicity non-guaranteed).

`0` is a reserved index that won't be assigned to any word.

Two or more words may be assigned to the same index, due to possible
collisions by the hashing function.
The [probability](
https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
of a collision is in relation to the dimension of the hashing space and
the number of distinct objects.

----

### one_hot


```python
keras.preprocessing.text.one_hot(text, n, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ')
```


One-hot encodes a text into a list of word indexes of size n.

This is a wrapper to the `hashing_trick` function using `hash` as the
hashing function; unicity of word to index mapping non-guaranteed.

__Arguments__

- __text__: Input text (string).
- __n__: int. Size of vocabulary.
- __filters__: list (or concatenation) of characters to filter out, such as
    punctuation. Default: ``!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	

``,
    includes basic punctuation, tabs, and newlines.

- __lower__: boolean. Whether to set the text to lowercase.
- __split__: str. Separator for word splitting.

__Returns__

List of integers in [1, n]. Each integer encodes a word
(unicity non-guaranteed).
    
----

### text_to_word_sequence


```python
keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	
', lower=True, split=' ')
```


Converts a text to a sequence of words (or tokens).

__Arguments__

- __text__: Input text (string).
- __filters__: list (or concatenation) of characters to filter out, such as
    punctuation. Default: ``!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	

``,
    includes basic punctuation, tabs, and newlines.

- __lower__: boolean. Whether to convert the input to lowercase.
- __split__: str. Separator for word splitting.

__Returns__

A list of words (or tokens).
    
