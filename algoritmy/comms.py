import json
import pandas as pd
import collections
import numpy as np
import tensorflow as tf
import math 


def load_jsons(path = "products-sample.json"):
    """
    Loads json database into a list of dictionaries 
    
    Arguments:
    path -- path to file containing the jsons
    
    Return:
    data -- list, where each entry is a dictionary corresponding
            to one json 
    """
    file = open(path)
    data = []
    for line in file:
        data.append(json.loads(line))
        
    return data 


def user_item_dataframe(data): 
    """
    Creates pandas DataFrame 
    from load_jsons output
    
    Arguments:
    data -- list of dictionaries with "user_id" and "product_id"
    
    Return: 
    df -- pd.Dataframe with columns = ["context", "word"] = ["user_id", "product_id"]
    """
    df = pd.DataFrame([{"context": row["user_id"], "word" : row["product_id"]} for row in data])
   
    return df


def create_contexts(df, remove_one_word_contexts = True):
    """
    Creates list of words in each context for word2vec
    
    Arguments: 
    df -- dataframe with 2 columns ["context", "word"]
    remove_one_word_contexts -- do I keep or remove one word contexts
    
    Return:
    word_bags -- list of lists of words belonging to context in context_ids
                 for example "product_id" lists 
    context_ids -- list of context ids 
                   for example "user_id"   
    """
    grouping = df.groupby("context")
    
    word_bags = grouping.word.apply(list)
    context_ids = list(grouping.groups.keys())
    
    if remove_one_word_contexts: 
        indices_keep = [i for i in range(len(word_bags)) if len(word_bags[i]) > 1]
        word_bags = [word_bags[i] for i in indices_keep]
        context_ids = [context_ids[i] for i in indices_keep]
    
    return word_bags, context_ids


def create_dictionary(word_bags, word_count_option = "A"):
    """
    Creates dictionary of most frequent words and its reverse dictionary 
    
    Arguments:
    word_bags -- list of list of words (result of create_contexts)
    word_count_option -- how to count word 
                         A -- by number of users that viewed it (occurence in concatenated word_bags) 
                         B -- by occurence in concatenated word-context pairs 
                         
    Return: 
    dictionary -- dictionary of "word" : how frequent it is (1 = most frequent, 2 = secont most)
                  example: {"the" : 1, "a" : 2, ... }
    reverse_dictionary -- how frequent : "word" 
    """
    word_context_pairs = [(i, j) for wb in word_bags for i in wb for j in wb if i != j] 
    
    if word_count_option == "A": 
        text_corpus = [word for wb in word_bags for word in wb]
    elif word_count_option == "B":
        text_corpus = [pair[0] for pair in word_context_pairs]
        
    words_by_count = [pair[0] for pair in collections.Counter(text_corpus).most_common()]
    dictionary = dict(zip(words_by_count, range(len(words_by_count))))
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    return dictionary, reversed_dictionary
 
    
def create_genbatch_prerequisities(word_bags, dictionary): 
    """
    Create structures for generate_batch function
    
    Arguments:
    word_bags -- list of lists of words 
    dictionary -- create_dictionary output
    
    Return:
    word_bags_ind -- word_bags converted to dictionary indicies (list of lists)
    text_corpus_ind -- word_bags_ind concatenated to one list
    pairs_id -- pairs of all indicies from wb where wb in word_bags_ind, list of pairs of int 
    """
    word_bags_ind = [[dictionary[word] for word in wb] for wb in word_bags]
    text_corpus_ind = [dictionary[word] for wb in word_bags for word in wb]
    pairs_ind = [(dictionary[w1], dictionary[w2]) for wb in word_bags for w1 in wb for w2 in wb if w1 != w2]
    
    return word_bags_ind, text_corpus_ind, pairs_ind 
    
    
def generate_batch(batch_size, genbatch_prereq, alg_variant = "A"):
    """
    Creates word-context batch 
    
    Arguments:
    batch_size -- size of batch
    genbatch_prereq -- result of create_genbatch_prerequisities
    alg_variant -- variant of algorithm for generating batch
                   A - sample uniformly from all (word,context word) pairs
                       in this case words in longer contexts have 
                       higher chance of sampling 
                   B - TODO - select random word from text corpus 
                       and select randomly word from its context
    
    Return:
    batch -- words np.array(int), shape = (batch_size, )
    labels -- contexts np.array(int), shape = (batch_size, 1) 
    """
    word_bags_ind, text_corpus_ind, pairs_ind = genbatch_prereq
    
    if alg_variant == "A":
        pairs_indices = np.random.choice(len(pairs_ind), batch_size, replace = False)
        batch_label_pairs = [pairs_ind[i] for i in pairs_indices]
        batch = np.array([p[0] for p in batch_label_pairs])
        labels = np.array([p[1] for p in batch_label_pairs]).reshape((-1,1))
    elif alg_variant == "B": 
        ## Find a random word in word_indices_text_corpus and generate 
        print("This is not finished -- TODO")
        return None, None
    
    return batch, labels 
    
def create_w2v_tf_model(batch_size, embedding_size, vocabulary_size, num_sampled, learn_rate = 1.0): 
    """
    Creates tensorflow model for word2vec 
    
    Arguments:
    batch_size -- batch_size for training 
    embedding_size -- dimension of word embedding 
    vocabulary size -- words in text corpus
    num_sampled -- number of generated negative examples in nce loss
    
    Return: 
    
    """
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

    # Add variable initializer.
    init = tf.global_variables_initializer()

    return train_inputs, train_labels, embeddings, nce_weights, nce_biases, loss, optimizer, init 


def softmax(x):
    """
    Compute softmax values for each sets of scores in x
    
    Arguments:
    x -- np float array
    
    Return: 
    sotfmax_x -- softmax of x 
    """
    e_x = np.exp(x - np.max(x))
    softmax_x = e_x / e_x.sum()
    return softmax_x
    
    
    
    
    
    