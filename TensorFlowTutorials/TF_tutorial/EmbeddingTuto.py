import tensorflow as tf
vocabulary_size = 20

embedding_size = 2
word_embedding = tf.get_variable("word_embeddings",
                                 [vocabulary_size,
                                  embedding_size])

print("\n Word Embedding: {}".format(word_embedding))

word_ids = [1, 5, 7, 4, 10]
embedded_word_ids = tf.nn.embedding_lookup(
        word_embedding, word_ids)
print("\n Embedded Word IDs: {}".format(embedded_word_ids))

