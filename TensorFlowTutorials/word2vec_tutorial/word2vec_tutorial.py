



embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embeddingsize],
                            stddev = 1.0 / math.sqrt(embedding_size)))
nce_biases

# Placeholders for inputs
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Compute the NCE loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
        tf.nn.nce_loss(weight = nce_weights,
                       biases = nce_biases,
                       labels = train_labels,
                       inputs = embed,
                       num_sampled = num_sampled,
                       num_classes = vocabulary_size))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1.0).minimize(loss)

for inputs, labels in generate_batch(...):
    feed_dict = {train_inputs: inputs, train_labels: labels}
    -, cur_loss = session.run([optimizer, loss], feed_dict = feed_dict)


