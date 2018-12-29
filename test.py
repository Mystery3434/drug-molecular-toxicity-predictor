import numpy as np
import tensorflow as tf

test_data = np.load('NR-ER-test\\names_onehots.npy').item()


labels = np.genfromtxt('NR-ER-test\\names_labels.csv', delimiter=',', encoding='latin-1')[:,1]
labels = np.reshape(labels, (labels.shape[0], 1))
print(np.sum(labels))

onehots = test_data['onehots']
onehots = np.reshape(onehots, (onehots.shape[0], onehots.shape[1], onehots.shape[2], 1))
names = np.asarray(test_data['names'])
names = np.reshape(names, (names.shape[0], 1))



def print_dims():
    print("Labels Dims:", labels.shape)
    print("One Hots Dims:", onehots.shape)
    print("Names Dims:", names.shape)


print_dims()


def create_placeholders(n_r, n_c):
    X = tf.placeholder(name = "X", dtype='float32', shape = (None, n_r, n_c, 1))
    Y = tf.placeholder(name = "Y", dtype='float32', shape = (None, 2))
    keep_prob_conv = tf.placeholder(tf.float32)
    keep_prob_FC = tf.placeholder(tf.float32)
    return X,Y, keep_prob_conv, keep_prob_FC


def initialize_parameters(beta = 0.0):

    #Regularization
    if beta!=0:
        regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
    else: regularizer=None

    W1 = tf.get_variable("W1", [8, 8, 1, 4], regularizer=regularizer,
                         initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [4, 4, 4, 8], regularizer=regularizer,
                         initializer=tf.contrib.layers.xavier_initializer())

    parameters = {"W1": W1, "W2": W2}

    return parameters, regularizer



def forward_propagation(X, parameters, keep_prob_conv, keep_prob_FC, type = "train", regularizer = None):
    W1 = parameters['W1']
    W2 = parameters['W2']



    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME', name="Z1")
    print(Z1.shape)
    A1 = tf.nn.relu(Z1, name="A1")
    D1 = tf.nn.dropout(A1, keep_prob=keep_prob_conv)

    P1 = tf.nn.max_pool(D1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name="P1")
    print(P1.shape)
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME', name="Z2")

    print(Z2.shape)
    A2 = tf.nn.relu(Z2, name="A2")
    D2 = tf.nn.dropout(A2, keep_prob=keep_prob_conv)

    P2 = tf.nn.max_pool(D2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="P2")
    print(P2.shape)

    P2 = tf.contrib.layers.flatten(P2)
    print(P2.shape)
    D2 = tf.nn.dropout(P2, keep_prob=keep_prob_FC)
    Z4 = tf.contrib.layers.fully_connected(D2, 25, normalizer_fn=tf.contrib.layers.batch_norm, weights_regularizer = regularizer)

    print(Z4.shape)

    D3 = tf.nn.dropout(Z4, keep_prob=keep_prob_FC)

    Z5 = tf.contrib.layers.fully_connected(D3, 1, weights_regularizer = regularizer, activation_fn=None, normalizer_fn=tf.contrib.layers.batch_norm)
    print(Z5.shape)

    return Z5


def find_cost(Z5, Y):
    print(Z5.shape, Y.shape)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z5, labels=Y)
    print (cross_entropy)
    cost = tf.reduce_mean(cross_entropy,name = "cost")
    print(cost)
    return cost



def random_mini_batches(X, Y, mini_batch_size=32):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []


    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def model(X_t, Y_t, learning_rate = 0.001, num_epochs = 100, minibatch_size = 32, type = "train", keep_prob = 0.6):
    tf.reset_default_graph()
    (m,n_r,n_c,_) = X_t.shape
    costs = list()
    # Here keep_prob is a float value, while kp is its tensor
    X, Y, kp = create_placeholders(n_r, n_c)
    parameters = initialize_parameters()
    Z5 = forward_propagation(X, parameters, kp)
    cost = find_cost(Z5, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    if type == "train":
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver()
            for epoch in range(num_epochs):

                minibatch_cost = 0.
                num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set

                minibatches = random_mini_batches(X_t, Y_t, minibatch_size)

                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, kp: keep_prob})


                    minibatch_cost += temp_cost / num_minibatches

                # Print the cost every epoch
                if epoch % 5 == 0:
                    print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                if epoch % 1 == 0:
                    costs.append(minibatch_cost)


            saver.save(sess, './my_model', global_step=1)
            print("Cost: ", sess.run(cost, feed_dict={X: X_t, Y: Y_t, kp: keep_prob}))
    if type == "test":
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, './my_model')
            predicted_result = np.round(sess.run(tf.sigmoid(Z5), feed_dict={X: X_t, Y: Y_t, kp: 1.0}))
            print("Cost: ", sess.run(cost, feed_dict={X: X_t, Y: Y_t, kp: 1.0}))
            m = X_t.shape[0]

            random_result = np.reshape(np.random.randint(2, size=m), (m, 1))

            TP_random = np.sum(np.asarray([1 for i in range(m) if random_result[i, 0] == 1 and Y_t[i, 0] == 1]))
            TN_random = np.sum(np.asarray([1 for i in range(m) if random_result[i, 0] == 0 and Y_t[i, 0] == 0]))
            FP_random = np.sum(np.asarray([1 for i in range(m) if random_result[i, 0] == 1 and Y_t[i, 0] == 0]))
            FN_random = np.sum(np.asarray([1 for i in range(m) if random_result[i, 0] == 0 and Y_t[i, 0] == 1]))
            TP = np.sum(np.asarray([1 for i in range(m) if predicted_result[i, 0] == 1 and Y_t[i, 0] == 1]))
            TN = np.sum(np.asarray([1 for i in range(m) if predicted_result[i, 0] == 0 and Y_t[i, 0] == 0]))
            FP = np.sum(np.asarray([1 for i in range(m) if predicted_result[i, 0] == 1 and Y_t[i, 0] == 0]))
            FN = np.sum(np.asarray([1 for i in range(m) if predicted_result[i, 0] == 0 and Y_t[i, 0] == 1]))
            print(TP, TN, FP, FN)
            balance_accuracy = (1 / 2) * (((TP) / (TP + FN)) + ((TN) / (TN + FP)))
            balance_accuracy_random = (1 / 2) * (((TP_random) / (TP_random + FN_random)) + ((TN_random) / (TN_random + FP_random)))
            print("Random Balance Accuracy = ", balance_accuracy_random)
            print("Balance Accuracy = ", balance_accuracy)
            print("Number of predicted toxic chemicals: ",
                  sess.run(tf.reduce_sum(tf.round(tf.sigmoid(Z5))), feed_dict={X: X_t, Y: Y_t,kp: 1.0 }), np.sum(Y_t))

        return parameters


parameters = model(onehots, labels, type = "test")
