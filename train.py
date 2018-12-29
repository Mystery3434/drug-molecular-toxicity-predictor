import numpy as np
import tensorflow as tf
# np.set_printoptions(threshold=np.nan)


train_data = np.load('NR-ER-train\\names_onehots.npy').item()
labels = np.genfromtxt('NR-ER-train\\names_labels.csv', delimiter=',', encoding='latin-1')[:,1]
labels = np.reshape(labels, (labels.shape[0], 1))
print(np.sum(labels))

labels_onehots = np.zeros((labels.shape[0],2))
for i in range(labels.shape[0]):
    labels_onehots[i][int(labels[i,:])] = 1
print(labels_onehots)

onehots = train_data['onehots']
onehots = np.reshape(onehots, (onehots.shape[0], onehots.shape[1], onehots.shape[2], 1))
names = train_data['names']
names = np.reshape(names, (names.shape[0], 1))


test_data = np.load('NR-ER-test\\names_onehots.npy').item()
labels_test = np.genfromtxt('NR-ER-test\\names_labels.csv', delimiter=',', encoding='latin-1')[:,1]
labels_test = np.reshape(labels_test, (labels_test.shape[0], 1))
labels_onehots_test = np.zeros((labels_test.shape[0],2))
onehots_test = test_data['onehots']
onehots_test = np.reshape(onehots_test, (onehots_test.shape[0], onehots_test.shape[1], onehots_test.shape[2], 1))
for i in range(labels_test.shape[0]):
    labels_onehots_test[i][int(labels_test[i,:])] = 1
print(labels_onehots_test)




def print_dims():
    print("Labels Dims:", labels.shape)
    print("One Hots Dims:", onehots.shape)
    print("Names Dims:", names.shape)


print_dims()


def create_placeholders(n_r, n_c):
    X = tf.placeholder(name = "X", dtype='float32', shape = (None, n_r, n_c, 1))
    Y = tf.placeholder(name = "Y", dtype='float32', shape = (None, 1))
    keep_prob_conv = tf.placeholder(tf.float32)
    keep_prob_FC = tf.placeholder(tf.float32)
    return X,Y, keep_prob_conv, keep_prob_FC


def initialize_parameters(beta = 0.0):

    #Regularization
    if beta!=0:
        regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
    else: regularizer=None

    W1 = tf.get_variable("W1", [16, 16, 1, 8], regularizer=regularizer,
                         initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [8, 8, 8, 168], regularizer=regularizer,
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


def find_cost(Z5, Y, regularizer=None):
    print(Z5.shape, Y.shape)

    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=Z5, targets=Y, pos_weight=2)
    print (cross_entropy)
    cost = tf.reduce_mean(cross_entropy,name = "cost")

    if regularizer is not None:
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    else:
        reg_term = 0

    cost += reg_term

    print(cost)
    return cost


def random_mini_batches(X, Y, mini_batch_size=32):


    m = X.shape[0]  # number of training examples
    mini_batches = []


    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(m / mini_batch_size)  # number of mini batches of size mini_batch_size in partitionning
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


def model(X_t, Y_t, X_test, Y_test, learning_rate = 0.09, num_epochs = 6, minibatch_size = 16, keep_prob_conv = 1, keep_prob_fc = 0.7, mode = "train"):
    tf.reset_default_graph()
    (m,n_r,n_c,_) = X_t.shape
    costs = list()
    # Here keep_prob is a float value, while kp is its tensor
    X, Y, kp_conv, kp_fc = create_placeholders(n_r, n_c)
    parameters,regularizer = initialize_parameters()
    Z5 = forward_propagation(X, parameters, kp_conv, kp_fc, regularizer=regularizer)
    cost = find_cost(Z5, Y, regularizer)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    if mode == "train":
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

                    _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, kp_conv: keep_prob_conv, kp_fc:keep_prob_fc})

                    minibatch_cost += temp_cost / num_minibatches

                # Print the cost every 5 epochs
                if epoch % 5 == 0:
                    print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                if epoch % 1 == 0:
                    costs.append(minibatch_cost)


            saver.save(sess, './my_model')
            #print("Cost: ", sess.run(cost, feed_dict={X: X_t, Y: Y_t,kp: 1.0}))
            num_minibatches = int(m / minibatch_size)
            prepredictions = sess.run(Z5,
                                      feed_dict={X: X_t[0:minibatch_size, :, :, :],
                                                 Y: Y_t[0: minibatch_size, :],
                                                 kp_conv: 1, kp_fc: 1})
            for i in range(1,num_minibatches):
                prepredictions = np.append(prepredictions, sess.run(Z5, feed_dict={
                    X: X_t[i * minibatch_size: i * minibatch_size + minibatch_size, :, :, :],
                    Y: Y_t[i * minibatch_size: i * minibatch_size + minibatch_size, :], kp_conv: 1, kp_fc:1}), axis = 0)

            if m % minibatch_size != 0:
                prepredictions = np.append(prepredictions, sess.run(Z5, feed_dict={
                    X: X_t[num_minibatches * minibatch_size: m, :, :, :],
                    Y: Y_t[num_minibatches * minibatch_size: m, :], kp_conv: 1, kp_fc:1}), axis=0)



            predicted_result = np.argmax(prepredictions,1)
            m = X_t.shape[0]

            random_result = np.reshape(np.random.randint(2, size = m), (m, 1))


            TP = np.sum(np.asarray([1 for i in range(m) if predicted_result[i]==1 and Y_t[i]==1]))
            TN = np.sum(np.asarray([1 for i in range(m) if predicted_result[i]==0 and Y_t[i]==1]))
            FP = np.sum(np.asarray([1 for i in range(m) if predicted_result[i]==1 and Y_t[i]==1]))
            FN = np.sum(np.asarray([1 for i in range(m) if predicted_result[i]==0 and Y_t[i]==1]))
            print(TP, TN, FP, FN)
            balance_accuracy = (1/2) * (((TP)/(TP+FN))+((TN)/(TN+FP)))

            print("Balance Accuracy = ", balance_accuracy)

            ##TESTING THE TESTING SET


            m = X_test.shape[0]
            num_minibatches = int(m / minibatch_size)
            prepredictions_test = sess.run(Z5,
                                      feed_dict={X: X_test[0:minibatch_size, :, :, :],
                                                 Y: Y_test[0: minibatch_size, :],
                                                 kp_conv: 1, kp_fc: 1})
            for i in range(1, num_minibatches):
                prepredictions_test = np.append(prepredictions_test, sess.run(Z5, feed_dict={
                    X: X_test[i * minibatch_size: i * minibatch_size + minibatch_size, :, :, :],
                    Y: Y_test[i * minibatch_size: i * minibatch_size + minibatch_size, :], kp_conv: 1, kp_fc: 1}), axis=0)

            if m % minibatch_size != 0:
                prepredictions_test = np.append(prepredictions_test, sess.run(Z5, feed_dict={
                    X: X_test[num_minibatches * minibatch_size: m, :, :, :],
                    Y: Y_test[num_minibatches * minibatch_size: m, :], kp_conv: 1, kp_fc: 1}), axis=0)

            predicted_result_test = np.argmax(prepredictions_test, 1)


            random_result = np.reshape(np.random.randint(2, size=m), (m, 1))


            TP_test = np.sum(np.asarray([1 for i in range(m) if predicted_result_test[i] == 1 and Y_test[i] == 1]))
            TN_test = np.sum(np.asarray([1 for i in range(m) if predicted_result_test[i] == 0 and Y_test[i] == 1]))
            FP_test = np.sum(np.asarray([1 for i in range(m) if predicted_result_test[i] == 1 and Y_test[i] == 1]))
            FN_test = np.sum(np.asarray([1 for i in range(m) if predicted_result_test[i] == 0 and Y_test[i] == 1]))
            print(TP_test, TN_test, FP_test, FN_test)
            balance_accuracy_test = (1 / 2) * (((TP_test) / (TP_test + FN_test)) + ((TN_test) / (TN_test + FP_test)))

            print("Balance Accuracy_test = ", balance_accuracy_test)


            return parameters




parameters = model(onehots[:6000,:,:,:], labels[:6000,:], onehots[6000:,:,:], labels[6000:,:])
print(parameters)
