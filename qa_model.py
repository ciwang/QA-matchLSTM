from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import time
import logging
import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
from qa_data import PAD_ID

logging.basicConfig(level=logging.INFO)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim # wat is this
        # pseudocode

    def encode(self, inputs, masks, encoder_state_input, scope="", reuse=False): #wat is masks
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        # symbolic function takes in Tensorflow object, returns tensorflow object
        # pseudocode
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.size, state_is_tuple=False)
        with vs.variable_scope(scope, reuse=reuse):
            (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cell, self.cell, inputs, sequence_length=masks, \
                initial_state_fw=encoder_state_input, initial_state_bw=encoder_state_input, dtype=tf.float64)

        return tf.concat((out_fw, out_bw), 2)

#     def encode_w_attn(self, inputs, masks, encoder_state_input, scope="", reuse=False):
#         self.attn_cell = GRUAttnCell(self.size, encoder_state_input)
#         with vs.variable_scope(scope, reuse):
#             (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.attn_cell, self.attn_cell, inputs, sequence_length=masks, \
#                 dtype=tf.float64)
#         return tf.concat((out_fw, out_bw), 2)

# class GRUAttnCell(tf.contrib.rnn.GRUCell):
#     def __init__(self, num_units, encoder_output, scope=None):
#         self.hs = encoder_output
#         super(GRUAttnCell, self).__init__(num_units)

#     def __call__(self, inputs, state, scope=None):
#         gru_out, gru_state = super(GRUAttnCell, self).__call__(inputs, state, scope)
#         with vs.variable_scope(scope or type(self).__name__):
#             with vs.variable_scope("Attn"):
#                 ht = tf.contrib.layers.fully_connected(inputs=gru_out, num_outputs=self._num_units)
#                 ht = tf.expand_dims(ht, axis=1)
#             scores = tf.reduce_sum(self.hs * ht, reduction_indices=2, keep_dims=True)
#             context = tf.reduce_sum(self.hs * scores, reduction_indices=1)
#             with vs.variable_scope("AttnConcat"):
#                 out = tf.nn.relu(tf.contrib.layers.fully_connected([context, gru_out], self._num_units))
#         return (out, out)

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, inputs):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        # given: h_q, h_p (hidden representations of question and paragraph)
        # TODO: CUT DOWN TO BATCH_SIZE
        # each 2-d TF variable
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.output_size, state_is_tuple=False)
        #tf.expand_dims(h_q, axis=2)
        #tf.expand_dims(h_p, axis=2)
        #print(h_q.get_shape())
        #print(h_p.get_shape())
        with vs.variable_scope("start"):
            # start index of answer
            output_s, _ = tf.nn.dynamic_rnn(self.cell, inputs, dtype=tf.float64)
        with vs.variable_scope("end"):
            # end index of answer
            output_e, _ = tf.nn.dynamic_rnn(self.cell, inputs, dtype=tf.float64)

        # linear function:
        # h_q W + b + h_p W + b
        return (output_s[:, -1, :], output_e[:, -1, :])

class QASystem(object):
    def __init__(self, encoder, decoder, FLAGS, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        # ==== constants ====
        self.lr = 0.001

        # ==== set up placeholder tokens ========
        self.FLAGS = FLAGS
        self.paragraphs_placeholder = tf.placeholder(tf.int32, shape=[None, self.FLAGS.output_size])
        self.questions_placeholder = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_size])
        self.q_masks_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.p_masks_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.start_answer = tf.placeholder(tf.int32, shape=[None, self.FLAGS.output_size])
        self.end_answer = tf.placeholder(tf.int32, shape=[None, self.FLAGS.output_size])

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
            self.setup_training_op()

        # ==== set up training/updating procedure ====
        pass


    def calculate_attention(self, output_q, output_p):
        max_t = output_p.get_shape()[1]
        W = tf.get_variable("W", shape=(2*self.FLAGS.output_size, self.FLAGS.output_size),
            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        attention_p = []
        h_qs = [output_q[:, i, :] for i in range(max_t)]
        for s in range(max_t):
            if s % 10 == 0: print(s)
            h_s = output_p[:, s, :]
            weights = tf.reduce_sum(tf.multiply(h_qs, h_s), axis=[1,2])
            c_s = tf.reduce_sum(tf.multiply(weights, h_qs), axis=0) #batch_size x output_size
            h_att = tf.matmul(tf.concat((c_s, h_s), 1), W)
            attention_p.append(h_att)

        return tf.transpose(tf.stack(attention_p), perm=[1,0,2])

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        encoder = Encoder(self.FLAGS.state_size, self.FLAGS.embedding_size)
        output_q = encoder.encode(self.questions_var, self.q_masks_placeholder, None)
        h_q = output_q[:, -1, :]
        output_p = encoder.encode(self.paragraphs_var, self.p_masks_placeholder, h_q, reuse=True)
        h_p = output_p[:, -1, :]
        # attention_p = self.calculate_attention(output_q, output_p)
        # output_attention_p = encoder.encode_w_attn(self.paragraphs_var, self.p_masks_placeholder, h_p)
        # attention_p = output_attention_p[:, -1, :]
        decoder = Decoder(self.FLAGS.output_size)
        #self.a_s, self.a_e = decoder.decode(h_q, h_p)
        #inputs = tf.concat((output_q, output_p), axis=2)
        self.a_s, self.a_e = decoder.decode(output_p)


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            loss_s = tf.nn.softmax_cross_entropy_with_logits(labels=self.start_answer, logits=self.a_s)
            loss_e = tf.nn.softmax_cross_entropy_with_logits(labels=self.end_answer, logits=self.a_e)
            self.loss = loss_s + loss_e

    def setup_training_op(self):
        self.train_op = get_optimizer("sgd")(self.lr).minimize(self.loss)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            # load data
            glove_matrix = np.load(self.FLAGS.embed_path)['glove']
            embeddings = tf.Variable(glove_matrix, trainable=False)
            self.questions_var = tf.nn.embedding_lookup(embeddings, self.questions_placeholder)
            self.paragraphs_var = tf.nn.embedding_lookup(embeddings, self.paragraphs_placeholder)

    def optimize(self, session, train_q, train_p, train_a, q_masks, p_masks):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed[self.questions_placeholder] = train_q
        input_feed[self.paragraphs_placeholder] = train_p
        input_feed[self.q_masks_placeholder] = q_masks
        input_feed[self.p_masks_placeholder] = p_masks
        # create 1-hot vectors here?
        a_starts = np.array([a[0] for a in train_a])
        a_ends = np.array([a[1] for a in train_a])
        one_hot_start = np.zeros((a_starts.size, self.FLAGS.output_size))
        one_hot_end = np.zeros((a_ends.size, self.FLAGS.output_size))
        one_hot_start[np.arange(a_starts.size), a_starts] = 1
        one_hot_end[np.arange(a_ends.size), a_ends] = 1

        input_feed[self.start_answer] = one_hot_start
        input_feed[self.end_answer] = one_hot_end

        output_feed = [self.train_op, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_q, valid_p, valid_a, q_masks, p_masks):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        input_feed[self.paragraphs_placeholder] = valid_p
        input_feed[self.questions_placeholder] = valid_q
        input_feed[self.p_masks_placeholder] = p_masks
        input_feed[self.q_masks_placeholder] = q_masks
        a_starts = np.array([a[0] for a in valid_a])
        a_ends = np.array([a[1] for a in valid_a])
        one_hot_start = np.zeros((a_starts.size, self.FLAGS.output_size))
        one_hot_end = np.zeros((a_ends.size, self.FLAGS.output_size))
        one_hot_start[np.arange(a_starts.size), a_starts] = 1
        one_hot_end[np.arange(a_ends.size), a_ends] = 1

        input_feed[self.start_answer] = one_hot_start
        input_feed[self.end_answer] = one_hot_end

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_q, test_p, q_masks, p_masks):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x
        input_feed[self.paragraphs_placeholder] = test_p
        input_feed[self.questions_placeholder] = test_q
        input_feed[self.p_masks_placeholder] = p_masks
        input_feed[self.q_masks_placeholder] = q_masks

        output_feed = [self.a_s, self.a_e]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_q, test_p, q_masks, p_masks):

        yp, yp2 = self.decode(session, test_q, test_p, q_masks, p_masks)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset, log=False):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function (at end of each epoch)

        :return:
        """
        valid_cost = 0

        for q, p, a in valid_dataset:
            # questions in valid_dataset may need to be clipped
            q = [question[:self.FLAGS.question_size] + [PAD_ID] * (self.FLAGS.question_size - len(question)) for question in q]
            p = [paragraph[:self.FLAGS.output_size] + [PAD_ID] * (self.FLAGS.output_size - len(paragraph)) for paragraph in p]
            q_lengths = [len(x) for x in q]
            p_lengths = [len(x) for x in p]
            out = self.test(sess, q, p, a, q_lengths, p_lengths)
            valid_cost += sum(out[0])

        if log:
            logging.info("Validate cost: {}".format(valid_cost))

        return valid_cost

    def evaluate_answer(self, session, dataset, rev_vocab, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        f1 = 0.
        em = 0.

        count = 0
        # a is list of tuples
        for q, p, a in dataset:
            if count >= sample: break

            q_lengths = [len(x) for x in q]
            p_lengths = [len(x) for x in p]
            q = [question + [PAD_ID] * (self.FLAGS.question_size - len(question)) for question in q]
            p = [paragraph + [PAD_ID] * (self.FLAGS.output_size - len(paragraph)) for paragraph in p]
            a_s, a_e = self.answer(session, q, p, q_lengths, p_lengths)

            for i in range(len(q)):
                answer = ' '.join(map(lambda x: rev_vocab[x], p[i][a_s[i]:a_e[i] + 1]))
                true_answer = ' '.join(map(lambda x: rev_vocab[x], p[i][a[i][0]:a[i][1] + 1]))

                f1 += f1_score(answer, true_answer)
                em += exact_match_score(answer, true_answer)

            count += len(q)

        f1 /= sample
        em /= sample

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

    def train(self, session, dataset, val_dataset, train_dir, rev_vocab):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        dataset = list(dataset) # is this sketch or nah?
        val_dataset = list(val_dataset)

        # print initial loss
        # val_loss = self.validate(session, val_dataset, log=True)
        # self.evaluate_answer(session, dataset, rev_vocab, log=True)

        # split into train and test loops?
        num_processed = 0
        for e in range(self.FLAGS.epochs):
            random.shuffle(dataset)
            for q, p, a in dataset:
                tic = time.time()
                q = [question[:self.FLAGS.question_size] + [PAD_ID] * (self.FLAGS.question_size - len(question)) for question in q]
                p = [paragraph[:self.FLAGS.output_size] + [PAD_ID] * (self.FLAGS.output_size - len(paragraph)) for paragraph in p]
                q_lengths = [len(x) for x in q]
                p_lengths = [len(x) for x in p]
                loss = self.optimize(session, q, p, a, q_lengths, p_lengths)
                num_processed += 1
                toc = time.time()
                if (num_processed % 100 == 0):
                    logging.info("Num batches processed = %d | Train Epoch ETA = %f" % (num_processed, (len(dataset) - num_processed) * (toc - tic)))
                # diagnostics on memory used
                
            # save the model
            saver = tf.train.Saver()
            results_path = os.path.join(self.FLAGS.train_dir, "results/{:%Y%m%d_%H%M%S}/".format(datetime.datetime.now()))
            model_path = results_path + "model.weights/"
            if not os.path.exists(model_path):
    		    os.makedirs(model_path)
            save_path = saver.save(session, model_path)
            print("Model saved in file: %s" % save_path)
            val_loss = self.validate(session, val_dataset, log=True)
            self.evaluate_answer(session, dataset, rev_vocab, log=True)