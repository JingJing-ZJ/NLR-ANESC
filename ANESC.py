'''
Tensorflow implementation of Attributed Network Embedding With Self Cluster (ANESC)

@author: ZJ

'''


import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

def gamma_incrementer(step, gamma_0, current_gamma, num_steps):
    if step >1:
        exponent = (0-np.log10(gamma_0))/float(num_steps)
        current_gamma = current_gamma * (10 **exponent)
    return current_gamma


class ANESC(BaseEstimator, TransformerMixin):
    def __init__(self, data, id_embedding_size, attr_embedding_size,
                 cluster_number,batch_size, alpha, n_neg_samples,
                 initial_gamma,epoch):
        # bind params to class
        self.batch_size = batch_size
        self.node_N = data.id_N
        self.attr_M = data.attr_M
        self.X_train = data.X
        self.nodes = data.nodes
        self.id_embedding_size = id_embedding_size
        self.attr_embedding_size = attr_embedding_size
        self.alpha = alpha
        self.n_neg_samples = n_neg_samples
        self.epoch = epoch



        self.embedding_size = self.id_embedding_size + self.attr_embedding_size
        self.cluster_number = cluster_number
        self.initial_gamma = initial_gamma

        # init all variables in a tensorflow graph
        self._init_graph()

        print(self.batch_size,self.initial_gamma)
    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():#, tf.device('/cpu:0'):

            # Input data.
            self.train_data_id = tf.placeholder(tf.int32, shape=[None])  # batch_size * 1
            self.train_data_attr = tf.placeholder(tf.float32, shape=[None, self.attr_M])  # batch_size * attr_M
            self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])  # batch_size * 1

            # Variables.
            network_weights = self._initialize_weights()
            self.weights = network_weights

            # Model.
            # Look up embeddings for node_id.
            self.id_embed =  tf.nn.embedding_lookup(self.weights['in_embeddings'], self.train_data_id) # batch_size * id_dim
            self.attr_embed =  tf.matmul(self.train_data_attr, self.weights['attr_embeddings'])  # batch_size * attr_dim
            self.embed_layer = tf.concat( [self.id_embed, self.alpha * self.attr_embed],1) # batch_size * (id_dim + attr_dim)

            # 添加gammma项和聚类项
            self.clustering_differences = tf.expand_dims(self.embed_layer, 1) - self.cluster_means
            self.cluster_distances = tf.norm(self.clustering_differences, ord=2, axis=2)
            self.to_be_averaged = tf.reduce_min(self.cluster_distances, axis=1)
            self.loss_Cluster = tf.reduce_mean(self.to_be_averaged)
            #
            self.gamma = tf.placeholder("float")

            ## can add hidden_layers component here!

            # Compute the loss, using a sample of the negative labels each time.
            self.loss_SNE =  tf.reduce_mean(tf.nn.sampled_softmax_loss(self.weights['out_embeddings'], self.weights['biases'],
                                                  self.train_labels, self.embed_layer,self.n_neg_samples, self.node_N))

            self.loss=self.loss_SNE+self.gamma*self.loss_Cluster

            # Optimizer.
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)


            # init
            init = tf.initialize_all_variables()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['in_embeddings'] = tf.Variable(tf.random_uniform([self.node_N, self.id_embedding_size], -1.0, 1.0))  # id_N * id_dim
        all_weights['attr_embeddings'] = tf.Variable(tf.random_uniform([self.attr_M,self.attr_embedding_size], -1.0, 1.0)) # attr_M * attr_dim
        all_weights['out_embeddings'] = tf.Variable(tf.truncated_normal([self.node_N, self.id_embedding_size + self.attr_embedding_size],
                                    stddev=1.0 / math.sqrt(self.id_embedding_size + self.attr_embedding_size)))
        all_weights['biases'] = tf.Variable(tf.zeros([self.node_N]))

        self.cluster_means = tf.Variable(tf.random_uniform([self.cluster_number, self.embedding_size],
                                                           -0.1 / self.embedding_size, 0.1 / self.embedding_size))

        return all_weights

    def partial_fit(self, X): # fit a batch
        feed_dict = {self.train_data_id: X['batch_data_id'], self.train_data_attr: X['batch_data_attr'],
                                   self.train_labels: X['batch_data_label'],
                                  self.gamma:self.current_gamma}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss


    def train(self): # fit a dataset

        print('Using in + out embedding')


        self.current_step = 0
        self.current_gamma = self.initial_gamma

        for epoch in range( self.epoch ):
            total_batch = int( len(self.X_train['data_id_list']) / self.batch_size)



            # print('total_batch in 1 epoch: ', total_batch)
            # Loop over all batches
            for i in range(total_batch):
                # generate a batch data
                batch_xs = {}

                self.current_step = self.current_step + 1

                start_index = np.random.randint(0, len(self.X_train['data_id_list']) - self.batch_size)
                batch_xs['batch_data_id'] = self.X_train['data_id_list'][start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_attr'] = self.X_train['data_attr_list'][start_index:(start_index + self.batch_size)]
                batch_xs['batch_data_label'] = self.X_train['data_label_list'][start_index:(start_index + self.batch_size)]

                # gamma参数更新
                self.current_gamma = gamma_incrementer(self.current_step, self.initial_gamma, self.current_gamma,
                                                       self.epoch * total_batch)

                # Fit training using batch data
                cost = self.partial_fit(batch_xs)

            # Display logs per epoch
            Embeddings_out = self.getEmbedding('out_embedding', self.nodes)
            Embeddings_in = self.getEmbedding('embed_layer', self.nodes)
            Embeddings = Embeddings_out + Embeddings_in

            print('epoch,self.current_gamma', epoch + 1, self.current_gamma)

        with open('./data/washington/embedding.txt', 'w') as w:
            for i in Embeddings:
                i = list(map(lambda x: str(x), i))
                line = ' '.join(i)
                w.write(line + '\n')


    def getEmbedding(self, type, nodes):
        if type == 'embed_layer':
            feed_dict = {self.train_data_id: nodes['node_id'], self.train_data_attr: nodes['node_attr']}
            Embedding = self.sess.run(self.embed_layer, feed_dict=feed_dict)
            return Embedding
        if type == 'out_embedding':
            Embedding = self.sess.run(self.weights['out_embeddings'])
            return Embedding  # nodes_number * (id_dim + attr_dim)

