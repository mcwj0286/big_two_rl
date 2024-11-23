# import tensorflow as tf
import numpy as np
import joblib
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class PPONetwork(object):
    
    def __init__(self, sess, obs_dim, act_dim, name):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.name = name
        
        with tf.variable_scope(name):
            X = tf.placeholder(tf.float32, [None, obs_dim], name="input")
            available_moves = tf.placeholder(tf.float32, [None, act_dim], name="availableActions")
            activation = tf.nn.relu
            initializer = tf.keras.initializers.VarianceScaling(scale=np.sqrt(2.))
            
            h1 = activation(tf.layers.dense(X, 512, name='fc1', kernel_initializer=initializer))
            h2 = activation(tf.layers.dense(h1, 256, name='fc2', kernel_initializer=initializer))
            pi = tf.layers.dense(h2, act_dim, name='pi', kernel_initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01))
            
            h3 = activation(tf.layers.dense(h1, 256, name='fc3', kernel_initializer=initializer))
            vf = tf.layers.dense(h3, 1, name='vf')[:, 0]
        
        availPi = tf.add(pi, available_moves)    
        
        def sample():
            u = tf.random_uniform(tf.shape(availPi))
            return tf.argmax(availPi - tf.log(-tf.log(u)), axis=-1)
        
        a0 = sample()
        el0in = tf.exp(availPi - tf.reduce_max(availPi, axis=-1, keepdims=True))
        z0in = tf.reduce_sum(el0in, axis=-1, keepdims=True)
        p0in = el0in / z0in
        onehot = tf.one_hot(a0, availPi.get_shape().as_list()[-1])
        neglogpac = -tf.log(tf.reduce_sum(tf.multiply(p0in, onehot), axis=-1))
        
        def step(obs, availAcs):
            a, v, neglogp = sess.run([a0, vf, neglogpac], {X: obs, available_moves: availAcs})
            return a, v, neglogp
                
        def value(obs, availAcs):
            return sess.run(vf, {X: obs, available_moves: availAcs})
        
        self.availPi = availPi
        self.neglogpac = neglogpac
        self.X = X
        self.available_moves = available_moves
        self.pi = pi
        self.vf = vf        
        self.step = step
        self.value = value
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
        def getParams():
            return sess.run(self.params)
        
        self.getParams = getParams
        
        def loadParams(paramsToLoad):
            restores = []
            for p, loadedP in zip(self.params, paramsToLoad):
                restores.append(p.assign(loadedP))
            sess.run(restores)
                
        self.loadParams = loadParams
        
        def saveParams(path):
            modelParams = sess.run(self.params)
            joblib.dump(modelParams, path)
                
        self.saveParams = saveParams
        
        def load_model(self, path):
            params = joblib.load(path)
            self.loadParams(params)
                
        self.load_model = load_model