

############### FONCTION DE LA COUCHE DE CONVOLUTION ###################

def fc(tensor, output_dim, IsTrainingMode, name, KP_dropout, act=tf.nn.relu):
    with tf.name_scope(name):
        input_dim = tensor.get_shape()[1].value
        Winit = tf.truncated_normal([input_dim, output_dim], stddev=np.sqrt(2.0/input_dim))
        W = tf.Variable(Winit)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
        print (name,'input  ',tensor)
        print (name,'W  ',W.get_shape())
        #Binit = tf.constant(0.0, shape=[output_dim])  # Avec la BatchNorm, plus besoin de B
        #B = tf.Variable(Binit)
        tensor = tf.matmul(tensor, W) # + B
        tensor = tf.layers.batch_normalization(tensor, axis=-1, training=IsTrainingMode, trainable=True)
        tensor = act(tensor)
        if KP_dropout != 1.0:
            tensor = tf.cond(IsTrainingMode,lambda: tf.nn.dropout(tensor, KP_dropout), lambda: tf.identity(tensor))
    return tensor


def conv(tensor, outDim, filterSize, stride, IsTrainingMode, name, KP_dropout, act=tf.nn.relu):
    with tf.name_scope(name):
        inDimH = tensor.get_shape()[1].value
        inDimW = tensor.get_shape()[2].value
        inDimD = tensor.get_shape()[3].value
        Winit = tf.truncated_normal([filterSize, filterSize, inDimD, outDim], stddev=np.sqrt(2.0/(inDimH*inDimW*inDimD)))
        W = tf.Variable(Winit)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
        print (name, 'input  ', tensor)
        print (name, 'W  ', W.get_shape())
        #Binit = tf.constant(0.0, shape=[outDim])  # Avec la BatchNorm, plus besoin de B
        #B = tf.Variable(Binit)
        tensor = tf.nn.conv2d(tensor, W, strides=[1, stride, stride, 1], padding='SAME') # + B
        tensor = tf.layers.batch_normalization(tensor, axis=-1, training=IsTrainingMode, trainable=True)
        tensor = act(tensor)
        if KP_dropout != 1.0:
            tensor = tf.cond(IsTrainingMode,lambda: tf.nn.dropout(tensor, KP_dropout), lambda: tf.identity(tensor))
    return tensor


def maxpool(tensor, poolSize, name):
    with tf.name_scope(name):
        tensor = tf.nn.max_pool(tensor, ksize=(1,poolSize,poolSize,1), strides=(1,poolSize,poolSize,1), padding='SAME')
    return tensor


def flat(tensor):
    tensor = tf.layers.flatten(tensor, name=None)
    print ('flat output  ', tensor)
    return tensor


def unflat(tensor, outDimH,outDimW,outDimD):
    tensor = tf.reshape(tensor, [-1,outDimH,outDimW,outDimD])
    print ('unflat output  ', tensor)
    return tensor
