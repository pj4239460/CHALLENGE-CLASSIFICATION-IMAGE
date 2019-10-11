
######### Importation des librairies utiles pour le projet.############

import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import Layers

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from time import time

%matplotlib inline
sns.set()

############   Chargement et analyse de nos données.################
#Nombre d'exemples dans nos bases d'entrainement et de test.
num_train_images  = 111430
num_test_images  = 10130

# taille des images 56*56*3 (RGB)
image_dim = 9408 # (= 56 x 56 x 3)

train_images_fname  = './DataBases/db_train.raw'
test_images_fname  = './DataBases/db_test.raw'
train_labels_fname  = './DataBases/label_train.txt'

#Chargement des labels de la base d'apprentissage.
train_images_label = np.loadtxt(train_labels_fname, dtype=np.float64)

############   Transformation de nos données en formes matricielles #######


# Pour la base de Train
fTrain = open(train_images_fname, 'rb')
train_images_data  = np.empty([num_train_images, image_dim], dtype=np.float32)
for i in range(num_train_images):
    train_images_data[i,:] = np.fromfile(fTrain, dtype=np.uint8, count=image_dim).astype(np.float32)
fTrain.close()

# Pour la base de Test
fTest = open(test_images_fname, 'rb')
test_images_data  = np.empty([num_test_images, image_dim], dtype=np.float32)
for i in range(num_test_images):
    test_images_data[i,:] = np.fromfile(fTest, dtype=np.uint8, count=image_dim).astype(np.float32)
fTest.close()

#Vérification de la taille de nos données
print("Informations Bases de Train et Test: ")
print("- Train  :", train_images_data.shape)
print("- Test   :", test_images_data.shape)
print("- Labels :", train_images_label.shape)


#Cette méthode nous permet de visualiser quelques images de notre base d'entrainement.
def printImagesFromLabel(label, nrows=9, ncols=9):
    print("========================================================================================")
    print("- Pour Label : ", label)
    print("========================================================================================")
    index = np.where(train_images_label == label)[0]
    fig = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))
    for i in range(1, ncols*nrows+1):
        plt.subplot(nrows, ncols, i)
        plt.imshow(train_images_data[index[i]].reshape(56, 56, 3).astype(np.uint8))
        plt.xticks([]), plt.yticks([])
        plt.title("class {:.0f}".format(train_images_label[index[i]]))
    plt.show()
    print("========================================================================================")




#Récupération des images pour les labels 0 et 1
for label in [0, 1]:
    printImagesFromLabel(label)


####### Pourcentage de la repartition des images dans notre base d'entrainement.######

train_label_POSITIVE = train_images_label[train_images_label==1]
NB_train_POSITIVE = len(train_label_POSITIVE)

train_label_NEGATIVE = train_images_label[train_images_label==0]
NB_train_NEGATIVE = len(train_label_NEGATIVE)

print("- Nombre Labels positifs : {:d} ==> {:.2f} %".format(NB_train_POSITIVE, 100*NB_train_POSITIVE/num_train_images))
print("- Nombre Labels négatifs  : {:d} ==> {:.2f} %".format(NB_train_NEGATIVE, 100*NB_train_NEGATIVE/num_train_images))


#######   Application de la méthode du Under Sampling à notre base d'entrainement. ######
RandUS = RandomUnderSampler(ratio='auto', random_state=0)
X_US, y_US = RandUS.fit_sample(train_images_data, train_images_label)


# Affichage de la Nouvelle distribution après le Under Sampling
train_label_POSITIVE = y_US[y_US==1]
NB_train_POSITIVE = len(train_label_POSITIVE)

train_label_NEGEGATIVE = y_US[y_US==0]
NB_train_NEGATIVE = len(train_label_NEGEGATIVE)

print("- Nombre Labels positifs : {:d} ==> {:.2f} %".format(NB_train_POSITIVE, 100*NB_train_POSITIVE/len(y_US)))
print("- Nombre Labels négatifs  : {:d} ==> {:.2f} %".format(NB_train_NEGATIVE, 100*NB_train_NEGATIVE/len(y_US)))

# Vérification de la taille de nos données après le Under Sampling

print("Informations apres le Under Sampling. ")
print("- Train Under sampling  :", X_US.shape)
print("- Labels Under Sampling :", y_US.shape)


############### Encodage One Hot de nos labels ####################

#Nombre de classe correspondant à notre classification
C = 2
Y = y_US.astype(int)
y_US_encoded = np.eye(C)[Y.reshape(-1)]

print("Forme Label en Train après encodage :", y_US_encoded.shape)


################ Normalisation de notre base et séparation en deux bases de données  ##########

# Separation
X_train, X_test, y_train, y_test = train_test_split(X_US, y_US_encoded,
                                                    test_size=0.05, shuffle=True, random_state=42)

# Normalisation
X_train = np.divide(X_train, 255., dtype=float)
X_test = np.divide(X_test, 255., dtype=float)

# Verification des tailles
print("Informations sur nos nouvelles bases de données.")
print("- Train  :", X_train.shape)
print("- Test  :", X_test.shape)
print("- Labels Train  :", y_train.shape)
print("- Labels Test :", y_test.shape)


############### Tableau issu du "one hot encoding"  #################



tableau_Label = pd.DataFrame(data =y_test, index= np.arange(1, len(y_test)+1),columns = ['Label 0','Label 1'])
tableau_Label.head()


###############  Visualisation de l'éffet de la normalisation sur nos bases d'entrainement et de test ###########
# sur la base d'entrainement (X_train)
print("Base de Train:\n",X_train)
print("=========================================================================")

# sur la base de test (X_test)
print("Base de Test:\n",X_test)




################################################################################################################
######### CONSTRUCTION DE L'ARCHITECTURE DE MON RESEAU DE CONVOLUTION ########################################
################################################################################################################
tf.reset_default_graph()

# Parameters to tune
learn_rate = 3e-3
KeepProb_Dropout = 0.8
L2_reg = 0.001

# Parameters of the model
nb_block = 4
nb_conv_per_block = 2
nbfilter_ini = 10


with tf.name_scope('input'):
    dim = 9408  # =(56*56*3)
    x = tf.placeholder(tf.float32, [None, dim], name='x')
    y_true = tf.placeholder(tf.float32, [None, 2], name='y_true')
    ITM = tf.placeholder("bool", name='Is_Training_Mode')

with tf.name_scope('CNN'):
    t = Layers.unflat(x,56,56,3)
    nbfilter = nbfilter_ini
    for k in range(nb_block):
        for i in range(nb_conv_per_block):
            t = Layers.conv(t, outDim=nbfilter, filterSize=3, stride=1, IsTrainingMode=ITM,
                            name='conv_%d_%d'%(nbfilter,i), KP_dropout=KeepProb_Dropout)
        t = Layers.maxpool(t, poolSize=2, name='pool')
        nbfilter *= 2

    t = Layers.flat(t)
    t = Layers.fc(t,256, ITM, 'fc_0', KeepProb_Dropout, act=tf.nn.relu)
    t = Layers.fc(t, 64, ITM, 'fc_1', KP_dropout = 0.9, act=tf.nn.relu)
    z = Layers.fc(t,  2, ITM, 'fc_2', KP_dropout=1.0,   act=tf.identity)


with tf.name_scope('cost_function'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y_true))
    l2_loss = L2_reg * tf.add_n( [tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.WEIGHTS)] )
    cost_func = cross_entropy + l2_loss
    tf.summary.scalar('cost_func', cost_func)

with tf.name_scope('metrics'):
    y_true_vector = tf.cast(tf.argmax(y_true, axis=1), tf.float32)
    y_pred_vector = tf.cast(tf.argmax(z, axis=1), tf.float32)
    with tf.name_scope('score_Challenge'):
        N0 = tf.reduce_sum(tf.cast(tf.equal(y_true_vector, 0), tf.float32))
        N1 = tf.reduce_sum(tf.cast(tf.equal(y_true_vector, 1), tf.float32))
        score_chall = 0.5 * (tf.reduce_sum((1-y_true_vector)*(1-y_pred_vector))/N0 \
                             + tf.reduce_sum(y_true_vector*y_pred_vector)/N1)
        tf.summary.scalar('score_challenge', score_chall)

with tf.name_scope('learning_rate'):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learn_rate, global_step, decay_steps=200, decay_rate=0.9, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost_func, global_step=global_step)

merged = tf.summary.merge_all()


############################## LES FOCNTIONS UTILES A L'ENTRAINEMENT DU MODEL ########################################


def create_random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (number of examples, input size)
    Y -- true "label", of shape (number of examples, 2)
    mini_batch_size - size of the mini-batches, integer
    seed -- to set the randomness

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]     # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(m/mini_batch_size) # number of mini batches of size mini_batch_size in (X, Y)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches



def train_model(experiment_name_base, experiment_name, X_train, y_train, X_test, y_test,
                minibatch_size=256, num_epochs_to_run=1, epoch_start=0, it_start=0, print_cost=True):

    print ("-------    ", experiment_name, "  -------\n")
    it = it_start

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=None)  # by default, tf keeps only the 5 last models

    if epoch_start == 0:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, experiment_name)

    writer = tf.summary.FileWriter(experiment_name_base, sess.graph)

    score_train_history = []
    score_test_history = []

    for epoch in range(num_epochs_to_run):

        t0 = time()
        seed = epoch_start + epoch + 1  # at each epoch, set a different seed (to split differently into minibatches)
        minibatches = create_random_mini_batches(X_train, y_train, minibatch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            trainDict = {x:minibatch_X, y_true:minibatch_Y, ITM:True}

            # Run a gradient descent step on a minibatch
            sess.run(train_step, feed_dict=trainDict)
            it += 1

            # Print informations every 10 iterations
            if print_cost == True and it % 10 == 0:
                cost, sco, lr = sess.run([cost_func, score_chall, learning_rate], feed_dict=trainDict)
                n0 = np.sum(trainDict[y_true][:,0])
                n1 = np.sum(trainDict[y_true][:,1])
                print("iteration= %6d | rate= %f | cost= %f | score= %f | (#0=%d - #1=%d)" % (it, lr, cost, sco, n0, n1))

                # Write (Cost, LearningRate, Score) in tensorboard
                summary_merged = sess.run(merged, feed_dict=trainDict)
                writer.add_summary(summary_merged, it)

        if print_cost == True:
            # Compute Train Score using minibatches
            t = time()
            sco_train = 0
            m = y_train.shape[0]
            for minibatch in minibatches:
                # select a mini-batch
                (minibatch_X, minibatch_Y) = minibatch
                trainDict = {x:minibatch_X, y_true:minibatch_Y, ITM:False}
                # compute score on current mini-batch and add it to the total
                sco_tmp = sess.run(score_chall, feed_dict=trainDict)
                sco_train += sco_tmp
            sco_train = sco_train / (m/minibatch_size)
            # Keep track of train score
            score_train_history.append(sco_train)
            print ("Train Score: %.6f"%sco_train, "    time=%.3f"%(time()-t))

            # Compute Test Score using .eval  (works only for small dataset to evaluate)
            t = time()
            sco_test = score_chall.eval({x: X_test, y_true: y_test, ITM:False}, session=sess)
            # Keep track of test score
            score_test_history.append(sco_test)
            print ("Test Score : %.6f"%sco_test, "    time=%.3f"%(time()-t))

            print ("     Execution time after epoch %2d : %.3f (sec)\n" % (epoch_start+epoch+1, time()-t0))

        # At the end of each epoch, save the model
        saver.save(sess, experiment_name_base+"_epoch_%d"%(epoch_start+epoch+1))

    print("last iteration =", it)
    print("last epoch =", epoch_start+epoch+1)
    print()
    writer.close()
    sess.close()

    return it, epoch_start+epoch+1, score_train_history, score_test_history


################################# Lancement de l'entrainement ########################

minibatch_size = 256
num_epochs_to_run = 200
epoch_start = 0
it_start = 0

experiment_name_base = "./resultats_modeles/Model_KeepPD%.2f_MBSize%d_NbBlock%d_ConvPerB%d_NbFilter%d_Learn_R%.4f_L2Rg-%.3f.ckpt" % \
                    (KeepProb_Dropout, minibatch_size, nb_block, nb_conv_per_block, nbfilter_ini, learn_rate, L2_reg)
experiment_name = experiment_name_base

#  ENTRAINER LE MODELE
last_it, last_epoch, historique_score_train_with_200_epochs, historique_score_test_with_200_epochs = train_model(experiment_name_base,
                                                                                                             experiment_name,
                                                                                                             X_train, y_train,
                                                                                                             X_test, y_test,
                                                                                                             minibatch_size,
                                                                                                             num_epochs_to_run,
                                                                                                             epoch_start,
                                                                                                             it_start,
                                                                                                             print_cost = True)

np.savetxt("./historique/historique_score_train_with_200_epochs.txt", historique_score_train_with_200_epochs)
np.savetxt("./historique/istorique_score_test_with_200_epochs.txt", historique_score_test_with_200_epochs)


########## Courbe de Variation des scores de notre modèle sur les bases de Train et de Test ###############



#Chargement des scores sur la base de train.
score_train_history_with_200_epochs = np.loadtxt("./historique/historique_score_train_with_200_epochs.txt")
score_test_history_with_200_epochs = np.loadtxt("./historique/istorique_score_test_with_200_epochs.txt")
fig = plt.figure(figsize=(16, 12))
xx = np.arange(1, len(score_train_history_with_200_epochs)+1)
plt.plot(xx, score_train_history_with_200_epochs, label="Score en Train")
plt.plot(xx, score_test_history_with_200_epochs, label="Score en Test")
plt.xlabel("Epochs")
plt.ylabel("Scores")
plt.legend()
plt.show()


###############  Récupération des Scores en Train et en Test sous forme de tableau #######################

tableau_train_test_score = pd.DataFrame(data = np.vstack((score_train_history_with_200_epochs,
                                                             score_test_history_with_200_epochs)).T,
                        index   = np.arange(1, len(score_train_history_with_200_epochs)+1),
                        columns = ['Score Train','Score Test'])

# Nous trions ensuite sur la colonne 'Score Test'
tableau_train_test_score.sort_values('Score Test', ascending=False, inplace=True)

#Récupération de nos trois meilleurs modèles.
tableau_train_test_score.head(3)


######################### LES FONCTIONS D'EVALUATION SUR TOUTE LA BASE ##################################################
def give_mini_batch_score_for_all_Images(X, Y, mini_batch_size=256):

    m = X.shape[0]          # number of training examples
    mini_batches = []

    # Here we don't need to shuffle the data before splitting into mini-batches!

    # Partition (X, Y). Minus the end case.
    num_complete_minibatches = int(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_Y = Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches



def compute_score_all_trainImages(model_to_restore, train_images_data, train_images_label, mbs=256, print_time=True):

    with tf.Session() as sess:
        t0 = time()

        # Restore the model
        saver = tf.train.Saver()
        saver.restore(sess, model_to_restore)

        # One hot encoding of the binary vector of labels
        train_images_label_encoded = np.eye(2)[train_images_label.astype(int).reshape(-1)]

        # Split into mini-batches
        minibatches = give_mini_batch_score_for_all_Images(train_images_data, train_images_label_encoded, mini_batch_size=mbs)
        sco_train = 0

        for i, minibatch in enumerate(minibatches):
            # Select a mini-batch
            (minibatch_X, minibatch_Y) = minibatch

            # Normalize Inputs
            minibatch_X_norm = np.divide(minibatch_X, 255., dtype=float)

            # Compute score on current mini-batch and add it to total
            trainDict = {x:minibatch_X_norm, y_true:minibatch_Y, ITM:False}
            sco_tmp = sess.run(score_chall, feed_dict=trainDict)
            sco_train += sco_tmp

            if print_time == True and i % 100 == 0:
                print("iteration %d - Execution time from start: %.2f (sec)"%(i, time()-t0))

        m = train_images_label_encoded.shape[0]
        sco_train = sco_train / (m/mbs)

        print("Execution Time : %.3f (sec)" % (time()-t0))
        print("  ==> Score = %.6f" % sco_train)

        return sco_train


############################   EVALUATION DE MES TROIS MEILLEURS MODELS ##########################
# Models N°1: score en test = 0.921050 | Epoch = 59
model_to_restore = "./resultats_modeles/Model_KeepPD0.80_MBSize256_NbBlock4_ConvPerB2_NbFilter10_Learn_R0.0030_L2Rg-0.001.ckpt_epoch_59"

score_allTrainImages_ep59 = compute_score_all_trainImages(model_to_restore, train_images_data, train_images_label,
                                                  mbs=256, print_time=True)



# Models N°2: score en test = 0.920744 | Epoch = 42
model_to_restore = "./resultats_modeles/Model_KeepPD0.80_MBSize256_NbBlock4_ConvPerB2_NbFilter10_Learn_R0.0030_L2Rg-0.001.ckpt_epoch_42"

score_allTrainImages_ep42 = compute_score_all_trainImages(model_to_restore, train_images_data, train_images_label,
                                                  mbs=256, print_time=True)


# Models N°3: score en test = 0.919430 | Epoch = 54
model_to_restore = "./resultats_modeles/Model_KeepPD0.80_MBSize256_NbBlock4_ConvPerB2_NbFilter10_Learn_R0.0030_L2Rg-0.001.ckpt_epoch_54"

score_allTrainImages_ep54 = compute_score_all_trainImages(model_to_restore, train_images_data, train_images_label,
                                                  mbs=256, print_time=True)




##################### LES FONCTIONS DE LANCEMENT DES PREDICTIONS DANS UN FICHIER TEXTE  ####################

def prediction_for_minibatches(X, mini_batch_size=64):
    m = X.shape[0]
    mini_batches = []

    num_complete_minibatches = int(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batches.append(mini_batch_X)

    if m % mini_batch_size != 0:
        mini_batch_X = X[num_complete_minibatches * mini_batch_size : m, :]
        mini_batches.append(mini_batch_X)

    return mini_batches



def launch_final_prediction(model_to_restore, images, mbs=256, print_time=True):

    with tf.Session() as sess:
        t0 = time()

        # Restoration du modèle
        saver = tf.train.Saver()
        saver.restore(sess, model_to_restore)

        # Normalization des inputs, comme précédemment(après la séparation de notre base train)
        images_norm = np.divide(images, 255., dtype=float)

        # Découpage en minibatches
        #print("I am here 1")
        predictions = np.array([])
        #print("I am here 2")

        minibatches = prediction_for_minibatches(images_norm, mini_batch_size=mbs)
        #print("I am here 3")
        # Calcul de la prédiction pour chaque minibatch.
        for i, minibatch_X in enumerate(minibatches):
            #print("I am here 4")
            minibatch_pred = sess.run(y_pred_vector, feed_dict={x:minibatch_X, ITM:False})
            #print("I am here 5")
            # Ajout du minibatch dans le tableau de prédictions
            predictions = np.append(predictions, minibatch_pred, axis=0)
        print("Temps execution : %.2f (sec)" % (time()-t0))
        print("Nombre de ligne du fichier \'prediction.txt':", predictions.shape[0])
        #print(predictions)
        return  predictions


################################################################################################################
######### GENERATION DU FICHIER prediction.txt AVEC MON MEILLEUR MODELE ########################################
################################################################################################################

meilleur_model_epoch_59 = "./resultats_modeles/Model_KeepPD0.80_MBSize256_NbBlock4_ConvPerB2_NbFilter10_Learn_R0.0030_L2Rg-0.001.ckpt_epoch_59"
pred_test = launch_final_prediction(meilleur_model_epoch_59,test_images_data , mbs=256)

#Sauvegarde dans le fichier .txt
np.savetxt("./predictions/prediction.txt", pred_test, fmt="%2d")
