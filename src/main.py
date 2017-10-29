
# coding: utf-8

# ## Tensorflowで"Semi-supervised Clustering for Short Text via Deep Representation Learning"の実装¶
# http://aclweb.org/anthology/K16-1004

# In[1]:

import tensorflow as tf
import numpy as np
import sys
import random as rd
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
import MeCab
import subprocess
import itertools
import string
import sqlite3
from keras.preprocessing import sequence
rng = np.random.RandomState(1234)


# # 1.データの準備

# In[2]:

#
#ここで各ファイルのパスを設定します。
#

#学習に用いるデータ
dbpath = "../input/category_walkerplus.db"

#word2vecのモデル
model_path="../input/word2vec.gensim.model"

#辞書
dic_path="/usr/local/lib/mecab/dic/mecab-ipadic-neologd"


# In[3]:

model = Word2Vec.load(model_path)
tagger = MeCab.Tagger("-Ochasen -d {0}".format(dic_path))


# In[4]:

def cur(sourcedbname):
    con = sqlite3.connect(sourcedbname)
    cur = con.cursor()   
    sql = "select * from events"
    cur.execute(sql)
    return cur


# In[5]:

Cur=cur(dbpath)

labels=[]
texts=[]
for row in Cur:
    labels.append(row[0].replace("[","").replace("]","").split(",")[1].replace(" ",""))  #Big Category
    texts.append(row[1])


# ## 分散表現の獲得
# Data：入力の文章を分かち書きしw2vで埋め込み後padding shape=(データ数,maxlen,埋め込み次元)<br/>
# Labels:正解ラベル

# #### 分かち書き

# In[6]:

def _tokenize(text):
    sentence = []
    node = tagger.parse(text)
    #print node
    node = node.split("\n")
    for i in range(len(node)):
        feature = node[i].split("\t")
        if feature[0] == "EOS":
            break
        hinshi = feature[3].split("-")[0]
        if "名詞" in hinshi:
            #sentence.append(feature[2].decode('utf-8'))
            sentence.append(feature[2])
        elif "形容詞" in hinshi:
            #sentence.append(feature[2].decode('utf-8'))
            sentence.append(feature[2])
        elif "動詞" in hinshi:
            #sentence.append(feature[2].decode('utf-8'))
            sentence.append(feature[2])
        elif "形容動詞" in hinshi:
            #sentence.append(feature[2].decode('utf-8'))
            sentence.append(feature[2])
        elif "連体詞" in hinshi:
            #sentence.append(feature[2].decode('utf-8'))
            sentence.append(feature[2])           
        elif "助詞" in hinshi:
            #sentence.append(feature[2].decode('utf-8'))
            sentence.append(feature[2])
            
    return sentence


# ### 分散表現の獲得

# In[7]:

def getVector(text):
    texts = _tokenize(text)
    v = []
    for t in texts:
        if t in model.wv:
            if v == []:
                v = model.wv[t]
            else:
                v = np.vstack((v,model.wv[t]))
    if v != []:
        return v
    else:
        return np.array([])


# In[62]:

Data=np.array([getVector(text) for text in texts])


# In[63]:

Data = sequence.pad_sequences(Data, padding="post", truncating="post",dtype="float32")  #padding  


# In[64]:

label2Label={name:i for i,name in enumerate(np.unique(labels))}  #label(名前)->Label(数字)
Labels=np.array([label2Label[label] for label in labels])  


# # 2.教師データの選択

# In[65]:

def load_data(Data,Labels,training_percent=0.9, supervise_percent=0.1):
    """
    Args:
        Data：入力の文章を分かち書きしw2vで埋め込み後padding shape=(データ数,maxlen,埋め込み次元)
        Labels:正解ラベル
        training_percent:trainingデータに使用する割合
        supervise_percent: supervise(教師データ)として一部与える割合(training_dataに対して)
    Returns:
        train_X:Dataの学習用
        train_y:Labelsの学習用
        test_X:Dataのテスト用
        test_y:Labelsのテスト用
        supervised:学習用データにおいて一部与える教師データのindex
    """
    #trainとtestでclassが均等になるようにsplit
    cluster_num=np.unique(Labels)
    train_index = []
    
    for i in range(len(cluster_num)):
        num = Labels[Labels==i].shape[0]
        k = int(num*training_percent)
        train_index.extend(rd.sample(list(np.where(Labels==i)[0]),k))
    
    #text classification for cnn のためにあらかじめlayerを追加
    train_X = Data[train_index][:,:,:,np.newaxis]
    train_y = Labels[train_index]
    test_X = np.delete(Data,train_index,0)[:,:,:,np.newaxis]
    test_y = np.delete(Labels,train_index,0)
    
    supervised = []

    for i in range(len(cluster_num)):
        num = train_y[train_y==i].shape[0]
        k = int(num*supervise_percent)
       
        supervised.extend(rd.sample(list(np.where(train_y==i)[0]),k))

    
    return train_X,train_y,test_X,test_y,supervised


# In[66]:
def main():
    #0.1~0.9で実験
    for III in range(10):
        if III>6:
            print("ただいま",III,"です")
            supervise_percent=III/10

            train_X,train_y,test_X,test_y, supervised = load_data(Data,Labels,supervise_percent=supervise_percent)


            # ## 3.CNN for Text Classificationの実装

            # ### HyperParams

            # In[67]:

            #窓の幅
            filter_sizes = [3,5,7]
            #分散表現の次元
            vector_length = train_X.shape[2]
            #最大系列長
            sequence_length = train_X.shape[1]
            #フィルターの枚数
            num_filters = 16
            #隠れ層
            hid_dim=100
            #出力次元数
            output_dim=100
            #クラスタリングするクラス多数
            n_cluster=len(np.unique(train_y))

            alpha=0.01
            l=0


            # In[68]:

            class Conv:
                def __init__(self, sequence_length,embedding_size,filter_sizes, num_filters):
                    self.sequence_length=sequence_length
                    self.embedding_size=embedding_size
                    self.filter_sizes=filter_sizes
                    self.num_filters=num_filters
                def f_prop(self,x):
                    # Create a convolution + maxpool layer for each filter size
                    pooled_outputs = []
                    for i, filter_size in enumerate(self.filter_sizes):
                        with tf.name_scope("conv-maxpool-%s" % filter_size):
                            # Convolution Layer
                            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                            
                            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                            conv = tf.nn.conv2d(
                                x,
                                W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")
                            
                            # Apply nonlinearity
                            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                            # Maxpooling over the outputs
                            pooled = tf.nn.max_pool(
                                h,
                                ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name="pool")
                            pooled_outputs.append(pooled)

                    # Combine all the pooled features
                    num_filters_total = num_filters * len(self.filter_sizes)
                    self.h_pool = tf.concat(pooled_outputs, 3)
                    self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
                    return self.h_pool_flat

            class Dense:
                def __init__(self, in_dim, out_dim, function=lambda x: x):
                    # Xavier initializer
                    self.W = tf.Variable(rng.uniform(
                                    low=-np.sqrt(6/(in_dim + out_dim)),
                                    high=np.sqrt(6/(in_dim + out_dim)),
                                    size=(in_dim, out_dim)
                                ).astype('float32'), name='W')
                    self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
                    self.function = function

                def f_prop(self, x):
                    return self.function(tf.matmul(x, self.W) + self.b)
                    
            class LinearDense:
                def __init__(self, in_dim, out_dim):
                    # Xavier initializer
                    self.W = tf.Variable(rng.uniform(
                                    low=-np.sqrt(6/(in_dim + out_dim)),
                                    high=np.sqrt(6/(in_dim + out_dim)),
                                    size=(in_dim, out_dim)
                                ).astype('float32'), name='W')
                    self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))

                def f_prop(self, x):
                    return tf.matmul(x, self.W) + self.b


            # ### グラフの構築

            # In[69]:

            input_x = tf.placeholder(tf.float32, [None, sequence_length,vector_length,1], name="input_x")
            input_t = tf.placeholder(tf.float32, [None, output_dim], name="input_y")

            layers = [
                Conv(sequence_length=sequence_length,
                     embedding_size=vector_length,
                     filter_sizes=filter_sizes,
                     num_filters=num_filters),
                Dense(num_filters * len(filter_sizes),hid_dim , tf.nn.tanh),
                LinearDense(hid_dim, output_dim)
            ]

            def f_props(layers, x):
                for i, layer in enumerate(layers):
                    x = layer.f_prop(x)
                return x

            pred_y = f_props(layers, input_x)


            # ### 誤差関数の設計
            # <img src='../img/jsemi.png'>

            # In[70]:

            #目的関数の定義
            def _cost(pred_y, centers, neighbor_index, sup_index, mask):
                """
                Args:
                    pred_y: text-cnn の出力
                    centers:各ラベルごとの重心 shape=(cluster_num, output_dim)
                    neighbor_index:train_Xがどの重心に最も近いか shape=(data_num,)
                    sup_index:教師データとして用いるtrain_Xの対応するクラスターID index shape=(sup_num,)
                    mask:教師データとして用いるindexに1、それ以外に0が入ったmask
                """
                
                term1= tf.reduce_sum(tf.square(pred_y - tf.gather(centers, neighbor_index)))    
                term1_1 = alpha*tf.cast(term1, tf.float32)
                
                
                term2 = tf.reduce_sum(mask * tf.square(pred_y - tf.gather(centers, sup_index) ))
                term2_1 = (1-alpha)*tf.cast(term2, tf.float32)

                cost = tf.add(term1_1, term2_1)
                
                for i in range(n_cluster):
                    i_index = i * tf.ones_like(sup_index, dtype='int32')
                    
                    x1 = tf.reduce_sum(mask*tf.square(pred_y - tf.gather(centers, sup_index)),1)#正解の重心との距離
                    x2 = tf.reduce_sum(mask*tf.square(pred_y - tf.gather(centers, i_index)),1)     #クラスターIDがiの重心との距離
                    x1 = tf.reshape(x1, [-1, 1])
                    x2 = tf.reshape(x2, [-1, 1])
                    
                    term2_2 = l+x1-x2
                    
                    condition = tf.greater(term2_2, 0)
                    term2_3 = tf.reduce_sum(mask * tf.where(condition, term2_2, tf.zeros_like(term2_2)))
                    term2_3 = tf.cast(term2_3, tf.float32)

                    cost=tf.add(cost, (1-alpha)*term2_3)
                    
                return cost


            # In[71]:

            #  centers:各ラベルごとの重心 shape=(cluster_num, output_dim) 
            centers= tf.placeholder(tf.float32, [None, output_dim], name="centers")

            #  neighbor_index:train_Xがどの重心に最も近いか shape=(data_num,)
            neighbor_index= tf.placeholder(tf.int32, [None], name="neighbor_index")

            #  sup_index:教師データとして用いるtrain_Xの対応するクラスターID shape=(sup_num,)
            sup_index=tf.placeholder(tf.int32, [None], name="supervised_index")

            #  mask:教師データとして用いるindexに1、それ以外に0が入ったmask
            mask=tf.placeholder(tf.float32, [None,1], name="mask")

            #自作の誤差関数
            cost = _cost(pred_y, centers, neighbor_index, sup_index, mask)

            # 最小化にはAdamを用いる
            train = tf.train.AdamOptimizer().minimize(cost)


            # # 4.Iteration
            # <img src='../img/iter.png'>

            # In[72]:

            def assign_to_nearest(samples, centroids):
                """
                Args:
                    samples:text-cnn後の出力
                    centroids:クラスターそれぞれの重心 shape=(cluster_num,output_dim)
                Returns:
                    nearest: 入力データと最も近いクラスターID shape=(data_num, )
                """
                #1-1.KNearest Neighborで一番近いクラスタと紐付け
                neigh = KNeighborsClassifier(n_neighbors=1)
                neigh.fit(centroids, np.arange(len(centroids)))
                nearest = neigh.predict(samples)
                
                #1-2.ハンガリアンアルゴリズムでラベル付きデータと重心を紐付ける
                
                sup_labels =train_y[supervised]
                hglabel = np.unique(sup_labels)
                
                hgx=[]  #教師データのラベルごとに平均した点（重心）を求める
                for i in hglabel:
                    ind=np.where(sup_labels==i)[0]
                    hgx.append(np.mean(samples[supervised][ind], axis=0))
                hgx = np.array(hgx)

                #教師ラベルごとの重心と現在の重心との距離行列
                DistanceMatrix = np.linalg.norm(hgx[:,np.newaxis,:]-centroids[np.newaxis,:,:],axis=2)  
                
                # ハンガリアンアルゴリズムで合計が一番小さくなるように紐づける 
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(DistanceMatrix)
                
                #ラベルとclusterIDを紐づける
                label2id={hglabel[i]:col for i,col in enumerate(col_ind)}
                
                
                return nearest ,label2id


            # In[73]:

            def estimate_centroids(samples, nearest, label2id, use_supervised=True):
                """
                重心を再推定する
                重心の再推定式は簡略化してラベル付きと無しの加重平均
                Args:
                    samples:text-cnn後の出力
                    nearest: 入力データと最も近いクラスターID shape=(data_num, )
                    label2id:labelとクラスターIDを紐づけるdict
                    use_supervised:教師データを用いるかどうか（checkの際K-Meansが上手くいくか検証する際に用いいる）
                Returns:
                    centroids:クラスターそれぞれの重心 shape=(cluster_num,output_dim)
                    sup_cent:一部の教師データごとに現時点で対応するIDの重心が入った配列
                """
                sup_pred=samples[supervised]
                newce=np.array([label2id[t] for t in train_y[supervised]]) #現時点での一部の教師データと対応するクラスターIDの配列
                        
                centroids=[]
                for i in range(n_cluster):
                    sum1 = np.sum(alpha*len(np.where(nearest==i)[0]))
                    sum2 = np.sum(alpha*samples[nearest==i], axis=0)
                
                    if use_supervised:
                        
                        sum33 = np.sum((1-alpha)*len(np.where(newce==i)[0]))
                        sum44 = np.sum((1-alpha)*sup_pred[newce==i], axis=0)
                        
                        centroids.append((sum2+sum44)/(sum1+sum33))
                        
                    else:
                        centroids.append((sum2)/(sum1))
                        
                centroids = np.array(centroids) 

                #3.NNのパラメータ更新
                sup_cent = np.array([centroids[c_id] for c_id in newce])
                
                return centroids, sup_cent



            # # 5.学習

            # In[ ]:

            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            # 学習に関するHyperParams 
            max_epochs= 300

            nb_epoch=1 #バッチ学習一周を何回行うか
            batch_size = 32
            n_batches = train_X.shape[0]//batch_size


            #初期値獲得
            samples=sess.run(pred_y, feed_dict={input_x:train_X})
            centroids = np.array(rd.sample(list(samples),n_cluster))

            #Mask
            mask_array = np.zeros(len(train_X), dtype=int)
            mask_array[supervised] = 1        
                

            for iters in range(max_epochs):
                print("Epoch : ",iters)
                savename="../output/{0:03d}".format(iters)
                
                samples=sess.run(pred_y, feed_dict={input_x:train_X})
                
                nearest, label2id = assign_to_nearest(samples, centroids)

                centroids, sup_cent = estimate_centroids(samples, nearest, label2id, use_supervised=True)
                

                
                newce=np.array([label2id[t] for t in train_y])  #正解ラベル->対応するクラスターIDした配列    
                
                for _ in tqdm(range(nb_epoch)):
                    shuffled_train_X,  shuffled_train_y, shuffled_nearest, shuffled_newce, shuffled_mask =shuffle(train_X,train_y, nearest, newce,mask_array)
                   
                    for i in range(n_batches):  
                        start = i * batch_size
                        end = start + batch_size

                        batch_X = shuffled_train_X[start:end]
                        batch_y = shuffled_train_y[start:end]
                        batch_nearest = shuffled_nearest[start:end]
                        batch_newce   = shuffled_newce[start:end]
                        batch_mask     = shuffled_mask[start:end].reshape(-1,1)

                        sess.run(train, feed_dict={input_x:batch_X, centers : centroids, neighbor_index:batch_nearest, sup_index:batch_newce, mask:batch_mask })
               
                
                #収束を確認
                if iters>0:
                    if np.all(nearest== prev_nearest):
                            break
                    else:
                            from sklearn.metrics import accuracy_score
                            print(accuracy_score(nearest, prev_nearest))
                            prev_nearest = nearest
                            
                else:
                            prev_nearest = nearest


            # In[27]:

            def plot_clusters(all_samples, centroids, indices, save=False,save_name='output.png',iter_num=-1):
                
                import matplotlib.pyplot as plt
                
                if all_samples.shape[1] >2:
                    from sklearn.manifold import TSNE
                    all_samples= TSNE(n_components=2, random_state=1).fit_transform(all_samples)

                #Plot out the different clusters
                #Choose a different colour for each cluster
                colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
                
                for i, centroid in enumerate(centroids):
                    #Grab just the samples fpr the given cluster and plot them out with a new colour
                    samples = all_samples[indices==i]
                    plt.scatter(samples[:,0], samples[:,1], c=colour[i])
                    
                    #Also plot centroid
                    #plt.plot(centroid[0], centroid[1], markersize=10, marker="x", color='k', mew=1)
                    #plt.plot(centroid[0], centroid[1], markersize=10, marker="x", color='m', mew=5)
                    
                    if iter_num>-1:
                        plt.title('Iter : {}'.format(iter_num))
                
                if save:
                    plt.savefig( save_name )
                else:
                    plt.show()
                plt.close()


            # In[43]:

            _samples=sess.run(pred_y, feed_dict={input_x:train_X})
                
            _nearest, _label2id = assign_to_nearest(_samples, centroids)

            _centroids, _sup_cent = estimate_centroids(_samples, _nearest, _label2id, use_supervised=True)


            _id2label = {v: k for k, v in label2id.items()}

            _true_y=train_y
            _near_y=np.array([_id2label[_] for _ in nearest])


            # In[29]:

            # クラスタリング結果(近傍ごとに色分け)
            #plot_clusters(_samples, _centroids, _nearest)


            # In[54]:

            # クラスタリング結果（正解ラベルごとに色分け）
            #plot_clusters(_samples, centroids,  np.array([_label2id[_] for _ in train_y]))


            # In[36]:

            unsupervised = list(set([i for i in range(len(train_X))])-set(supervised))


            # In[53]:

            # ラベルなしクラスタリング結果(近傍ごとに色分け)
            #plot_clusters(_samples[unsupervised], _centroids, _nearest[unsupervised])


            # In[55]:

            # ラベルなしクラスタリング結果(近傍ごとに色分け)
            #plot_clusters(_samples[unsupervised], _centroids, np.array([_label2id[_] for _ in train_y])[unsupervised])


            # ## 正解率評価(近傍と写像後の対応するID)

            # In[61]:

            from sklearn.metrics import precision_recall_fscore_support

            RRR=precision_recall_fscore_support(_true_y[unsupervised], _near_y[unsupervised], average='weighted')

            f = open('../output/main/result.txt','a')
            f.write("Iter:"+str(III))
            f.write("ACC:"+str(RRR))
            f.write('\n')
            f.close()

            # In[50]:

            import matplotlib.pyplot as plt
            import itertools

            def plot_confusion_matrix(cm, classes,
                                      normalize=False,
                                      title='Confusion matrix',
                                      cmap=plt.cm.Blues):
             
                """
                This function prints and plots the confusion matrix.
                Normalization can be applied by setting `normalize=True`.
                """
                if normalize:
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    print("Normalized confusion matrix")
                else:
                    print('Confusion matrix, without normalization')

                print(cm)

                plt.imshow(cm, interpolation='nearest', cmap=cmap)
                plt.title(title)
                plt.colorbar()
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes, rotation=45)
                plt.yticks(tick_marks, classes)

                fmt = '.2f' if normalize else 'd'
                thresh = cm.max() / 2.
                for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(j, i, format(cm[i, j], fmt),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')


            # In[59]:

            from sklearn.metrics import confusion_matrix

            cnf_matrix = confusion_matrix(_true_y[unsupervised],   _near_y[unsupervised])
            np.set_printoptions(precision=2)

            # Plot non-normalized confusion matrix
            #plt.figure()

            import matplotlib as mpl
            mpl.rcParams['font.family'] = 'AppleGothic'
             
            #class_names=['食べる買う','文化芸術スポ-ツ','その他','趣味生活','お祭り','季節のイベント']
            class_names=[v.replace("ー","-").replace("・","") for k,v in {v: k for k, v in label2Label.items()}.items()]
    
            plot_confusion_matrix(cnf_matrix, classes=class_names,
                                  title='Confusion matrix, without normalization')

            _savename='../output/main/confusion_{}.png'.format(III)
            plt.savefig(_savename)
            plt.close()



# In[ ]:
if __name__ == '__main__':
    main()


