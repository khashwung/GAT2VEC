from deepwalk import graph
from gensim.models import Word2Vec
import os
import psutil
import Evaluation
import pandas as pd
from multiprocessing import cpu_count
import random
from Evaluation.Classification import Classification
p = psutil.Process(os.getpid())

'''
GAT2VEC learns an embedding jointly from structural contexts and attribute contexts
employing a single layer of neural network.
'''


class gat2vec(object):
    def __init__(self, dataset, label):
        print("Initializing gat2vec")
        self.dataset = dataset
        self._seed = 1
        # retrieve the dataset dir
        self.dataset_dir = os.getcwd() + "/data/" + dataset + "/"
        # 半监督学习中，有标签样本的比例
        self.TR = [0.1, 0.3, 0.5]
        self.label = label
        # 以下：s是structure，a是attribution的意思
        print("loading structural graph")
        self.Gs = self._get_graph()
        if self.label == False:
            print("loading attribute graph")
            # 加载属性图
            self.Ga = self._get_graph('na')

    '''
    load the adjacency list
    '''
    def _get_graph(self, gtype='graph'):
        fname_struct = self.dataset_dir + self.dataset + '_'+ gtype + '.adjlist'
        print(fname_struct)
        # 利用deepwalk包加载节点相邻图
        G = graph.load_adjacencylist(fname_struct)
        print("Number of nodes: {}".format(len(G.nodes())))
        return G

    '''return random walks '''
    def _get_random_walks(self, G, num_walks, wlength, gtype='graph'):
        # 按照num_walks和wlength这两个参数随机游走路径
        walks = graph.build_deepwalk_corpus(G, num_paths=num_walks, path_length=wlength, alpha=0,
                                        rand=random.Random(self._seed))
        return walks

    ''' filter attribute nodes from walks in attributed graph'''

    def _filter_walks(self,walks, node_num):
        filter_walks = []
        for walk in walks:
            if walk[0] <= node_num:
                # 如果当前节点是在Gs图范围内，并且该节点的walk序列在也Gs图的范围内，才会保留该walk，相当于将节点规范化到Gs范畴内
                fwalks = [nid for nid in walk if int(nid) <= node_num]
                filter_walks.append(fwalks)
        return filter_walks


    ''' Trains jointly attribute contexts and structural contexts'''
    def _train_word2Vec(self, walks, dimension_size, window_size, cores, output, fname):
        print("Learning Representation")
        # gensim的word2vec模型
        model = Word2Vec(walks, size=dimension_size, window=window_size, min_count=0, sg=1,
                         workers=cores)
        if output is True:
            # 保存输出embedding文件，不保存输出模型
            model.wv.save_word2vec_format(fname)
            print("Learned Represenation Saved")
            return fname
        return model


    def train_gat2vec(self, data, nwalks, wlength, dsize, wsize, output):
        print("Random Walks on Structural Graph")
        walks_structure = self._get_random_walks(self.Gs, nwalks, wlength)
        # 图中总共有这么多节点
        num_str_nodes = len(self.Gs.nodes())
        if self.label:
            # 如果是有label，但是无多属性图的数据集，会按照有监督的方式进行训练
            print("Training on Labelled Data")
            gat2vec_model = self.train_labelled_gat2vec(data, walks_structure, num_str_nodes, nwalks, wlength, dsize, wsize, output)
        else:
            # 如果是无label，但是有多属性的数据集，会按照无监督的方式进行训练
            print("------------ATTRIBUTE walk--- ")
            fname = "./embeddings/" + self.dataset + "_gat2vec.emb"
            # 属性图也进行随机游走，这里的随机游走长度为wlength * 2，为啥上面的图不为2倍，这里就变成2倍了
            # 解释：论文中有说，content vertex和attribute vertex的bipartie图中，我们其实只关心content vertex，因此首先
            # random walk会从content vertex出发，其次要想遍历和结构网络同样数量的content vertex，就必须走两倍的路程
            walks_attribute  = self._get_random_walks(self.Ga, nwalks, wlength * 2)
            filter_walks = self._filter_walks(walks_attribute, num_str_nodes)
            # 将两个walk序列合并
            walks = walks_structure + filter_walks
            # 可以看到，无监督本质上是采用word2vec的方式进行了
            gat2vec_model = self._train_word2Vec(walks, dsize, wsize, 8, output, fname)
        return gat2vec_model


    ''' Trains on labelled dataset, i.e class labels are used as an attribute '''

    def train_labelled_gat2vec(self, data, walks_structure, num_str_nodes, nwalks, wlength, dsize, wsize, output,
                               evaluate=True):
        alloutput = pd.DataFrame()
        for tr in self.TR:
            # 文件名。看起来这里面是读取_10,_30,_50这三个文件
            # tr是啥？
            f_ext = "label_" + str(int(tr * 100)) + '_na'
            # walks_attribute, num_atr_nodes = self._get_random_walks(nwalks, wlength * 2, f_ext)
            self.Ga = self._get_graph(f_ext)
            # 这里随机游走的长度变成wlength * 2，因为是实体，属性bipartie图
            walks_attribute = self._get_random_walks(self.Ga, nwalks, wlength * 2)
            filter_walks = self._filter_walks(walks_attribute, num_str_nodes)
            walks = walks_structure + filter_walks
            fname = "./embeddings/" + self.dataset + "_gat2vec_label_" + str(int(tr * 100)) + ".emb"
            gat2vec_model = self._train_word2Vec(walks, dsize, wsize, 8, output, fname)
        return gat2vec_model

    ''' Trains on the bipartite graph only
        只有bipartie图，没有结构图，其实个人感觉这个才是最通用的，因为很多我们构建出的那些结构图也是根据某种行为得到的
        可以看到训练方式和普通的没什么区别，除了没有将结构图的随机walk序列添加进去外。
    '''
    def train_gat2vec_bip(self, data, nwalks, wlength, dsize, wsize, output):
        print("Learning Representation on Bipartite Graph")
        num_str_nodes = len(self.Gs.nodes())
        print("Random Walking...")
        walks_attribute = self._get_random_walks(self.Ga, nwalks, wlength * 2)
        filter_walks = self._filter_walks(walks_attribute, num_str_nodes)
        fname = "./embeddings/" + self.dataset + "_gat2vec_bip.emb"
        gat2vec_model = self._train_word2Vec(filter_walks, dsize, wsize, 8, output, fname)
        return gat2vec_model


    def param_walklen_nwalks(self, param, data, nwalks=10, wlength=80, dsize=128, wsize=5, output=True):
        print("PARAMETER SENSITIVTY ON " + param)
        alloutput = pd.DataFrame()
        # p_value = [40,80,120,160,200]
        # p_value = [5, 10, 15, 20, 25]
        p_value = [1, 5,10,15,20,25]
        walks_st = []
        walks_at = []
        wlength = 80
        # nwalks = 10
        num_str_nodes = len(self.Gs.nodes())
        print("performing joint on both graphs...")
        for nwalks in p_value:
            print(nwalks)
            walks_st.append(self._get_random_walks(self.Gs, nwalks, wlength))
            walks_attribute = self._get_random_walks(self.Ga, nwalks, wlength * 2)
            walks_at.append(self._filter_walks(walks_attribute, num_str_nodes))

        print("Walks finished.... ")
        for i, walks_structure in enumerate(walks_st):
            for j, filter_walks in enumerate(walks_at):
                ps = p_value[i]
                pa = p_value[j]
                print("parameters.... ", ps, pa)
                walks = walks_structure + filter_walks
                fname = "./embeddings/" + self.dataset + "_gat2vec_" + param + "_nwalks_"+ str(ps)+str(pa) + ".emb"
                gat2vec_model = self._train_word2Vec(walks, dsize, wsize, 8, output, fname)
                p = (ps, pa)
                alloutput = self._param_evaluation(data, alloutput, p, param, gat2vec_model)

        print(alloutput)
        alloutput.to_csv(data + "_paramsens_" + param +"_nwalks_" +".csv", index=False)
        return gat2vec_model

    def _param_evaluation(self, data, alloutput, param_val, param_name, model):
        if data == 'blogcatalog':
            multilabel = True
        else:
            multilabel = False
        eval = Classification(data, multilabel)
        outDf = eval.evaluate(model, False)
        outDf['ps'] = param_val[0]
        outDf['pa'] = param_val[1]
        return alloutput.append(outDf)
