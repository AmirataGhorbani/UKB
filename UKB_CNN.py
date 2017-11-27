import numpy as np
import tensorflow as tf
import random
from amirata_functions import *
import scipy
import scipy.stats as stats
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True
    
    
class UKB_CNN(object):
    
    def __init__(self, name, num_length, cat_length, num_classes, path,
                 hidden_sizes, task_weights=1,num_output_layers=0,output_hidden_sizes=None,learning_rate = 0.001, dropout = 1, class_weight=1,
                 activation="relu", reg = 0, batch_norm=True, loss = "weighted_CE", num_tasks=1,
                activation_param = 0):
        self.activation_param = activation_param
        self.num_tasks = num_tasks
        self.task_weights = [task_weights] * num_tasks if type(task_weights)!=list else task_weights
        self.class_weight = class_weight if type(class_weight[0])==list else [class_weight]*num_classes
        self.name = name 
        self.num_forward_layers = len(hidden_sizes)
        self.num_output_layers = [num_output_layers]*num_tasks if type(num_output_layers)!=list else num_output_layers
        self.batch_norm = batch_norm
        if output_hidden_sizes is not None:
            if type(output_hidden_sizes)==int:
                self.output_hidden_sizes = [[output_hidden_sizes]]*num_tasks if self.num_output_layers[0]==1 else None
            elif type(output_hidden_sizes)==list:
                if type(output_hidden_sizes[0])==list and len(output_hidden_sizes)==num_tasks:
                    self.output_hidden_sizes = output_hidden_sizes
                elif type(output_hidden_sizes[0])==int:
                    self.output_hidden_sizes = [output_hidden_sizes]*num_tasks
                else:
                    raise ValueError("Error in output_hidden_sizes!")
            else:
                raise ValueError("output_hidden_sizes should be a list or list of lists!")
            
                
        self.num_length = num_length 
        self.cat_length = cat_length 
        self.epsilon=tf.constant(1e-8)
        self.learning_rate = learning_rate
        self.hidden_sizes = [hidden_sizes]*num_forward_layers if type(hidden_sizes)!=list else hidden_sizes
        self.dropout = dropout
        self.num_classes = [num_classes]*num_tasks if type(num_classes)!=list else num_classes
        self.path = path
        self.reg = reg
        self.define_placeholders()
        self.activation = self.get_activation(activation)
        input_data = tf.cond(self.hessian_ph,lambda:tf.reshape(self.flat_input_ph,[1,self.num_length+self.cat_length]),
                             lambda:self.input_ph)

        self.build(input_data, self.output_ph, self.learning_rate, loss)
    
    def define_placeholders(self):
        output_shape = [None,] if self.num_tasks==1 else [None,self.num_tasks] 
        input_shape = [None, self.num_length+self.cat_length]
        self.dropout_ph = tf.placeholder_with_default(tf.constant(1.),shape=())
        self.output_ph = tf.placeholder(dtype= tf.int32, shape=output_shape)
        self.is_training_ph = tf.placeholder_with_default(tf.constant(False),shape=())
        self.input_ph=tf.placeholder(dtype=tf.float32, shape = input_shape)
        self.input = self.input_ph
        self.hessian_ph = tf.placeholder_with_default(tf.constant(False),shape=())
        self.hessian_size = self.num_length+self.cat_length
        self.flat_input_ph = tf.placeholder_with_default(tf.constant(np.zeros(self.hessian_size),dtype=tf.float32),
                                                           shape = (self.hessian_size,))
        return
    
    def make_Pdic(self, order='euclidean'):
        init_W = tf.contrib.layers.xavier_initializer()
        init_b = tf.zeros_initializer()
        flat_length = self.num_length+self.cat_length
        prev_size = self.hidden_sizes[-1]
        norm_power = 2 if (order=='euclidean') else 1
        Pdic = {}
        sum_weights = tf.constant(0.,dtype=tf.float32)
        for mishmoram in range(self.num_tasks):
            for layer in range(self.num_output_layers[mishmoram]):
                next_size = self.output_hidden_sizes[mishmoram][layer]
                Pdic["H_{}_{}".format(mishmoram,layer)] = tf.get_variable\
                (self.name+"H_{}_{}".format(mishmoram,layer), shape =[prev_size,next_size],initializer=init_W)
                Pdic["v_{}_{}".format(mishmoram,layer)] = tf.get_variable\
                (self.name+"v_{}_{}".format(mishmoram,layer), shape=[next_size],initializer=init_b)
                sum_weights += tf.norm(Pdic["H_{}_{}".format(mishmoram,layer)],order)**norm_power
                prev_size = next_size
            Pdic["H_{}".format(mishmoram)] = tf.get_variable\
            (self.name+"H_{}".format(mishmoram), shape =[prev_size,self.num_classes[mishmoram]],initializer=init_W)
            Pdic["v_{}".format(mishmoram)] = tf.get_variable\
            (self.name+"v_{}".format(mishmoram), shape=[self.num_classes[mishmoram]],initializer=init_b)
            sum_weights += tf.norm(Pdic["H_{}".format(mishmoram)],order)**norm_power
        prev_size = flat_length
        for layer in range(self.num_forward_layers):
            next_size = self.hidden_sizes[layer]
            Pdic["W{}".format(layer)] = tf.get_variable\
            (self.name+"W{}".format(layer),shape=[prev_size,next_size],initializer=init_W)
            Pdic["b{}".format(layer)] = tf.get_variable\
            (self.name+"b{}".format(layer),shape=[next_size],initializer=init_b)
            sum_weights += tf.norm(Pdic["W{}".format(layer)],order)**norm_power
            prev_size = next_size
        return Pdic,sum_weights
        
    def get_activation(self, name):
        if name == "relu":
            return tf.nn.relu
        elif name == "softplus":
            return tf.nn.softplus
        elif name == "tanh":
            return tf.nn.tanh
        elif name == "sigmoid":
            return tf.sigmoid
        elif name == "elu":
            return tf.nn.elu
        elif name == "linear":
            return self.my_linear
        elif name == "my_relu":
            return self.my_relu
        elif name == "my_tanh":
            return self.my_tanh
        else:
            raise RuntimeError("Invalid activation function!")
    def my_linear(self,x):
        return x
    def my_tanh(self,x):
        return tf.nn.tanh(x*self.activation_param)/self.activation_param
    def my_relu(self,x):
        return tf.maximum(x,self.activation_param*x)
    
    def fc_layer(self, W, b, feed, activation):
        out = tf.matmul(feed,W)+b
        self.layers.append(out)
        self.layer_names.append("Dense")
        out_activated = out if activation is None else activation(out)
        self.layers.append(out_activated)
        self.layer_names.append("Activation")
        if self.batch_norm:
            self.layers.append(tf.layers.batch_normalization(self.layers[-1],axis=-1,training=self.is_training_ph))
            self.layer_names.append("BatchNorm")
        out_dropped = tf.nn.dropout(self.layers[-1], self.dropout_ph)
        self.layers.append(out_dropped)
    
    def network(self, feed):
        self.layers = [feed]
        self.layer_names = []
        flat_length = self.num_length+self.cat_length
        self.layers.append(tf.reshape(self.layers[-1], shape=[-1, flat_length]))
        for layer in range(self.num_forward_layers):
            W = self.Pdic["W{}".format(layer)]
            b = self.Pdic["b{}".format(layer)]
            self.fc_layer(W, b, self.layers[-1], self.activation)
        return self.layers[-1]
    
    def out_network(self,feed,task):
        layers = [feed]
        for layer in range(self.num_output_layers[task]):
            H = self.Pdic["H_{}_{}".format(task,layer)]
            v = self.Pdic["v_{}_{}".format(task,layer)]
            layers.append(self.fc_layer(H, v, layers[-1], self.activation))
        return tf.matmul(layers[-1],self.Pdic["H_{}".format(task)])+self.Pdic["v_{}".format(task)]
    
    def build(self, feed, out, learning_rate, loss):
        self.Pdic, sum_weights = self.make_Pdic()
        self.hidden = self.network(feed)
        if self.num_tasks==1:
            self.out = tf.matmul(self.hidden,self.Pdic["H_0"])+self.Pdic["v_0"]
        else:
            self.out=[]
            for i in range(self.num_tasks):
                self.out.append(self.out_network(self.hidden,i))
        self.output = self.out
        self.dic = {}
        self.dic["f1"] = tf.constant(0.,dtype=tf.float32)
        self.dic["cost"] = 0.5*self.reg*(sum_weights)
        if self.num_tasks==1:
            self.dic["precision"],self.dic["recall"],self.dic["f1"],self.dic["accuracy"],  cost=\
            self.accuracy_cost(self.out, self.output_ph, loss)
            self.dic["cost"]+=cost
        elif self.num_tasks>1:
            for i in range(self.num_tasks):
                (self.dic["precision_{}".format(i)],self.dic["recall_{}".format(i)],\
                self.dic["f1_{}".format(i)],self.dic["accuracy_{}".format(i)], cost) =\
                self.accuracy_cost(self.out[i], self.output_ph[:,i], loss)
                self.dic["cost"] += self.task_weights[i] * cost
                self.dic["f1"] += self.dic["f1_{}".format(i)]/self.num_tasks
        else:
            raise ValueError("Invalid number of tasks!")
            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):            
            self.dic["optmz"]= tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.dic["cost"])                                         
        return   
    
    def accuracy_cost(self, feed, true_label, loss="CE", task=0):
        out_label = tf.cast(tf.argmax(feed,1),tf.int32)
        one_hots = tf.one_hot(true_label, self.num_classes[task])
        if loss == "CE":
            cost=tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hots,logits=feed))
        elif loss == "L2":
            cost = tf.nn.l2_loss(feed-one_hots)
        elif loss =="weighted_CE":
            weight = self.class_weight[task][0]*tf.cast(tf.equal(true_label,0),tf.float32)
            for i in range(1,self.num_classes[task]):
                weight+=self.class_weight[task][i]*tf.cast(tf.equal(true_label,i),tf.float32)
            cost=tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hots,logits=feed,weights=weight))
        else:
            raise ValueError("Invalid Loss Type!")
        corrects = tf.cast(tf.equal(out_label, true_label),tf.float32)
        pos_intersect = tf.cast(out_label*true_label,tf.float32)
        recall = (tf.reduce_sum(pos_intersect)+self.epsilon)/(tf.reduce_sum(tf.cast(true_label,tf.float32))+self.epsilon)
        precision = (tf.reduce_sum(pos_intersect)+self.epsilon)/(tf.reduce_sum(tf.cast(out_label,tf.float32))+self.epsilon)
        f1 = 2/(1/recall+1/precision)
        accuracy = tf.reduce_mean(corrects)    
        self.pos_intersect=pos_intersect
        return precision,recall,f1,accuracy,cost     
    
    def optimize(self, sess, training_data, validation_data,
                 save, load , epochs=10,batch_size= 200,
                 tradeoff=False, verbose=True, save_always=False,
                 return_as_fit=False, early_stopping=0, initialize=True,report_every=1,measure="accuracy"):
        X_train1 = training_data[0]
        X_train2 = training_data[1]
        Y_train = training_data[-1]
        X_val = np.concatenate([validation_data[0],validation_data[1]],-1)
        Y_val = validation_data[-1]
        if len(Y_train.shape)==2 and Y_train.shape[1]==1:
            Y_train=Y_train[:,0]
        if len(Y_val.shape)==2 and Y_val.shape[1]==1:
            Y_val=Y_val[:,0]
        saver = tf.train.Saver()
        if initialize:
            sess.run(tf.global_variables_initializer())
        self.best_val = 0.
        best_cost = 1e8
        consecutive = 0
        self.cost_hist=([],[])
        if load:
            print("Loading model from :{}".format(self.path)) if verbose else None
            saver.restore(sess,self.path)
        self.best_val = sess.run(self.dic[measure], feed_dict={self.input_ph:X_val,self.output_ph:Y_val})
        val1 = self.best_val 
        print("validation {} before starting".format(measure),self.best_val) if verbose else None
        for epoch in range(epochs):
            batches = self.get_batches(Y_train.shape[0], batch_size)
            acc_val, acc_train, f1_val, f1_train, val_cost, tr_cost =self.do_epoch\
            (sess,epoch,batches,X_train1,X_train2,Y_train,X_val,Y_val,verbose,report_every=1)  
            self.cost_hist[0].append(tr_cost)
            self.cost_hist[1].append(val_cost)
            if measure=="accuracy":
                if self.save_weights(sess,save,saver,save_always,acc_val,acc_train,verbose,consecutive,early_stopping):
                    return self.best_val
            elif measure=="f1":
                if self.save_weights(sess,save,saver,save_always,f1_val,f1_train,verbose,consecutive,early_stopping):
                    return self.best_val
            else:
                raise ValueError("Invalid measure!")
        if self.best_val>val1:
            if epochs:
                saver.restore(sess,self.path)
            else:
                if save and not load:
                    saver.save(sess,self.path)
        print("Best {}:".format(measure),self.best_val)
        self.best_f1 = sess.run(self.dic["f1"], feed_dict={self.input_ph:X_val,self.output_ph:Y_val})
        self.best_acc = sess.run(self.dic["accuracy"], feed_dict={self.input_ph:X_val,self.output_ph:Y_val})
        return self.best_val
    
    def save_weights(self,sess,save,saver,save_always,measure_val,measure_train,verbose,consecutive,early_stopping):
        if save_always:
            if measure_train>self.best_val:
                self.best_val = measure_train
                best_cost = tr_cost
                saved_path = saver.save(sess, self.path)
                if self.best_val>0.999 and return_as_fit:
                    return True
        else:
            if measure_val>self.best_val: #FIXIT
                self.best_val = measure_val
                if save:
                    saved_path = saver.save(sess, self.path)
                consecutive = 0
                if verbose:
                    print("New best!")
            else:
                consecutive += 1
                if consecutive>early_stopping and early_stopping:
                    print("Early Stopping!")
                    return True
        
    def get_batches(self, data_size, batch_size):   
        mask = np.random.permutation(data_size)
        batches = [mask[k:k + batch_size] for k in range(0, data_size, batch_size)]
        return batches

    def backpropagate(self, sess, X_batch, Y_batch):
        sess.run(self.dic["optmz"], feed_dict={self.input_ph:X_batch,self.output_ph:Y_batch,
                                               self.is_training_ph:True,
                                               self.dropout_ph:self.dropout})
        
    def do_epoch(self, sess, epoch, batches,  X_train1, X_train2, Y_train,
                          X_val, Y_val, verbose, report_every=1):
        mskn = np.random.choice(range(X_train1.shape[0]),len(X_val))
        x_train = np.concatenate([X_train1[mskn],X_train2[mskn]],-1)
        y_train = Y_train[mskn]
        for batch in batches:
            X_batch = np.concatenate([X_train1[batch],X_train2[batch]],-1)
            self.backpropagate(sess, X_batch, Y_train[batch]) 
        if not epoch%report_every:
            acc_val, acc_train, f1_val, f1_train, val_cost, tr_cost = \
            self.report_results(sess,epoch,X_val,Y_val,x_train,y_train,verbose)
        return acc_val, acc_train, f1_val, f1_train, val_cost, tr_cost
    
    def report_results(self,sess,epoch,X_val,Y_val,x_train,y_train,verbose):
        if self.num_tasks==1:
            precision_val,recall_val,f1_val,acc_val, val_cost =\
            sess.run([self.dic["precision"],self.dic["recall"],self.dic["f1"],self.dic["accuracy"],self.dic["cost"]],
                                     feed_dict={self.input_ph:X_val,self.output_ph:Y_val})
            precision_train,recall_train,f1_train,acc_train, tr_cost =\
            sess.run([self.dic["precision"],self.dic["recall"],self.dic["f1"],self.dic["accuracy"],self.dic["cost"]],
                     feed_dict={self.input_ph:x_train,self.output_ph:y_train})
        
            print('''_______________________________________________________________________\n
            Epoch:{}\nVal/Train Accuracy:{}/{}\nRecall:{}\nPrecision:{}\nF1:{}
            Val/Train Cost:{}/{}'''.format(epoch,acc_val,acc_train,recall_val,precision_val,f1_val,
                                           val_cost,tr_cost)) if verbose else None
            return acc_val, acc_train, f1_val, f1_train, val_cost, tr_cost
        elif self.num_tasks>1:
            f1_val_tot,f1_train_tot = 0,0
            for i in range(self.num_tasks):
                precision_val,recall_val,f1_val,acc_val, val_cost =\
                sess.run([self.dic["precision_{}".format(i)],self.dic["recall_{}".format(i)],\
                          self.dic["f1_{}".format(i)],self.dic["accuracy_{}".format(i)],self.dic["cost"]],
                         feed_dict={self.input_ph:X_val,self.output_ph:Y_val})
                f1_val_tot += f1_val
                precision_train,recall_train,f1_train,acc_train, tr_cost =\
                sess.run([self.dic["precision_{}".format(i)],self.dic["recall_{}".format(i)],\
                          self.dic["f1_{}".format(i)],self.dic["accuracy_{}".format(i)],self.dic["cost"]],
                         feed_dict={self.input_ph:x_train,self.output_ph:y_train})
                f1_train_tot += f1_train
                print('''_________________________Epoch:{},Results for output:{}_______________________________\n
                \nVal/Train Accuracy:{}/{}\nRecall:{}\nPrecision:{}\nF1:{}
                Val/Train Cost:{}/{}'''.format(epoch,i,acc_val,acc_train,recall_val,precision_val,f1_val,
                                               val_cost,tr_cost)) if verbose else None
            return f1_val_tot/self.num_tasks,f1_train_tot/self.num_tasks, val_cost, tr_cost
        else:
            raise ValueError("Invalid Number of Tasks!")
        
    def predict(self, feed, sess=None):
        if sess is None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path) 
        if len(feed.shape)==3:
            in_img = np.expand_dims(feed,0)
        elif len(feed.shape)==4:
            in_img = feed
        else:
            raise RuntimeError("Invalid Input Shape!")
        output = sess.run(tf.nn.softmax(self.out), {self.input_ph:in_img})
        return output
    
    def scores(self, feed, sess=None):
        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)  
        output = sess.run\
        (self.out, {self.input_ph: feed})
        return output

    def load_model(self,path,sess,saved_net_name="",init=False):
        saved_vars_names = []
        for v in tf.contrib.framework.list_variables(path):
            if v[0][:len(saved_net_name)] == saved_net_name:
                saved_vars_names.append(v[0][len(saved_net_name):])
            else:
                saved_vars_names.append(v[0])
        vars_to_load = [var for var in tf.global_variables() if ((var.name[len(self.name):-2] in saved_vars_names\
                       and var.name[:len(self.name)]==self.name) or var.name[:-2] in saved_vars_names)]
        vars_to_load_names = []
        for var in vars_to_load:
            if var.name[len(self.name):-2] in saved_vars_names:
                vars_to_load_names.append(var.name[len(self.name):-2])
            elif var.name[:-2] in saved_vars_names:
                 vars_to_load_names.append(var.name[:-2])
            else:
                raise RuntimeError("bug!")
        load_dict = dict(zip(vars_to_load_names,vars_to_load))
        if init:
            init = tf.global_variables_initializer()
            sess.run(init)
        saver=tf.train.Saver(load_dict)
        saver.restore(sess,path)      
        return vars_to_load

