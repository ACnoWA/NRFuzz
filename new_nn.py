#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import math
import time
import keras
import random
import socket
import subprocess
import numpy as np
import tensorflow as tf
import keras.backend as K
from collections import Counter
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"]="0"
HOST = '127.0.0.1'
PORT = 12012

MAX_FILE_SIZE = 10000
MAX_BITMAP_SIZE = 2000
round_cnt = 0
# Choose a seed for random initilzation
# seed = int(time.time())
seed = 12
np.random.seed(seed)
random.seed(seed)
set_random_seed(seed)
seed_list = glob.glob('./seeds/*')
new_seeds = glob.glob('./seeds/id_*')
SPLIT_RATIO = len(seed_list)
# get binary argv
argvv = sys.argv[1:]


# process training data from afl raw data
def process_data():
    global MAX_BITMAP_SIZE
    global MAX_FILE_SIZE
    global SPLIT_RATIO
    global seed_list
    global new_seeds
    global output_layer_edges

    # shuffle training samples
    seed_list = glob.glob('./seeds/*')
    seed_list.sort()
    SPLIT_RATIO = len(seed_list)
    rand_index = np.arange(SPLIT_RATIO)
    np.random.shuffle(seed_list)
    new_seeds = glob.glob('./seeds/id_*')

	near_edge = {}

    call = subprocess.check_output

    # get MAX_FILE_SIZE
    cwd = os.getcwd()
    max_file_name = call(['ls', '-S', cwd + '/seeds/']).decode('utf8').split('\n')[0].rstrip('\n')
    MAX_FILE_SIZE = os.path.getsize(cwd + '/seeds/' + max_file_name)

    # create directories to save label, spliced seeds, variant length seeds, crashes and mutated seeds.
    os.path.isdir("./bitmaps/") or os.makedirs("./bitmaps")
    os.path.isdir("./splice_seeds/") or os.makedirs("./splice_seeds")
    os.path.isdir("./vari_seeds/") or os.makedirs("./vari_seeds")
    os.path.isdir("./crashes/") or os.makedirs("./crashes")
    os.path.isdir("./seeds_exe_path/") or os.makedirs("./seeds_exe_path")
    os.path.isdir("./select/") or os.makedirs("./select")
    os.path.isdir("./edge_info/") or os.makedirs("./edge_info")
    os.path.isdir("./ind_info/") or os.makedirs("./ind_info")

    # load near edges information into dictionary near_edge
    fl_n = open("./NearedgeInfo.txt", "r")
    contents = f.readlines()
    for item in contents:
		tmp = item.split(b' ')
		near_edge[tmp[0]] = tmp[1:-1]
	fl_n.close()

    # obtain raw bitmaps
    raw_bitmap = {}
	eg2seed_dict = {}
    tmp_cnt = []
    out = ''
    print argvv
    for f in seed_list:
        tmp_list = []
		tmp = []
        try:
            # append "-o tmp_file" to strip's arguments to avoid tampering tested binary.
            # 注意这两个call语句一定不能混用，会导致错误结果的
            if argvv[0] == './strip':
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', 'none', '-t', '500'] + argvv + [f] + ['-o', 'tmp_file'])
            else:
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', 'none', '-t', '500'] + argvv + [f])
        except subprocess.CalledProcessError:
            print("find a crash")
        for line in out.splitlines():
            edge = line.split(b':')[0]
            # 去除前导０
            edge = edge.lstrip("0")
            tmp_cnt.append(edge)
            tmp_list.append(edge)
	
		#record seeds near edge
		for item in tmp_list:
			if(near_edge.get(edge, 'None') == 'None'):
				continue
			else:
				tmp.extend(near_edge[edge])
		for it in tmp:
			if eg2seed_dict.has_key(it):
				eg2seed_dict.append(f)
			else:
				eg2seed_dict[it] = [f]

		'''
        file_name = "./seeds_exe_path/" + f.split('/')[-1] + ".txt"
        file_f = open(file_name, 'w')
        for i in tmp_list:
            file_f.write(i)
            file_f.write('\n')
        file_f.close()
		'''
        raw_bitmap[f] = tmp_list
    
	fl_n = open("./eg2sd.txt", 'w')
	fl_n.write(str(eg2seed_dict))
	fl_n.close()
	
    counter = Counter(tmp_cnt).most_common()
    tmp_cnt.sort()
    file_f = open("./all_exePath.txt", 'w')
    for i in tmp_cnt:
        file_f.write(i)
        file_f.write('\n')
    file_f.close()
    # save bitmaps to individual numpy label
    label = [int(f[0]) for f in counter]
    print('the len of label is {}'.format(len(label)))
    bitmap = np.zeros((len(seed_list), len(label)))
    for idx, i in enumerate(seed_list):
        tmp = raw_bitmap[i]
        for j in tmp:
            if int(j) in label:
                bitmap[idx][label.index((int(j)))] = 1

    # label dimension reduction
    fit_bitmap, res_index = np.unique(bitmap, axis=1, return_index=True)
    #get which edges the final output layer is
    output_layer_edges = [label[i] for i in res_index]
    np.savetxt("output_edges.txt", output_layer_edges)

    print("data dimension" + str(fit_bitmap.shape))

    # save training data
    MAX_BITMAP_SIZE = fit_bitmap.shape[1]
    for idx, i in enumerate(seed_list):
        file_name = "./bitmaps/" + i.split('/')[-1]
        np.save(file_name, fit_bitmap[idx])


# training data generator
def generate_training_data(lb, ub):
    seed = np.zeros((ub - lb, MAX_FILE_SIZE))
    bitmap = np.zeros((ub - lb, MAX_BITMAP_SIZE))
    for i in range(lb, ub):
        tmp = open(seed_list[i], 'rb').read()
        ln = len(tmp)
        if ln < MAX_FILE_SIZE:
            tmp = tmp + (MAX_FILE_SIZE - ln) * b'\x00'
        seed[i - lb] = [j for j in bytearray(tmp)]

    for i in range(lb, ub):
        file_name = "./bitmaps/" + seed_list[i].split('/')[-1] + ".npy"
        bitmap[i - lb] = np.load(file_name)
    return seed, bitmap


# learning rate decay
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.7
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print(step_decay(len(self.losses)))


# compute jaccard accuracy for multiple label
def accur_1(y_true, y_pred):
    y_true = tf.round(y_true)
    pred = tf.round(y_pred)
    summ = tf.constant(MAX_BITMAP_SIZE, dtype=tf.float32)
    wrong_num = tf.subtract(summ, tf.reduce_sum(tf.cast(tf.equal(y_true, pred), tf.float32), axis=-1))
    right_1_num = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(y_true, tf.bool), tf.cast(pred, tf.bool)), tf.float32), axis=-1)
    return K.mean(tf.divide(right_1_num, tf.add(right_1_num, wrong_num)))


def train_generate(batch_size):
    global seed_list
    while 1:
        np.random.shuffle(seed_list)
        # load a batch of training data
        for i in range(0, SPLIT_RATIO, batch_size):
            # load full batch
            if (i + batch_size) > SPLIT_RATIO:
                x, y = generate_training_data(i, SPLIT_RATIO)
                x = x.astype('float32') / 255
            # load remaining data for last batch
            else:
                x, y = generate_training_data(i, i + batch_size)
                x = x.astype('float32') / 255
            yield (x, y)


# get vector representation of input
def vectorize_file(fl):
    seed = np.zeros((1, MAX_FILE_SIZE))
    tmp = open(fl, 'rb').read()
    ln = len(tmp)
    if ln < MAX_FILE_SIZE:
        tmp = tmp + (MAX_FILE_SIZE - ln) * b'\x00'
    seed[0] = [j for j in bytearray(tmp)]
    seed = seed.astype('float32') / 255
    return seed


# splice two seeds to a new seed
def splice_seed(fl1, fl2, idxx):
    tmp1 = open(fl1, 'rb').read()
    ret = 1
    randd = fl2
    while ret == 1:
        tmp2 = open(randd, 'rb').read()
        if len(tmp1) >= len(tmp2):
            lenn = len(tmp2)
            head = tmp2
            tail = tmp1
        else:
            lenn = len(tmp1)
            head = tmp1
            tail = tmp2
        f_diff = 0
        l_diff = 0
        for i in range(lenn):
            if tmp1[i] != tmp2[i]:
                f_diff = i
                break
        for i in reversed(range(lenn)):
            if tmp1[i] != tmp2[i]:
                l_diff = i
                break
        if f_diff >= 0 and l_diff > 0 and (l_diff - f_diff) >= 2:
            splice_at = f_diff + random.randint(1, l_diff - f_diff - 1)
            head = list(head)
            tail = list(tail)
            tail[:splice_at] = head[:splice_at]
            with open('./splice_seeds/tmp_' + str(idxx), 'wb') as f:
                f.write(bytearray(tail))
            ret = 0
        print(f_diff, l_diff)
        randd = random.choice(seed_list)


# compute gradient for given input
def gen_adv2(index, seed_file, model, layer_list, idxx, splice):
    adv_list = []
    loss = layer_list[-2][1].output[:, index]
    grads = K.gradients(loss, model.input)[0]
    iterate = K.function([model.input], [loss, grads])

	fls_len = 2
	while seed_file[0] == seed_file[1]:
		seed_file[1] = random.chioce(seed_list)

	for ind in range(fls_len):
		x = vectorize_file(seed_file)
    	loss_value, grads_value = iterate([x])
    #np.argsort()将数组中的元素从小到大排列，返回其对应的index; axis=1表示按行排列，即对每一个行向量中的元素进行排列
    idx = np.flip(np.argsort(np.absolute(grads_value), axis=1)[:, -MAX_FILE_SIZE:].reshape((MAX_FILE_SIZE,)), 0)
    val = np.sign(grads_value[0][idx])
    adv_list.append((idx, val, seed_file))

    return adv_list

# compute gradient for given input without sign
def gen_adv3(index, seed_file, model, layer_list, idxx, splie):
    adv_list = []
    loss = layer_list[-2][1].output[:, index]
    grads = K.gradients(loss, model.input)[0]
    iterate = K.function([model.input], [loss, grads])
	
	fls_len = 2
	while seed_file[0] == fl[1]:
		seed_file[1] = random.chioce(seed_list)
	
	for ind in range(fls_len):
    	x = vectorize_file(seed_file)
    	loss_value, grads_value = iterate([x])
    	idx = np.flip(np.argsort(np.absolute(grads_value), axis=1)[:, -MAX_FILE_SIZE:].reshape((MAX_FILE_SIZE,)), 0)
    	val = np.random.choice([1, -1], MAX_FILE_SIZE, replace=True)
    	adv_list.append((idx, val, fl[index]))

    return adv_list


# grenerate gradient information to guide furture muatation
def gen_mutate2(model, edge_num, sign):
    call = subprocess.check_output
    exe_path = ''
    edge_list = []
    edge_info = {}
   

    # select seeds
    print("#######debug" + str(round_cnt))
    if round_cnt == 0:
        new_seed_list = seed_list
    else:
        new_seed_list = new_seeds
   
    #select seeds to record nearby edges on their execution path
    if len(seed_list) < edge_num:
        rand_seed = [seed_list[i] for i in np.random.choice(len(seed_list), edge_num, replace=True)]
    else:
        rand_seed = [seed_list[i] for i in np.random.choice(len(seed_list), edge_num, replace=False)]
    
    # function pointer for gradient computation
    fn = gen_adv2 if sign else gen_adv3

    # select output neurons to compute gradient
    interested_indice = np.random.choice(MAX_BITMAP_SIZE, edge_num)
	interester_edge = [output_layer_edges[i] for i in interested_indice]
		
    layer_list = [(layer.name, layer) for layer in model.layers]

    fn_n = open('eg2seed.txt', 'r')
	str_dict = f.read()
	eg2seed = eval(str_dict)
	fn_n.close()

	with open('gradient_info_p', 'w') as f:
		for idxx in range(len(interested_indice[:])):
			if idxx % 100 == 0:
				del model
				K.clear_session()
				model = build_model()
				model.load_weights('hard_label.h5')
    			layer_list = [(layer.name, layer) for layer in model.layers]
			
			ind = int(interested_indice[idxx])
			ind_edge = output_layer_edge[ind]
			if eg2seed.has_key(ind_edge):
				seed_ls = eg2seed[ind_edge]
				ls_len = len(seed_ls)
				if ls_len > 2:
					seed_fl = sample(seed_ls, 2)
				else if ls_len < 2:
					seed_fl = seed_ls
					seed_fl.append(choice(seed_list))
				else:
					seed_fl = seed_ls
			else:
				print("this edge not in dict\n")
				seed_fl = sample(seed_list, 2)

            adv_list = fn(ind, seed_fl, model, layer_list, idxx, 1)
            tmp_list.append(adv_list)    
			for ele in adv_list:
            	ele0 = [str(el) for el in ele[0]]
                ele1 = [str(int(el)) for el in ele[1]]
                ele2 = ele[2]
                f.write(",".join(ele0) + '|' + ",".join(ele1) + '|' + ele2 + "\n")
    

def build_model():
    batch_size = 32
    num_classes = MAX_BITMAP_SIZE
    epochs = 50

    model = Sequential()
    model.add(Dense(4096, input_dim=MAX_FILE_SIZE))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    opt = keras.optimizers.adam(lr=0.0001)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[accur_1])
    model.summary()

    return model


def train(model):
    loss_history = LossHistory()
    lrate = keras.callbacks.LearningRateScheduler(step_decay)
    callbacks_list = [loss_history, lrate]
    model.fit_generator(train_generate(16),
                        steps_per_epoch=(SPLIT_RATIO / 16 + 1),
                        epochs=50,
                        verbose=1, callbacks=callbacks_list)
    # Save model and weights
    model.save_weights("hard_label.h5")


def gen_grad(data):
    global round_cnt
    t0 = time.time()
    process_data()
    model = build_model()
    train(model)
    # model.load_weights('hard_label.h5')
    gen_mutate2(model, 5, data[:5] == b"train")
    round_cnt = round_cnt + 1
    print(time.time() - t0)


def setup_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    conn, addr = sock.accept()
    print('connected by neuzz execution moduel ' + str(addr))
    gen_grad(b"train")
    conn.sendall(b"start")
    while True:
        data = conn.recv(1024)
        if not data:
            break
        else:
            gen_grad(data)
            conn.sendall(b"start")
    conn.close()


if __name__ == '__main__':
    setup_server()
