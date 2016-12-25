import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time
from word2embeddings.nn.util import zero_value, random_value_normal
from cis.deep.utils.theano import debug_print
import numpy as np
import theano
import theano.tensor as T
import random

from sklearn import metrics

from theano.tensor.signal import downsample
from random import shuffle

from preprocess_20news import load_20news
from common_functions import store_model_to_file, distance_matrix_matrix, load_model_from_file, cosine_matrix_matrix, create_GRU_para_sizeList, load_word2vec_to_init, cosine_tensor3_tensor4, rmsprop, cosine_tensors, Adam, GRU_Batch_Tensor_Input_with_Mask_with_MatrixInit, Conv_with_input_para, LSTM_Batch_Tensor_Input_with_Mask, create_ensemble_para, L2norm_paraList, Diversify_Reg, create_GRU_para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para, load_word2vec
def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, L2_weight=0.00001, max_performance=0.46, Div_reg=0.001, emb_size=50, batch_size=20, maxDocLen=500, class_size=20, hidden_size=20):
    model_options = locals().copy()
    print "model options", model_options

    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results
    rootPath='/mounts/data/proj/wenpeng/Dataset/FB_socher/path/'
    word2vec=load_word2vec()
    word2id,docs,masks, labels =load_20news(maxDocLen, set(word2vec.keys()))  #minlen, include one label, at least one word in the sentence

    train_size=len(docs)
    test_size=train_size
    if train_size != len(labels):
        print 'train_size != len(labels):', train_size, len(labels)
        exit(0)

    vocab_size=  len(word2id)+1 # add one zero pad index
    print 'vocab size:', vocab_size
#     rel_rand_values=rng.normal(0.0, 0.01, (rel_vocab_size, rel_emb_size))   #generate a matrix by Gaussian distribution

    rand_values=random_value_normal((vocab_size, emb_size), theano.config.floatX, rng)
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=rand_values, borrow=True)

    class_values=random_value_normal((class_size, hidden_size), theano.config.floatX, rng)
    class_embeddings=theano.shared(value=class_values, borrow=True)

    #now, start to build the input form of the model

    doc_ids=T.imatrix('path_id_matrix')
    doc_masks=T.fmatrix('path_mask')

#     entity_vocab=T.ivector() #vocab size
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    true_batch_size=doc_ids.shape[0]
    #para
    LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
    NN_para =LSTM_para_dict.values()     #U1 includes 3 matrices, W1 also includes 3 matrices b1 is bias
    
    W = create_ensemble_para(rng, hidden_size, hidden_size)
    params=NN_para+[embeddings, class_embeddings,W]

    docs_input=embeddings[doc_ids.flatten()].reshape((true_batch_size,maxDocLen, emb_size)).dimshuffle(0,2,1) # (batch, hidden, len)
    lstm_layer=LSTM_Batch_Tensor_Input_with_Mask(docs_input, doc_masks,  hidden_size, LSTM_para_dict)
    doc_reps=T.tanh(lstm_layer.output_sent_rep.dot(W))#+T.sum(docs_input, axis=2)#debug_print(lstm_layer.output_sent_rep, 'doc_reps')  # (batch_size, hidden_size)

    #cosine with classes
    class_embeddings=class_embeddings#debug_print(class_embeddings, 'class_embeddings')
#     simi=debug_print(T.nnet.softmax(cosine_matrix_matrix(doc_reps, class_embeddings)),'simi') #(batch, classes)
#     simi_matrix = cosine_matrix_matrix(doc_reps, class_embeddings)
#     simi = simi_matrix/T.sum(simi_matrix, axis=1).dimshuffle(0,'x')
    simi = distance_matrix_matrix(doc_reps, class_embeddings)

    #compute target distributions
    simi_sqr=(simi**2)/T.sum(simi, axis=0).dimshuffle('x',0) #(batch, class_size)
    target_matrix = simi_sqr/T.sum(simi_sqr, axis=1).dimshuffle(0,'x')

#     loss=T.mean(1.0 - T.max(simi, axis=1))
    # loss = -T.mean(T.log(T.max(simi, axis=1)))
    loss = T.mean(T.sum(target_matrix * T.log(target_matrix/simi), axis=1))# + T.mean(1.0  - T.sum(target_matrix*simi, axis=1)/(T.sqrt(T.sum(target_matrix**2, axis=1))*T.sqrt(T.sum(simi**2, axis=1))))
    # loss = T.sum(simi_sqr - simi)
    test_class_ids=T.argmax(simi, axis=1)




#     L2_reg =L2norm_paraList([U1, W1]) #ent_embeddings, rel_embeddings,
    diversify_reg= Diversify_Reg(class_embeddings)

    cost=loss#+Div_reg*diversify_reg

    grads = T.grad(cost, params)    # create a list of gradients for all model parameters
    accumulator=[]
    for para_i in params:
        eps_p=np.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-8)))   #1e-8 is add to get rid of zero division
        updates.append((acc_i, acc))

#     updates=Adam(cost=cost, params=params, lr=learning_rate)

#     grads = T.grad(cost, params)
#     opt = rmsprop(params)
#     updates = opt.updates(params, grads, np.float32(0.01) / np.cast['float32'](batch_size), np.float32(0.9))

    train_model = theano.function([doc_ids, doc_masks], cost, updates=updates,on_unused_input='ignore')

    test_model = theano.function([doc_ids, doc_masks], test_class_ids, on_unused_input='ignore')


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50000000000  # look as this many examples regardless
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False


    n_train_batches=train_size/batch_size
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]
    n_test_batches=test_size/batch_size
    test_remain = test_size%batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-test_remain]


    max_acc=0.0
    max_nmi=0.0
    combined=range(train_size)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        random.shuffle(combined) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
        iter_accu=0
        cost_i=0.0
        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1

            batch_indices=combined[batch_id:batch_id+batch_size]
            #init_heads,path_id_matrix, path_w2v_tensor3, path_mask, target_entities, neg_entities_tensor
            #path_id_matrix, path_w2v_tensor3, path_mask, taret_rel_idlist, target_w2v_matrix, labels



#             rel_idmatrix=[train_paths_store[id] for id in batch_indices]
#             ent_vocab_set=set(range(ent_vocab_size))
            cost_i+= train_model(np.asarray([docs[id] for id in batch_indices], dtype='int32'),
                                 np.asarray([masks[id] for id in batch_indices],dtype=theano.config.floatX))
#                                       neg_entity_tensor(ent_idmatrix, rel_idmatrix, tuple2tailset, neg_size, ent_vocab_set))

            #after each 1000 batches, we test the performance of the model on all test data
            if iter%50==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter_accu), 'uses ', (time.time()-past_time)/60.0, 'min'
                print 'Testing...'
                past_time = time.time()

                pred_labels=[]
                for test_batch_id in test_batch_start: # for each test batch
                    batch_labels=test_model(np.asarray(docs[test_batch_id:test_batch_id+batch_size], dtype='int32'),
                                                  np.asarray(masks[test_batch_id:test_batch_id+batch_size],dtype=theano.config.floatX))

                    pred_labels+=list(batch_labels)
                if len(pred_labels) != len(labels):
                    print 'len(pred_labels) != len(labels):', len(pred_labels), len(labels)
                    exit(0)
#                 print labels[:100]
                # print 'pred_labels:', pred_labels[:100]
                NMI=metrics.normalized_mutual_info_score(labels, pred_labels)



                if NMI > max_nmi:
                    max_nmi=NMI
#                     if max_acc > max_performance:
#                         store_model_to_file(rootPath+'Best_Paras_arc3_task1_gru_twoinit_remHis_'+str(max_acc), params)
#                         print 'Finished storing best  params at:', max_acc
                print 'current NMI:', NMI, '\t\t\t\t\tmax NMI:', max_nmi





            if patience <= iter:
                done_looping = True
                break

        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()
    print('Optimization complete.')

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))





if __name__ == '__main__':
    evaluate_lenet5()
