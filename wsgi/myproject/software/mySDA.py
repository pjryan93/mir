import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from LogesticRegression import LogisticRegression, load_data
from mlp import HiddenLayer
from da import dA

class customSda:
	def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):
    self.sigmoid_layers = []
    self.dA_layers = []
    self.params = []
    self.n_layers = len(hidden_layers_sizes)
    self.notes = {"C":1.0,
                  "C#":2.0,
                  "D":3.0,
                  "D#":4.0,
                  "E":5.0,
                  "F":6.0,
                  "F#":7.0,
                  "G":8.0,
                  "G#":9.0,
                  "A":10.0,
                  "A#":11.0,
                  "B":12.0,
                  "B#":13.0,
                  "N":14.0
                }

    def getDataSet(inds, sIn, eIn):
        sIn = int(sIn)
        eIn = int(eIn)
        aud = ess.MonoLoader(filename="/Users/patrickryan/cdev/project/siteFolder/wsgi/myproject/software/data/whilemyguitar.wav")()
        fftData = list()

        for i in range(sIn,eIn):
            s1 = inds[i]
            s2 = round(inds[i+1])
            data = getFFT(aud,s1,s2)
            fftData.append(data[0:15000])
        vals = loadCorrect()
        chordsNumbs = list()
        for i in range(sIn,eIn):
            chordsNumbs.append(self.notes[vals[i]])
        dataSet = (fftData,chordsNumbs)
        retValue_X ,retValue_Y =  shared_dataset2(dataSet)
        return (retValue_X,retValue_Y)
    def getAllDatasets(songId):
        datab = dataBase()
        sng = datab.getSong(songId)
        segs = datab.getSegments(songId)
        inds = zeros(len(segs))
        for i in range(0,len(segs)-2):
            inds[i] = int(segs[i][2])
        aud = ess.MonoLoader(filename="/Users/patrickryan/cdev/project/siteFolder/wsgi/myproject/software/data/whilemyguitar.wav")()
        fftData = list()
        x1 = dtype(int)
        x2 = dtype(int)
        print len(inds)
        x1 = 0
        x2 = 90
        for i in range(x1,x2):
            startIndex = dtype(int)
            endIndex = dtype(int)
            startIndex = inds[i]
            endIndex = inds[i+1]
            data = getFFT(aud,startIndex,endIndex)
            fftData.append(data[0:15000])
        vals = loadCorrect()
        chordsNumbs = list()
        for i in range(int(x1),int(x2)):
            chordsNumbs.append(self.notes[vals[i]])
        dataSet = (fftData,chordsNumbs)
        train_set_x, train_set_y = shared_dataset2(dataSet)


        ds = (train_set_x,train_set_y)
        dy = getDataSet(inds,90,100)
        dz = getDataSet(inds,101,120)
        return (ds,dy,dz)
    def learn():
        """
        print tempSong
        """
        notes = {"C":1.0,
                "C#":2.0,
                  "D":3.0,
                  "D#":4.0,
                  "E":5.0,
                  "F":6.0,
                  "F#":7.0,
                  "G":8.0,
                  "G#":9.0,
                  "A":10.0,
                  "A#":11.0,
                  "B":12.0,
                  "B#":13.0,
                  "N":14.0
                  }
        datab = dataBase()
        #datab.addSong(tempSong,1)
        songId = 16
        dx ,dy, dz = getAllDatasets(songId)
        train_set_x = dx(0)
        numpy_rng = random.RandomState(89677)
        sda = SdA(
                numpy_rng=numpy_rng,
                n_ins=1500,
                hidden_layers_sizes=[1000, 1000, 1000],
                n_outs=14
        )
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        print n_train_batches
        batch_size = 1
        print '... getting the pretraining functions'
        pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,batch_size=batch_size)
        print '... pre-training the model'
        start_time = time.clock()
        corruption_levels = [.1, .2, .3]
        pretraining_epochs = 18
        pretrain_lr = 0.001
        for i in xrange(sda.n_layers):
                # go through pretraining epochs
                for epoch in xrange(pretraining_epochs):
                    # go through the training set
                    c = []
                    for batch_index in xrange(n_train_batches):
                        c.append(pretraining_fns[i](index=batch_index,
                                 corruption=corruption_levels[i],
                                 lr=pretrain_lr))
                    print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                    print mean(c)

        end_time = time.clock()
        print type(c)
        print len(c)
        print shape(c)
        for i in c:
            print type(i)
            print shape(i)
            print i


        print >> sys.stderr, ('The pretraining code for file ' +
                                  os.path.split(__file__)[1] +
                                  ' ran for %.2fm' % ((end_time - start_time) / 60.))
        datasets = list()


        datasets.append(dy)
        datasets.append(dz)
        datasets.append(dx)
        datasets = asarray(datasets)

        print '... getting the finetuning functions'
        train_fn, validate_model, test_model = sda.build_finetune_functions(
                datasets=datasets,
                batch_size=batch_size,
                learning_rate=0.1
        )

        patience = 10 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.  # wait this much longer when a new best is
                                # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = inf
        test_score = 0.
        start_time = time.clock()

        done_looping = False
        epoch = 0
        training_epochs=1000,
        while (epoch < training_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = train_fn(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    this_validation_loss = mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = test_model()
                        test_score = mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print(
            (
                'Optimization complete with best validation score of %f %%, '
                'on iteration %i, '
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
        )
        print >> sys.stderr, ('The training code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

        return sda
    def shared_dataset2(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')