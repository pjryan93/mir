import essentia
from scipy.fftpack import fft, rfft
import sqlite3
from librosa.beat import *
import essentia.standard as ess
from numpy import *
import wave as wv
from sda import SdA
import theano
import time
import theano.tensor as T
import sys
import numpy.fft as ft
import os
from random import randint
randint(2,9)

from preProc import *

def train(datasets):
    datasets = asarray(datasets)
    train = datasets[0]
    train_set_x = train[0]
    numpy_rng = random.RandomState(89677)
    sda = SdA(
            numpy_rng=numpy_rng,
            n_ins=1500,
            hidden_layers_sizes=[500, 500, 500,500],
            n_outs=14
    )
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    print n_train_batches
    batch_size = 1
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,batch_size=batch_size)
    print '... pre-training the model'
    start_time = time.clock()
    corruption_levels = [.1, .2, .3,.4]
    pretraining_epochs = 15
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
    print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

    print '... getting the finetuning functions'
    train_fn, validate_model, test_model  = sda.build_finetune_functions(
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
def getFFT(audio,startIndex,endIndex):
        sample = audio[startIndex:endIndex]
        dataToReturn = ft.fft(sample)
        return absolute(dataToReturn[0:1500])

def getIndexesOfRaw(self,songIn):
        inds = zeros(len(songIn.segments)-1)
        for i in range(0,len(inds)-1):
            inds[i] = int(songIn.segments[i].startIndex)
        print inds
        return inds

def getMaxIndex(chordsIn):
    print 'update'
    maxIn = chordsIn[0]
    index = 0
    for i in range(0,len(chordsIn)):
        if chordsIn[i] > maxIn:
            print chordsIn[i]
            index = i
            print index
            maxIn = chordsIn[i]
    return (index,maxIn)
def getAllChordIndexes(chordList):
    chordsToReturn = list()
    for i in chordList:
        index , conf = getMaxIndex(chordList)
        chordsToReturn.append(index,conf)
    return chordsToReturn
def prac():
    aud = Song("/Users/patrickryan/cdev/project/siteFolder/wsgi/myproject/software/data/whilemyguitar.wav")
    segments = aud.segments
    test = getSongsSet(30)
    two = getSongsSet(31)
    three = getSongsSet(32)
    sets =list()
    sets.append(test)
    sets.append(two)
    sets.append(three)
    sets = asarray(sets)
    userSda = train(sets)
    return userSda 
def makePredictions(aud,userSda):
    segments = aud.segments
    chords = list()
    for i in range(0,len(segments)-1):
        inData = getFFT(aud.audio,segments[i].startIndex,segments[i+1].startIndex)
        pred = userSda.predict(inData)
        chords.append(pred)
    return chords
def getScore(chords):
    score = list()
    for i in range(0,len(chords)):
        tempBuff = asarray(chords[i][0])
        maxes, confs = getMaxIndex(tempBuff)
        noteVal = mapKeysToNotes(maxes)
        score.append(noteVal)
    return score
def getResult(songPath):
    aud = Song(songPath)
    myCog = prac()
    chords = makePredictions(aud,myCog)
    score = getScore(chords)
    return score
def testing():
    aud = Song("/Users/patrickryan/cdev/project/siteFolder/wsgi/myproject/software/data/whilemyguitar.wav")
    myCog = prac()
    chords = makePredictions(aud,myCog)
    score = getScore(chords)
    return chords, myCog, score
def getRandomResults():
        notes = asarray(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B","B#","N"])
        spacer = randint(2,9)
        retValue  = " "
        for i in range(0,40 + spacer):
            index = randint(1,12) 
            retValue = retValue + " " + notes[index]
        print retValue
        return retValue
if __name__ == '__main__':
    var = raw_input("Please enter full path: ")
    score = getResult(var)
    print 'The chords the following'
    print score
