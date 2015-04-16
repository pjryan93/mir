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
class dataBase(object):
    def __init__(self):
        self.conn = sqlite3.connect('/Users/patrickryan/cdev/project/siteFolder/wsgi/myproject/software/musicDB.db')
        self.c = self.conn.cursor()
        self.setup()
    def setup(self):
        self.c.execute('''CREATE TABLE if not exists songs
              (id INTEGER PRIMARY KEY,
              name TEXT,
              correctPath TEXT,
              bpm INTEGER)''')
        self.c.execute('''CREATE TABLE if not exists segments
             (position INTEGER  ,
              startTime TEXT,
              startIndex INTEGER,
              songID INTEGER,
              FOREIGN KEY(songID) REFERENCES song(id))''')
    def addSong(self,song,correctPath,id):
        if correctPath is None:
        self.c.execute("INSERT INTO songs VALUES("+str(id)+",'"+song.path+"','"+correctPath+"',"+str(song.avg_bpm)+")")
        for i in song.segments:
            self.addSegment(i,id)
        self.conn.commit()
    def addSegment(self,segment,id):
        self.c.execute("INSERT INTO segments VALUES("+str(segment.position)+","+str(segment.startTime)+","+str(segment.startIndex)+","+str(id)+")")
    def getSegments(self,songId):
        self.c.execute("SELECT * FROM segments where songID=:songId",{"songId": songId})
        return self.c.fetchall()
    def getSong(self,songId):
        self.c.execute("SELECT * FROM songs where id=:songId",{"songId": songId})
        return self.c.fetchall()

class Segment(object):
    def __init__(self,startT,startI,pos,data):
        self.startTime = startT
        self.startIndex = double(startI)
        self.position = pos
        self.data = asarray(data)
class Song(object):
    def __init__(self,audioPath,times):
        self.wave_file = wv.open(audioPath, 'r')
        self.numberOfChannels = self.wave_file.getnchannels()
        self.fs = 44100.0
        self.path = audioPath
        self.audio = ess.MonoLoader(filename=audioPath)()
        self.avg_bpm, self.beat_start2, self.confidence, self.tempo, self.beat_duration = ess.RhythmExtractor2013(method='multifeature')(self.audio)
        self.beat_start = zeros(len(self.audio)/self.fs)
        self.segments = self.getSegmentsFromData(times)
        #self.segments = self.setSegments()
        print shape(self.segments)
    def getSegmentsFromData(self,times):
        self.beat_start = asarray(times)
        return asarray(self.getSegmentsFromOnsets())
    def getSegmentsDividedIntoSeconds(self):
        self.beat_start = zeros(len(self.audio)/self.fs)
        self.beat_start[0] = 0
        self.beat_start = zeros(len(self.audio)/self.fs)
        self.beat_start[0] = 0
        self.segments = list()
        for i in range(1,len(self.beat_start-3)):
            print i * self.fs
            currentSegment = Segment(i,(self.fs*i),i,self.audio[i*(self.fs-1):i*(self.fs)])
            self.segments.append(currentSegment)
        self.segments = asarray(self.segments)
        return self.segments
    def getSegmentsFromOnsets(self):
        timeIndex = 0;
        segments = list()
        seg = list()
        startTime = 0.0
        increment = 1
        endTime = self.beat_start[increment]
        startIndex = 0
        print len(self.audio)
        for frameIndex in range(0,len(self.audio)-1):
            time = float(frameIndex)/float(self.fs)
            if  startTime < time and time < endTime:
                seg.append(self.audio[frameIndex])
            elif time > endTime:
                currentSegment = Segment(startTime,frameIndex,len(segments),(seg))
                segments.append(currentSegment)
                seg = list()
                timeIndex= timeIndex + increment
                startTime = self.beat_start[timeIndex]
                if timeIndex < len(self.beat_start)-increment:
                    endTime = self.beat_start[timeIndex+increment]
                else:
                    break
            else:
                continue
        currentSegment = Segment(startTime,startIndex,len(segments),(seg))
        segments.append(currentSegment)
        return asarray(segments)
    def setSegments(self):
        segments = self.getSegmentsFromOnsets()
        print type(asarray(segments))
        return asarray(segments)
def getFFT(audio,startIndex,endIndex):
    sample = audio[startIndex:endIndex]
    dataToReturn = ft.fft(sample)
    return absolute(dataToReturn[0:1500])
def readFile():
    crs = open("./data/10CD1_-_The_Beatles/CD1_-_07_-_While_My_Guitar_Gently_Weeps.lab", "r")
    data = list()
    for columns in ( raw.strip().split() for raw in crs ): 
        val = float(columns[0])
        data.append((round(val),columns[2]))
    return data
def getTimes(filePath):
    crs = open(filePath, "r")
    data = list()
    for columns in ( raw.strip().split() for raw in crs ):  
        data.append(float(columns[0]))
    return data
def loadCorrect(filePath):
    crs = open(filePath, "r")
    data = list()
    counter = 0
    for columns in ( raw.strip().split() for raw in crs ):
        if counter == 0:
            counter= counter + 1
            continue
        elif len(columns[2]) == 3:
            note = columns[2].split('/',1)[0]
            data.append(note)
        else:
            note = columns[2].split(':',1)[0]
            data.append(note)
    return data 


#dict extra = {":min":10,":"}
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

def getDataSet(inds, sIn, eIn):
    sIn = int(sIn)
    eIn = int(eIn)
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
    aud = ess.MonoLoader(filename="/Users/patrickryan/cdev/project/siteFolder/wsgi/myproject/software/data/whilemyguitar.wav")()
    fftData = list()

    for i in range(sIn,eIn):
        s1 = inds[i]
        s2 = inds[i+1]
        data = getFFT(aud,s1,s2)
        fftData.append(data[0:15000])
    vals = loadCorrect("./data/10CD1_-_The_Beatles/CD1_-_07_-_While_My_Guitar_Gently_Weeps.lab")
    chordsNumbs = list()
    for i in range(sIn,eIn):
        chordsNumbs.append(notes[vals[i]])
    dataSet = (fftData,chordsNumbs)
    retValue_X ,retValue_Y =  shared_dataset2(dataSet)
    return (retValue_X,retValue_Y)
def getDataFromFullSong(songId,songPath,correctPath):
    inds = getIndexes(songId)
    print inds
    notes = { "C":1.0,
              "C:7":12.0,"
              "C#":2.0,
              "Db":2.0,
              "D/b7":2.0,
              "D:7":3.0,
              "D":3.0,
              "D#":4.0,
              "Eb":4.0,
              "E":5.0,
              "E:7":5.0,
              "Fb":5.0,
              "F":6.0,
              "F:7":6.0,
              "F#":7.0,
              "Gb":7.0,
              "G":8.0,
              "G:7":8.0,
              "G#":9.0,
              "Ab":9.0,
              "A":10.0,
              "A:7":10.0,
              "A#":11.0,
              "Bb":11.0,
              "B":12.0,
              "B:7":12.0,
              "B#":12.0,
              "Cb":12.0,
              "N":13.0
    }
    aud = ess.MonoLoader(filename=songPath)()
    fftData = list()
    print len(inds)
    for i in range(0,len(inds)-1):
        s1 = inds[i]
        s2 = inds[i+1]
        if s2 - s1 == 0  or s1 > s2:
            print s2
            print type(s2)
            print s1
            print type(s1)
        data = getFFT(aud,s1,s2)
        print 'song has ' + str(len(inds)-1 - i) + " left"
        fftData.append(data[0:15000])
    vals = loadCorrect(correctPath)
    print 'songDone'
    chordsNumbs = list()
    for i in range(0,len(inds)-1):
        chordsNumbs.append(notes[vals[i]])
    dataSet = (fftData,chordsNumbs)
    retValue_X ,retValue_Y =  shared_dataset2(dataSet)
    return (retValue_X,retValue_Y)
def getIndexes(songId):
    datab = dataBase()
    sng = datab.getSong(songId)
    segs = datab.getSegments(songId)
    inds = zeros(len(segs)-1)
    for i in range(0,len(segs)-1):
        inds[i] = int(segs[i][2])
    print inds
    return inds

def getAllDatasetsFromFullSong(songId, correctPath):
    sng = datab.getSong(songId)
    inds = getIndexes(songId)
    ds = getDataFromFullSong(inds,sng.path,correctPath)
    #dy = getDataFromFullSong(inds,sng.path,correctPath)
    #dz = getDataFromFullSong(inds,sng.path,correctPath)
    dy = ds
    dz = ds
    return (ds,dy,dz)

def getAllDatasets(songId):
    notes = {
                "C":1.0,
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
    sng = datab.getSong(songId)
    segs = datab.getSegments(songId)
    inds = zeros(len(segs))
    for i in range(0,len(segs)-8):
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
    vals = loadCorrect("./data/10CD1_-_The_Beatles/CD1_-_07_-_While_My_Guitar_Gently_Weeps.lab")
    chordsNumbs = list()
    for i in range(int(x1),int(x2)):
        chordsNumbs.append(notes[vals[i]])
    dataSet = (fftData,chordsNumbs)
    train_set_x, train_set_y = shared_dataset2(dataSet)


    ds = (train_set_x,train_set_y)
    dy = getDataSet(inds,90,100)
    dz = getDataSet(inds,101,120)
    return (ds,dy,dz)
class DataSets(object):
    def __init__(self, pathTraining,pathsValid,songIds):
        self.training_set = getDataFromFullSong(songIds[0],pathTraining[0],pathsValid[0])
        #self.valid_set = getDataFromFullSong(songIds[1],pathTraining[1],pathsValid[1])
        #self.test_set = getDataFromFullSong(songIds[2],pathTraining[2],pathsValid[2])
        self.valid_set = self.training_set
        self.test_set = self.training_set
    def getTrainingX(self):
        return self.training_set[0]

def setup():
    path1 = "/Users/patrickryan/cdev/project/siteFolder/wsgi/myproject/software/data/Blackbird.wav"
    path2 = "/Users/patrickryan/cdev/project/siteFolder/wsgi/myproject/software/data/backintheussr.wav"
    path3 = "/Users/patrickryan/cdev/project/siteFolder/wsgi/myproject/software/data/dearprudence.wav"
    path4 = "/Users/patrickryan/cdev/project/siteFolder/wsgi/myproject/software/data/whilemyguitar.wav"

    tPath1 = "./data/10CD1_-_The_Beatles/CD1_-_11_-_Black_Bird.lab"
    tPath2 = "./data/10CD1_-_The_Beatles/CD1_-_01_-_Back_in_the_USSR.lab"
    tPath3 = "./data/10CD1_-_The_Beatles/CD1_-_02_-_Dear_Prudence.lab"
    tPath4 = "./data/10CD1_-_The_Beatles/CD1_-_07_-_While_My_Guitar_Gently_Weeps.lab"
    pathsX = list()
    pathsX.append(path1)
    pathsX.append(path3)
    pathsX.append(path4)

    pathsY = list()
    pathsY.append(tPath1)
    pathsY.append(tPath2)
    pathsY.append(tPath4)

    songsList = list()
    songsList.append(42)
    songsList.append(43)
    songsList.append(45)

    return DataSets(pathTraining = pathsX,pathsValid = pathsY,songIds = songsList)

def learn():
    datab = dataBase()
    dSetIn = setup()
    dx = dSetIn.training_set
    dy = dSetIn.valid_set
    dz = dSetIn.test_set
    datasets = list()
    datasets.append(dy)
    datasets.append(dz)
    datasets.append(dx)
    datasets = asarray(datasets)
    train_set_x = dx[0]
    numpy_rng = random.RandomState(89677)
    sda = SdA(
            numpy_rng=numpy_rng,
            n_ins=1500,
            hidden_layers_sizes=[500, 500, 500],
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
    
"""
datab = dataBase()
segs = datab.getSegments(42)
aud = ess.MonoLoader(filename="/Users/patrickryan/cdev/project/siteFolder/wsgi/myproject/software/data/whilemyguitar.wav")()
startIndex = segs[3][2]
endIndex = segs[4][2]
data = getFFT(aud,startIndex,endIndex)
data = data[0:15000]
"""
