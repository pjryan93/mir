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

class PreProcHelper(object):
    def __init__(self):
        self.datab = dataBase()
    def getLabeldDataFromSong(self,songId):
        songPath = self.datab.getSong(songId)
        dataSet = (self.getFFTforSong(songId,songPath[0][1]),self.mapNotesToKeys(songId))
        return shared_dataset2(dataSet)
    def getFFTforSong(self,songId,songPath):
        aud = ess.MonoLoader(filename=songPath)()
        indexes = self.getIndexes(songId)
        fftData = list()
        for i in range(0,len(indexes)-1):
            s1 = indexes[i]
            s2 = indexes[i+1]
            data = self.getFFT(aud,s1,s2)
            print 'song has ' + str(len(indexes)-1 - i) + " left to perform the FFT on"
            fftData.append(data[0:1500])
        return asarray(fftData)
    def getFFTforSongRaw(self,songIn):
        aud = ess.MonoLoader(filename=songPath)()
        indexes = self.getIndexes(songId)
        print indexes
        fftData = list()
        for i in range(0,len(indexes)-1):
            s1 = indexes[i]
            s2 = indexes[i+1]
            data = self.getFFT(aud,s1,s2)
            print 'song has ' + str(len(indexes)-1 - i) + " left to perform the FFT on"
            fftData.append(data[0:1500])
        return asarray(fftData)
    def getIndexes(self,songId):
        sng = self.datab.getSong(songId)
        segs = self.datab.getSegments(songId)
        inds = zeros(len(segs)-1)
        for i in range(0,len(segs)-1):
            inds[i] = int(segs[i][2])
        print inds
        return inds
    def getFFT(self,audio,startIndex,endIndex):
        sample = audio[startIndex:endIndex]
        dataToReturn = ft.fft(sample)
        return absolute(dataToReturn[0:1500])
    def mapNotesToKeys(self,songId):
        correctData = loadCorrect(self.datab.getSong(songId)[0][2])
        notes = { "C":1.0,
                  "C:7":12.0,
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
                  "A/b3":10,
                  "A:7":10.0,
                  "A#":11.0,
                  "Bb":11.0,
                  "B":12.0,
                  "B:7":12.0,
                  "B#":12.0,
                  "Cb":12.0,
                  "N":13.0
        }
        chordsNumbs = list()
        for i in range(0,len(correctData)-1):
            chordsNumbs.append(notes[correctData[i]])
        return chordsNumbs
def mapKeysToNotes(indexIn):
    notes = asarray(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B","B#","N"])
    return notes[indexIn]
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
class dataBase(object):
    def __init__(self):
        self.conn = sqlite3.connect('/Users/patrickryan/cdev/project/siteFolder/wsgi/myproject/software/database/music.db')
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
        if correctPath is not None:
            self.c.execute("INSERT INTO songs VALUES("+str(id)+",'"+song.path+"','"+correctPath+"',"+str(song.avg_bpm)+")")
        else:
            self.c.execute("INSERT INTO songs VALUES("+str(id)+",'"+song.path+"','"+'nothing'+"',"+str(song.avg_bpm)+")")
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
    def __init__(self,audioPath,times = None):
        self.wave_file = wv.open(audioPath, 'r')
        self.numberOfChannels = self.wave_file.getnchannels()
        self.fs = 44100.0
        self.path = audioPath
        self.audio = ess.MonoLoader(filename=audioPath)()
        self.avg_bpm, self.beat_start2, self.confidence, self.tempo, self.beat_duration = ess.RhythmExtractor2013(method='multifeature')(self.audio)
        self.beat_start = zeros(len(self.audio)/self.fs)
        if times is not None:
            self.segments = self.getSegmentsFromData(times)
        else:
            self.segments = self.getSegmentsDividedIntoSeconds()
        print len(self.segments)
        print self.segments[2].startIndex
        #self.segments = self.setSegments()
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
def getPaths():
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
    pathsY.append(tPath3)
    pathsY.append(tPath4)

    songsList = list()
    songsList.append(42)
    songsList.append(43)
    songsList.append(45)
    return pathsX, pathsY
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
def addSongs():
    paths, correctPaths = getPaths()
    helper = PreProcHelper()
    for i in range(0,len(paths)):
       songBuffer = Song(paths[i],getTimes(correctPaths[i]))
       helper.datab.addSong(songBuffer,correctPaths[i],i+30)
def getSongsSet(songId):
    helper = PreProcHelper()
    return helper.getLabeldDataFromSong(songId)


#addSongs()
"""
data = list()
for i in range(30,33):
    data.append(getSongsSet(i))
"""


