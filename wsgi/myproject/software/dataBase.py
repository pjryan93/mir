import sqlite3
from .util import Song, Segment
# create new db and make connection
class dataBase(object):
    def __init__(self):
        self.conn = sqlite3.connect('/data/musicData.db')
        self.c = conn.cursor()
        c.execute('''CREATE TABLE songs
             (id INTEGER PRIMARY KEY,
              name TEXT, bpm Text)''')
        c.execute('''CREATE TABLE segments
             (position INTEGER PRIMARY KEY,
              startTime TEXT,
              startIndex INTEGER,
              songID INTEGER,
              FOREIGN KEY(songID) REFERENCES song(id))''')
    def addSong(self,song,id):
        c.execute("INSERT INTO songs VALUES("+id+","+song.path+","+song.avg_bpm+")")
        for i in song.segments:
            self.addSegment(i,id)
        c.commit()
    def addSegment(self,segment,id):
        c.execute("INSERT INTO segments VALUES("+segment.position+","+segment.startTime+","+segment.startIndex+","+id+")")
