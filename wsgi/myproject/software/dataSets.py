class DataSets(object):
    def __init__(self, pathTraining,pathsValid,songIds):
        self.training_set = getDataFromFullSong(songIds[0],pathTraining[0],pathsValid[0])
        self.valid_set = getDataFromFullSong(songIds[1],pathTraining[1],pathsValid[1])
        self.test_set = getDataFromFullSong(songIds[2],pathTraining[2],pathsValid[2])
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