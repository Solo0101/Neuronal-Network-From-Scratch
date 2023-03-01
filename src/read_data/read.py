import idx2numpy
import src.enviroment.constants as const

XTrainMat = idx2numpy.convert_from_file(const.imageTrainFile)
yTrainMat = idx2numpy.convert_from_file(const.labelTrainFile)

XTrainArr = XTrainMat.reshape(-1, XTrainMat.shape[0]).T
yTrainArr = yTrainMat.flatten()


XTestMat = idx2numpy.convert_from_file(const.imageTestFile)
yTestMat = idx2numpy.convert_from_file(const.labelTestFile)

XTestArr = XTestMat.reshape(-1, XTrainMat.shape[0]).T
yTestArr = yTestMat.flatten()






