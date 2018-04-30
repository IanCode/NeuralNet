import glob
import os

path = 'cats-and-dogs'

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
for filename in os.listdir(path):
	if filename[0] == 'd':
		newpath = "/home/ian/AI/NeuralNet/dogs/"+filename
		oldpath = "/home/ian/AI/NeuralNet/cats-and-dogs/"+ filename
		print("Oldpath: " + oldpath)
		print("Newpath: " + newpath)
		os.rename(oldpath, newpath)
	elif filename[0] == 'c':
		newpath = "/home/ian/AI/NeuralNet/cats/"+filename
		oldpath = "/home/ian/AI/NeuralNet/cats-and-dogs/"+ filename
		print("Oldpath: " + oldpath)
		print("Newpath: " + newpath)
		os.rename(oldpath, newpath)


