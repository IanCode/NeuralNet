import glob
import os

path = '\\cats-and-dogs\\'

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
for filename in os.listdir('cats-and-dogs'):
	if filename[0] == 'd':
		newpath = dir_path+"\\dogs\\"+filename
		oldpath = dir_path+path+ filename
		print("Oldpath: " + oldpath)
		print("Newpath: " + newpath)
		os.rename(oldpath, newpath)
	elif filename[0] == 'c':
		newpath = dir_path+"\\cats\\"+filename
		oldpath = dir_path+path+filename
		print("Oldpath: " + oldpath)
		print("Newpath: " + newpath)
		os.rename(oldpath, newpath)
