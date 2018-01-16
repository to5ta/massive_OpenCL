#!/usr/bin/python
import random

COUNT = 60
LOWER = 0
UPPER = 999




def writeNumbers(data):

	l = len(data)

	with open("random.txt", "w") as file:
		for i in range(1,l+1):
			file.write("{:>3}".format(data[i-1]))
			if i<l:
				file.write(", ")
			if i%20==0:
				file.write("\n")

	data.sort()

	with open("sorted.txt", "w") as file:
		for i in range(1,l+1):
			file.write("{:>3}".format(data[i-1]))
			if i<l:
				file.write(", ")
			if i%20==0:
				file.write("\n")
	return



def main():
	data = [random.randint(LOWER,UPPER) for x in range(COUNT)]
	writeNumbers(data)



if __name__ == "__main__":
	main()