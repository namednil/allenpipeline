"""
Simple evaluation script. Very stupid.
"""

import sys

def read_file(name):
	r = []
	with open(name) as f:
		for line in f:
			line = line.rstrip("\n")
			if line:
				r.append(line.split("\t"))
	return r


if __name__ == "__main__":
	f1 = read_file(sys.argv[1])
	f2 = read_file(sys.argv[2])

	assert len(f1) == len(f2)

	correct = 0
	for (i1,t1),(i2,t2) in zip(f1, f2):
		if t1 == t2:
			correct += 1
			
	print("Acc", correct / len(f1))

	
