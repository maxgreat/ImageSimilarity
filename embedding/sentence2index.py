import fastText
import torch
import torch.nn as nn
import numpy as np
import io
import os
import sys 
import argparse


def convertFile(inp, outp, vocab):
	vocab = torch.load(vocab)
	with open(inp) as fin:
		with open(outp, 'w') as fout:
			for line in fin:
				if '\t' in line:
					f, l = line.split('\t')
					els = l.split(' ')
					fout.write(f+'\t')
					for e in els:
						if e in vocab :
							fout.write(str(vocab[e])+' ')
					fout.write('\n')
				else:
					print(line)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('input', help="filename with text to convert")
	parser.add_argument('output', help="output filename")
	parser.add_argument('-v', '--vocab', help="vocab file", default='data/vocab.save')
	args = parser.parse_args()
	
	convertFile(args.input, args.output, args.vocab)
