# coding=utf-8
import errno
import numpy as np
import sys
import time
import os
import glob

def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise

def find_best_weights(log_dir):
	import re
	p = re.compile('.*?_weights_([0-9]+)_(.*?).hdf5')
	all = glob.glob(log_dir + "/*_weights*.hdf5")
	found = None
	found_ep = -1
	for fp in all:
		f = os.path.basename(fp)
		m = p.match(f)
		if m:
			ep = int(m.group(1))
			if ep > found_ep:
				found_ep = ep
				found = fp
	return found



def flip_axis(x, axis):
	x = np.asarray(x).swapaxes(axis, 0)
	x = x[::-1, ...]
	x = x.swapaxes(0, axis)
	return x

class Logger(object):
	def __init__(self):
		self.terminal = sys.stdout
		self.log = None
		self.lastwritten = 0.0

	def set_logfile(self,file):
		self.flush()
		if self.log :
			self.log.close()
		self.log = open(file, "a")

	def write(self, message):
		self.terminal.write(message)
		if self.log:
			self.log.write(message)
			now = time.time()
			if now - self.lastwritten > 3.0 :
				self.log.flush()
				self.lastwritten = now

	def flush(self):
		if self.log:
			self.log.flush()
		pass

	def close(self):
		if self.log:
			self.log.flush()
			self.log.close()
		pass

	def isatty(self):
		return False

class MinMaxScaler:
	def fit(self, x):
		self.min = np.min(x)
		self.max = np.max(x)
	def transform(self,x):
		y = (x - self.min) / (self.max - self.min)
		return y
