import numpy as np
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import os



def is_ancestral(g):
	'''
	Returns a boolean: true if the graph is ancestral, false otherwise.

	g = graph
	'''

	# p = |vertices|
	p = g.shape[0]

	# trv = traverse ; pa = parents ; sp = spouses
	trv = np.eye(p, dtype='bool')
	pa = g == 1
	sp = g == 2

	for i in range(p):
		trv = np.dot(pa, trv)
		# ancestral
		if not trv.any():
			return True
		# almost directed cycle
		if (trv*sp).any():
			return False

	# (not) directed cycle
	return not trv.any()



def is_mag(g):
	'''
	Returns a boolean: true if the graph is ancestral and maximal, 
	false otherwise.

	g = graph
	'''

	# p = |vertices|
	p = g.shape[0]

	# trv = traverse ; dis = districts ; an = ancestors ; 
	# pa = parents ; sp = spouses
	trv = np.eye(p, dtype='bool')
	dis = np.eye(p, dtype='bool')
	an = np.eye(p, dtype='bool')
	pa = g == 1
	sp = g == 2

	# ancestral flag
	ancestral = False
	
	for i in range(p):
		trv = np.dot(pa, trv)
		dis += np.dot(sp, dis)
		if not ancestral:
			# ancestral
			if not trv.any():
				ancestral = True
			# almost directed cycle
			elif (trv*sp).any():
				return False
			else:
				an += trv

	# directed cycle
	if not ancestral:
		return False

	# e = edge
	for e in zip(*np.where((sp)^(dis))):
		# check if e is a primitive inducing path
		if e[0] < e[1] and not pa[e[0],e[1]] and not pa[e[1],e[0]]:
			# r_trv = restricted traverse ; r_sp = restricted spouses
			r_trv = np.diag(an[e[0]]+an[e[1]])
			r_sp = np.dot(r_trv, np.dot(sp, r_trv))
			for i in range(p):
				r_trv = np.dot(r_sp, r_trv)
				# e is a primitive inducing path
				if r_trv[e[0],e[1]]:
					return False

	# no e is a primitive inducing path
	return True



def exists_inducing_path(g, a, b, s):
	'''
	Returns a boolean: true if there exists an inducing path in the 
	graph between vertex a and vertex b for set s, false otherwise.

	Latent variables are currently not handled.

	g = graph
	a = vertex
	b = vertex
	s = set of vertices
	'''

	# p = |vertices|
	p = g.shape[0]

	# an = ancestors ; pa = parents ; sp = spouses	
	an = np.eye(p, dtype='bool')
	pa = g == 1
	sp = g == 2

	for i in range(p):
		an += np.dot(pa, an)

	# end = endpoints ; r = restricted ; r_trv = restricted traverse
	end = np.diag(np.isin(np.arange(p, dtype='uint8'), [a,b]))
	r = an[a] + an[b]
	for v in s:
		r += an[v]
	r = np.diag(r)
	r_trv = np.dot(pa, end) + np.dot(pa, end).T + sp
	r_trv = np.dot(r, np.dot(r_trv, r))
	for i in range(p):
		r_trv += np.dot(r_trv, r_trv)

	return r_trv[a,b]



def get_mags(p):
	'''
	Returns a list of all maximal ancestral graphs with p vertices.

	p = |vertices|
	'''

	# mags = maxmial ancestral graphs ; g = graph template
	mags = []
	g = np.zeros((p,p), dtype='uint8')
	
	# populate the list
	get_mags_helper(mags, g)

	return mags



def get_mags_helper(mags, g, idx=0):
	'''
	A helper function for get_mags().

	mags = maximal ancestral graphs
	g = graph template
	idx = index
	'''

	if idx == g.shape[0]**2:
		if is_mag(g):
			mags.append(np.copy(g))
	else:
		i,j = np.unravel_index(idx, g.shape)
		if i == j:
			g[i,j] = 0
			get_mags_helper(mags, g, idx+1)
		elif i > j and g[j,i] == 1:
			g[i,j] = 0
			get_mags_helper(mags, g, idx+1)
		elif i > j and g[j,i] == 2:
			g[i,j] = 2
			get_mags_helper(mags, g, idx+1)
		else:
			for e in (0,1,2) if i < j else (0,1):
				g[i,j] = e
				get_mags_helper(mags, g, idx+1)



def is_mconnecting(g, s):
	'''
	Returns a boolean: true if the set is m-connecting for the graph, 
	false otherwise.

	g = graph
	s = set
	'''

	# m-connecting flag
	m_connecting = True

	if len(s) > 1:
		# ab = a/b pair
		for ab in it.combinations(s, 2):
			m_connecting = m_connecting and exists_inducing_path(g, 
				ab[0], ab[1], [v for v in s if v not in ab])
			if not m_connecting:
				break

	return m_connecting



def get_mconnecting_sets(g):
	'''
	Returns a list of the sets that are m-connecting for the graph.

	g = graph
	'''

	# p = |vertices|
	p = g.shape[0]

	# m-connecting sets
	sets = []

	for i in range(p):
		for s in it.combinations(np.arange(p, dtype='uint8'), i+1):
			if is_mconnecting(g, s):
				sets.append(s)

	return sets



def get_mecs(p):
	'''
	Returns a list of Markov equivalence classes

	p = |vertices|
	'''

	# maxmial ancestral graphs
	mags = get_mags(p)

	idx = [s for i in range(1,p+1) for s in it.combinations(
		np.arange(p, dtype='uint8'), i)]

	mecs = {}
	for mag in mags:
		mcs = get_mconnecting_sets(mag)
		key = np.empty(len(mcs), dtype='uint8')
		j = 0
		for i, s in enumerate(mcs):
			while idx[j] != s:
				j += 1 
			key[i] = j
		key = tuple(key)
		if key not in mecs:
			mecs[key] = []
		anc = np.eye(p, dtype='bool') + mag == 1
		for i in range(p):
			anc = np.dot(anc, anc)
		mecs[key].append((mag, anc))

	return mecs, idx



class Model:

	def __init__(self, data):
		
		if not isinstance(data, pd.DataFrame):
			print('pandas dataframe required')
			quit()

		self.data = data
		self.vbls = list(data)


	def get_data(self):

		return self.data


	def get_variables(self):

		return self.vbls


	def resample(self):
		print('method not overridden')
		quit()


	def log_prob(self, vbls):
		print('method not overridden')
		quit()


	def info(self, vbls, rsmp=False):

		p = len(vbls)

		info = np.float64(0)
		for i in range(1,p+1):
			for s in it.combinations(vbls, i):
				info += (-1)**((p-i)%2) * self.log_prob(list(s), rsmp)

		return info



class MVN(Model):

	def __init__(self, data):

		Model.__init__(self, data)

		self.cov = np.corrcoef(data.T)
		self.cov_rsmp = None

		self.n = len(data)
		self.n_rsmp = None

		self.const = -0.5*np.log(2*np.pi) - 0.5


	def resample(self, frac=1, n=None, rplc=True):

		if n == None:
			data = self.data.sample(frac=frac, replace=rplc)
			self.n_rsmp = int(frac*self.n)
		else:
			data = self.data.sample(n=n, replace=rplc)
			self.n_rsmp = n

		self.cov_rsmp = np.corrcoef(data.T)


	def log_prob(self, vbls, rsmp):

		idx = [self.vbls.index(v) for v in vbls]
		p = len(idx)

		cov = self.cov_rsmp if rsmp else self.cov
		cov = cov[np.ix_(idx,idx)]

		log_prob = -0.5*np.log(np.linalg.det(cov)) + p*self.const

		n = self.n_rsmp if rsmp else self.n

		return n*log_prob - (p*(p+1)/4)*np.log(n)



class DG(Model):

	def __init__(self, data):

		Model.__init__(self, data)

		self.var_map = {}
		idx = 0
		for v in self.vbls:
			if(data[v].dtypes == np.integer):
				crd = len(np.unique(data[v])) - 1
				self.var_map[v] = [idx+i for i in range(crd)]
				idx += crd
			elif(data[v].dtypes == np.inexact):
				self.var_map[v] = [idx]
				idx += 1
			else:
				print('unrecognized data-type: ' + v)
				quit()


		self.aug_data = pd.DataFrame(np.empty([len(data),idx]))
		idx = 0
		for v in self.vbls:
			crd = len(self.var_map[v])
			if crd == 1:
				self.aug_data[idx] = data[v]
			else:
				u = np.unique(data[v])[:-1]
				for i in self.var_map[v]:
					self.aug_data[idx+i] = (data[v]==u[i]).astype('float')
			idx += crd
			
		self.cov = np.corrcoef(self.aug_data.T)
		self.cov_rsmp = None

		self.n = len(data)
		self.n_rsmp = None

		self.const = -0.5*np.log(2*np.pi) - 0.5


	def resample(self, frac=1, n=None, rplc=True):

		if n == None:
			data = self.aug_data.sample(frac=frac, replace=rplc)
			self.n_rsmp = int(frac*self.n)
		else:
			data = self.aug_data.sample(n=n, replace=rplc)
			self.n_rsmp = n

		self.cov_rsmp = np.corrcoef(data.T)


	def log_prob(self, vbls, rsmp):

		idx = [i for v in vbls for i in self.var_map[v]]
		p = len(idx)

		cov = self.cov_rsmp if rsmp else self.cov
		cov = cov[np.ix_(idx,idx)]

		log_prob = -0.5*np.log(np.linalg.det(cov)) + p*self.const

		n = self.n_rsmp if rsmp else self.n

		return n*log_prob - (p*(p+1)/4)*np.log(n)



class EP:

	def __init__(self, model, dtype='float64'):

		if not isinstance(model, Model):
			print('Model class required')
			quit()

		self.data = model.get_data()
		self.vbls = model.get_variables()
		self.p = len(self.vbls)

		self.knwl = {'adj': np.zeros([self.p,self.p], dtype='bool'),
					 '!adj': np.eye(self.p, dtype='bool'),
					 'anc': np.eye(self.p, dtype='bool'),
					 '!anc': np.zeros([self.p,self.p], dtype='bool')}

		self.model = model
		self.dtype = dtype

		self.sel = []
		self.q = 0
		self.mecs = None
		self.idx = None


	def set_knowledge(self, filename):

		with open(filename, 'r') as f:
			knwl = json.load(f)

		self.knwl = {'adj': np.zeros([self.p,self.p], dtype='bool'),
					 '!adj': np.eye(self.p, dtype='bool'),
					 'anc': np.eye(self.p, dtype='bool'),
					 '!anc': np.zeros([self.p,self.p], dtype='bool')}

		for cstr in knwl:
			for rel in cstr['rels']:
				tmp = rel.split()

				if len(tmp) != 3:
					print('incorrect file format')
					quit()
				elif tmp[0] not in cstr['sets']:
					print('undefined set "' + tmp[0] + '" in rels')
					quit()
				elif tmp[1] not in ['adj', '!adj', 'anc', '!anc']:
					print('undefined relation "' + tmp[1] + '" in rels')
					quit()
				elif tmp[2] not in cstr['sets']:
					print('undefined set "' + tmp[2] + '" in rels')
					quit()

				for a in cstr[tmp[0]]:
					i = self.vbls.index(a)
					for b in cstr[tmp[2]]:
						j = self.vbls.index(b)
						if 'adj' in tmp[1]:
							self.knwl[tmp[1]][i,j] = True
						elif 'anc' == tmp[1]:
							self.knwl['!anc'][i,j] = True
						self.knwl[tmp[1]][j,i] = True

		for i in range(self.p):
			self.knwl['anc'] = np.dot(self.knwl['anc'], 
				self.knwl['anc'])

		for i in range(self.p):
			self.knwl['!anc'] = np.dot(self.knwl['!anc'], 
				self.knwl['anc'].T)

		if self.q != 0:
			self.update()


	def set_selected(self, selected, apply_knwl=True):

		if not all([v in self.vbls for v in selected]):
			print('invalid variables:', 
				[v for v in selected if v not in self.vbls])
			quit()

		self.sel = selected
		self.q = len(selected)
		self.update(apply_knwl)


	def get_data(self):

		return self.data


	def get_variables(self):

		return self.vbls


	def get_model(self):

		return self.model


	def get_knowledge(self):

		return self.knwl


	def get_selected(self):

		return self.sel


	def update(self, apply_knwl=True):

		try:
			with open('mecs_'+str(self.q)+'.p', 'rb') as f:
				self.mecs, self.idx = pickle.load(f)
		except:
			self.mecs, self.idx = get_mecs(self.q)
			with open('mecs_'+str(self.q)+'.p', 'wb') as f:
				pickle.dump((self.mecs, self.idx), f)

		# check knowledge
		if apply_knwl:
			delete = []
			idx = [self.vbls.index(v) for v in self.sel]
			i = np.ix_(idx,idx)
			for key in self.mecs:
				updated = []
				for mag, anc in self.mecs[key]:
					adj = (mag.T+mag).astype('bool')
					if ((~self.knwl['adj'][i] + adj).all() and 
						(~self.knwl['!adj'][i] + ~adj).all() and
						(~self.knwl['anc'][i] + anc).all() and
						(~self.knwl['!anc'][i] + ~anc).all()):
						updated.append((mag, anc))
				if len(updated):
					self.mecs[key] = updated
				else:
					delete.append(key)
			for key in delete:
				del self.mecs[key]


	def compute(self, plt_dir=None, rsmp=False, frac=1, n=None, rplc=True):

		if len(self.sel) == 0:
			print('nothing selected')
			quit()

		if rsmp:
			self.model.resample(frac=frac, n=n, rplc=rplc)

		infos = []
		for idx in self.idx:
			vbls = [self.sel[v] for v in idx]
			infos.append(self.model.info(vbls, rsmp))

		ep = {'di': np.full([self.q,self.q], -np.inf, dtype=self.dtype),
			  'bi': np.full([self.q,self.q], -np.inf, dtype=self.dtype),
			  'nr': np.full([self.q,self.q], -np.inf, dtype=self.dtype)}

		tmp = np.full([self.q,self.q], -np.inf)

		norm = np.dtype(dtype=self.dtype).type(-np.inf)

		for key in self.mecs:

			score = np.sum([infos[s] for s in key])

			# prior: uniform over MECs 
			score -= len(self.mecs[key])

			for mag, anc in self.mecs[key]:

				adj = mag + 2*mag.T

				tmp[np.where(adj==1)] = score
				ep['di'] = np.logaddexp(ep['di'], tmp)
				tmp[np.where(adj==1)] = -np.inf

				tmp[np.where(adj==6)] = score
				ep['bi'] = np.logaddexp(ep['bi'], tmp)
				tmp[np.where(adj==6)] = -np.inf

				tmp[np.where(adj==0)] = score
				ep['nr'] = np.logaddexp(ep['nr'], tmp)
				tmp[np.where(adj==0)] = -np.inf

				norm = np.logaddexp(norm, score)

		ep['di'] = np.exp(ep['di'] - norm)
		ep['bi'] = np.exp(ep['bi'] - norm)
		ep['nr'] = np.exp(ep['nr'] - norm)

		if plt_dir != None:
			plt_dir += '/'
			path = os.path.dirname(plt_dir)
			os.makedirs(path, exist_ok=True)
			for i in range(self.q):
				for j in range(i):
					plt.figure()
					plt.bar([self.sel[j]+' --> '+self.sel[i], 
						     self.sel[j]+' <-- '+self.sel[i], 
						     self.sel[j]+' <-> '+self.sel[i], 
						     self.sel[j]+' ... '+self.sel[i]], 
						    [ep['di'][i,j], ep['di'][j,i], 
						     ep['bi'][i,j], ep['nr'][i,j]])
					plt.xticks(rotation=7)
					plt.yticks(np.linspace(1,0,11))
					plt.ylim(0,1.1)
					plt.savefig(path+'/'+self.sel[j]+'_'+self.sel[i])
					plt.clf()

		return ep


	def resample(self, reps, plt_dir=None, frac=1, n=None, rplc=True):

		rslts = {}
		for i in range(self.q):
			for j in range(i):
				rslts[(self.sel[j], self.sel[i])] = []

		for rep in range(reps):
			ep = self.compute(rsmp=True, frac=frac, n=n, rplc=rplc)
			for i in range(self.q):
				for j in range(i):
					rslts[(self.sel[j], self.sel[i])].append(
						[ep['di'][i,j], ep['di'][j,i], 
						 ep['bi'][i,j], ep['nr'][i,j]])


		if plt_dir != None:
			plt_dir += '/'
			path = os.path.dirname(plt_dir)
			os.makedirs(path, exist_ok=True)
			for i in range(self.q):
				for j in range(i):
					plt.figure()
					plt.boxplot(np.array(rslts[(self.sel[j], self.sel[i])]), 
						labels=[self.sel[j]+' --> '+self.sel[i], 
						     	self.sel[j]+' <-- '+self.sel[i], 
						     	self.sel[j]+' <-> '+self.sel[i], 
						     	self.sel[j]+' ... '+self.sel[i]])
					plt.xticks(rotation=7)
					plt.yticks(np.linspace(1,0,11))
					plt.ylim(-0.1,1.1)
					plt.savefig(path+'/'+self.sel[j]+'_'+self.sel[i])
					plt.clf()

		return rslts