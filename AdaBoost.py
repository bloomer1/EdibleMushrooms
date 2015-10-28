import sys
import math

def read_trainingdata(train_file):
	file_ptr = open(train_file,'r')

	file_ptr = file_ptr.read()
	raw_dataset = file_ptr.rstrip('\t').splitlines()
	dataset = [[]]
	for datapoint in raw_dataset:
		datapoint = datapoint.split('\t')
		dataset.append(datapoint)
	dataset = dataset[1:]
	return dataset

def attr_val(dataset):
	att_val_list = dict()
	for datapoint in dataset:
			datapoint = datapoint[1:]
			for idx,attr in enumerate(datapoint):
				if idx not in att_val_list:
					att_val_list[idx] = set()
					att_val_list[idx].add(attr)
				else:
					att_val_list[idx].add(attr)

	'''
	for key,val in att_val_list.items():
		for v in val:
			print key,v
	'''
	return att_val_list
'''
Debug function
def x_count(dataset):
	x_c = 0
	e_c = 0
	for datapoint in dataset:
		if datapoint[1] == 'x':
			x_c = x_c + 1
		if datapoint[0] == 'e' and datapoint[1] == 'x':
			e_c = e_c + 1


	print x_c
	print e_c
'''
def preprocess_trainingdata(dataset):
	
	att_val_lbl = dict()

	for ridx,datapoint in enumerate(dataset):
		label = datapoint[0]
		for cidx,attr in enumerate(datapoint[1:]):
			key = str(cidx) + attr + str(ridx)
			att_val_lbl[key] = label

	#print att_val_lbl

	return att_val_lbl
	


def train(att_val_lbl,T,att_val_list,N,dataset):

	wts_vec = dict()
	for i in range(N):
		wts_vec[i] = 1/float(N)
	total_wts = 1

	p_label = 0
	e_label = 0
	att_count = 21

	for datapoint in dataset:
		if datapoint[0] == 'p':
			p_label = p_label + 1
		else:
			e_label = e_label + 1

	Proba_Plabel = (p_label)/float(p_label + e_label)
	Proba_Elabel = (e_label)/float(p_label + e_label)
	
	entropy_dataset = (-Proba_Plabel*math.log(Proba_Plabel,2)) + (-Proba_Elabel*math.log(Proba_Elabel,2))
	
	
	
	for iterations in range(T):
		gain_list = list()
		for key,val_list in att_val_list.items():
			s_value = dict()
			e_a_v = dict()
			for val in val_list:
				e_wts = 0.0
				p_wts = 0.0
				
				for i in range(N):
					k = str(key) + val + str(i)
					if k in att_val_lbl:
						if att_val_lbl[k] == 'e':
							e_wts = e_wts + wts_vec[i]
						else:
							p_wts = p_wts + wts_vec[i]
				attval = str(key) + val
				s_value[attval] = (e_wts + p_wts)
				Proba_Plabel = p_wts/float(p_wts + e_wts)
				Proba_Elabel = e_wts/float(p_wts + e_wts)
				if Proba_Plabel != 0.0 and Proba_Elabel != 0.0:
					entropy_attr_val = (-Proba_Plabel*math.log(Proba_Plabel,2)) + (-Proba_Elabel*math.log(Proba_Elabel,2))
				elif Proba_Plabel != 0.0:
					entropy_attr_val = 0.0
				else:
					entropy_attr_val = 0.0
				#print key,val
				#print entropy_attr_val
				e_a_v[attval] = entropy_attr_val
			#print s_value
			#print e_a_v
			raw_gain = 0.0
			gain = 0.0
			for k,v in e_a_v.items():
				raw_gain = raw_gain + (abs(s_value[k])/float(N))*e_a_v[k]
			gain =  entropy_dataset - raw_gain
			gain_list.append(gain)

		#print gain_list
		max_att_val = gain_list[0]
		max_att_idx = 0
		for cidx, gain in enumerate(gain_list):
			if gain > max_att_val:
				max_att_val = gain
				max_att_idx =  cidx

		print max_att_idx
		val_list = att_val_list[max_att_idx]
		p_count = 0
		e_count = 0
		# code improvement needed, uncessary iterations
		for val in val_list:
			for datapoint in dataset:
				if datapoint[max_att_idx + 1] == val and datapoint[0] == 'e':
					e_count = e_count + 1
				elif datapoint[max_att_idx + 1] == val:
					p_count = p_count + 1
			print val, e_count, p_count
			p_count = 0
			e_count = 0




				

		return	

				
				

					
				

	



		










def main():

	if len(sys.argv) != 4:
		print "Usage: <no of boosting iterations> <traiining_file> <testing_file>"
		return

	T = int(sys.argv[1])
	train_file = sys.argv[2]
	test_file = sys.argv[3]

	#print T
	#print train_file
	#print test_file

	dataset = read_trainingdata(train_file)
	att_val_list = attr_val(dataset)
	#print dataset[0][0]	
	#x_count(dataset)
	att_val_lbl = preprocess_trainingdata(dataset)
	#print len(dataset)
	decisionTree_map = train(att_val_lbl,T,att_val_list,len(dataset),dataset)







if __name__ == '__main__':
	main()