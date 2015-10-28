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
	


def train(att_val_lbl,T,att_val_list,N,dataset,alpha_list):

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
	epsilon_list = []
	
	learned_decision_tree = dict()
	
	
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

		#print max_att_idx
		val_list = att_val_list[max_att_idx]
		val_lbl = dict()
		p_count = 0
		e_count = 0
		# code improvement needed, uncessary iterations
		for val in val_list:
			for datapoint in dataset:
				if datapoint[max_att_idx + 1] == val and datapoint[0] == 'e':
					e_count = e_count + 1
				elif datapoint[max_att_idx + 1] == val:
					p_count = p_count + 1
			
			if e_count > p_count:
				val_lbl[val] = 'e'
			else:
				val_lbl[val] = 'p'


			#print val, e_count, p_count
			p_count = 0
			e_count = 0
		#print val_lbl
		learned_decision_tree[str(max_att_idx) + '#' + str(iterations)] = val_lbl


		epsilon = 0.0
		alpha = 0.0
		
		for ridx,datapoint in enumerate(dataset):
			attr = datapoint[max_att_idx+1]
			#print val_lbl[attr], datapoint[0]
			if datapoint[0] != val_lbl[attr]:
				epsilon = epsilon + wts_vec[ridx]
				
		#print epsilon
		epsilon_list.append(epsilon)
		alpha = (1/float(2))*math.log((1 - epsilon)/epsilon)
		#print alpha
		alpha_list.append(alpha)

		#print wts_vec
		for ridx,datapoint in enumerate(dataset):
			attr = datapoint[max_att_idx+1]
			if datapoint[0] != val_lbl[attr]:
				wts_vec[ridx] = wts_vec[ridx]*math.exp((-alpha)*(-1))
				
			else:
				wts_vec[ridx] = wts_vec[ridx]*math.exp((-alpha)*(1))
		#print wts_vec

		sum_wts = 0.0
		for key,val in wts_vec.items():
			sum_wts = sum_wts + val

		#print sum_wts

		for key,val in wts_vec.items():
			wts_vec[key] = wts_vec[key]/float(sum_wts)
			#print key, wts_vec[key]


	#print learned_decision_tree
	#print alpha_list
	return learned_decision_tree




def test(test_file,learned_decision_tree,alpha_list):

	testset = read_trainingdata(test_file)
	predicted_table = dict()
	accuracy = 0.0
	correct_records = 0
	total_records = len(testset)
	#print total_records
	
	#print learned_decision_tree
	#print alpha_list
	
	
	for ridx,testpoint in enumerate(testset):
		res_list = []
		for key,val in learned_decision_tree.items():
			attr = key.split('#')[0]
			test_attr_val = testpoint[int(attr)+1]
			#print attr, test_attr_val
			train_attr_lbl = val[test_attr_val]
			res_list.append(train_attr_lbl)
		#print res_list
		final_res = 0.0
		for idx,alpha_val in enumerate(alpha_list):
			if res_list[idx] == 'e':
				final_res = final_res + alpha_val*(1)
			else:
				final_res = final_res + alpha_val*(-1)

		if final_res > 0:
			predicted_table[ridx] = 'e'
		else:
			predicted_table[ridx] = 'p'
	'''
	for key,val in predicted_table.items():
		print key,val
	'''

	for ridx,testpoint in enumerate(testset):
		if testpoint[0] == predicted_table[ridx]:
			correct_records = correct_records + 1


	accuracy = correct_records/float(total_records)
	return accuracy


	












				

		

				
				

					
				

	



		










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
	alpha_list = []
	#print len(dataset)
	learned_decision_tree = train(att_val_lbl,T,att_val_list,len(dataset),dataset,alpha_list)
	#print alpha_list
	#print learned_decision_tree

	accuracy = test(test_file,learned_decision_tree,alpha_list)

	print accuracy

	for alpha_val in alpha_list:
		print alpha_val










if __name__ == '__main__':
	main()