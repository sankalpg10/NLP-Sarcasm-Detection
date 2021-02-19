# test train split 70-30(test)
#  70 into 70-30(Eval)

#  output - 3 files train.csv, 

from sklearn.model_selection import train_test_split
import argparse
import json 
import csv 
import os
import io
  
def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument(
		"--data_path",
		default=None,
		type=str,
		required=True,
		help="The input data directory which contains the consolidated data in json format",
	)

	parser.add_argument(
		"--test_fraction",
		default=0.3,
		type=float,
		help="Fraction of the entire data which is to be considered as test data",
	)

	parser.add_argument(
		"--val_fraction",
		default=0.3,
		type=float,
		help="Fraction of train+validation data to be considered as validation data",
	)
	args = parser.parse_args()
	split_files(args)


def split_files(args):
	# Opening JSON file and loading the data 

	headlines = []
	labels=[]
	for line in open(os.path.join(args.data_path,'Sarcasm_Headlines_Dataset_v2 - Copy.json'), 'r'):
		data=json.loads(line)
		headlines.append("\""+data['headline']+"\"")
		labels.append(data['is_sarcastic'])

	print(len(headlines))
	print(headlines[14])
	print(labels[14])


	X_trval, X_test, Y_trval, Y_test = train_test_split(headlines, labels, test_size=args.test_fraction, random_state=42) #split whole data into train+eval and test data 
	X_train, X_val, Y_train, Y_val = train_test_split(X_trval, Y_trval, test_size=args.val_fraction, random_state=42) #split train+eval data into train and eval data


	# train validation and test files

	train_file = open(os.path.join(args.data_path,'train.csv'), 'w') 
	validation_file = open(os.path.join(args.data_path,'valid.csv'), 'w') 
	test_file = open(os.path.join(args.data_path,'test.csv'), 'w') 

	csv_train = csv.writer(train_file, delimiter=',',quotechar="\"") 
	csv_validation = csv.writer(validation_file, delimiter=',',quotechar="\"") 
	csv_test = csv.writer(test_file, delimiter=',',quotechar="\"") 

	# header=["Headlines","label"]
	# csv_train.writerow(header) 
	# csv_validation.writerow(header) 
	# csv_test.writerow(header) 


	for i in range(len(X_train)): 
		csv_train.writerow([X_train[i],Y_train[i]]) 
	train_file.close() 

	for i in range(len(X_val)): 
		csv_validation.writerow([X_val[i],Y_val[i]]) 
	validation_file.close() 

	for i in range(len(X_test)): 
		csv_test.writerow([X_test[i],Y_test[i]]) 
	test_file.close() 


if __name__ == "__main__":
	main()


