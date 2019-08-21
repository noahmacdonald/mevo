import datetime
from dateutil.relativedelta import *
import pprint as p

old = open("resurvey3.csv", "r")
ind = open("ind.1", "r")
new = open("test.2", "w+")

def build_table():
	ind_rep = {}
	for line in ind:
		arr = line.split(",")
		arr[1] = arr[1][:-1]
		ind_rep[arr[0]] = arr[1]

	p.pprint(ind_rep)

	for line in old:
		arr = line.split(",")
		cc = arr[5]
		date = arr[1] + '-01-' + arr[2]
		date = datetime.datetime.strptime(date, '%m-%d-%Y')
		if cc == "2":
			date = date + relativedelta(months=+2)
		if cc == "3":
			date = date + relativedelta(months=+5)
		if cc == "4":
			date = date + relativedelta(months=+9)
		if cc == "5":
			date = date + relativedelta(months=+12)
		to_reg = date.strftime("%m-%d-%Y")
		ind_when = ind_rep.get(to_reg)
		arr.append(to_reg)
		arr[73] = arr[73][:-1]
		if arr[74] != None:
			for i in range(0, len(arr)):
				new.write(arr[i])
				if i < 74:
					new.write(',')
				else:
					new.write('\n')



# import pandas as pd
# # Read in data and display first 5 rows
# features = pd.read_csv('resurvey3.csv')
# h = features.head(5)
# print(h)

build_table()



