from flask import Flask, render_template, url_for, jsonify, request
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import datetime
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import json
import csv
from ast import literal_eval
 
application = Flask(__name__)

init = False
rendered = 0



@application.route('/<string:status>/', methods = ['GET', 'POST'])
def index(status):
	global init
	global rendered

	data = None

	headings = ['month', 'day', 'year', 'respondents', 'female', 'male', 'when_buying', 'buying', 'leasing', 'undecided', 'no_thought', 'my_choice', 'shared_choice', 'their_choice', 'me_driving', 'spouse_driving', 'kid_driving', 'other_driving', 'last_90', 'last_5mos', 'last_9mos', 'last_year', 'just_now', 'recognizing', 'researching', 'comparing', 'almost_chosen', 'chosen', 'r_family', 'r_newsize', 'r_safety', 'r_newer', 'r_deal', 'r_leaseup', 'r_needcar', 'r_tech', 'r_fuel', 'r_reliable', 'r_accident', 'r_luxury', 'r_bettercar', 'r_status', 'subcompact', 'smallcar', 'midsize', 'minivan', 'small_suv', 'midsize_suv', 'large_suv', 'small_pickup', 'large_pickup', 'sports', 'luxury', 'lux_suv', 'ultra_lux', 'perf_lux', 'cnt_considered_calc', 'electric', 'autonomous', 'cnt_considered_self', 'last_bought90d', 'last_bought1y', 'last_bought3y', 'last_bought5y', 'last_bought_5y', 'last_boughtnever', 'never_mar', 'married', 'widowed', 'divorced', 'poor', 'middle', 'rich', '1model', '1_model', 'gdp', 'crude_oil', 'cpi', 'un_emp', 'actual']

	if request.method == 'POST':
		file = open("static/conf.csv", "w")
		file.write("month,day,year,respondents,female,male,when_buying,buying,leasing,undecided,no_thought,my_choice,shared_choice,their_choice,me_driving,spouse_driving,kid_driving,other_driving,last_90,last_5mos,last_9mos,last_year,just_now,recognizing,researching,comparing,almost_chosen,chosen,r_family,r_newsize,r_safety,r_newer,r_deal,r_leaseup,r_needcar,r_tech,r_fuel,r_reliable,r_accident,r_luxury,r_bettercar,r_status,subcompact,smallcar,midsize,minivan,small_suv,midsize_suv,large_suv,small_pickup,large_pickup,sports,luxury,lux_suv,ultra_lux,perf_lux,cnt_considered_calc,electric,autonomous,cnt_considered_self,last_bought90d,last_bought1y,last_bought3y,last_bought5y,last_bought_5y,last_boughtnever,never_mar,married,widowed,divorced,poor,middle,rich,1model,1_model,gdp,crude_oil,cpi,un_emp,actual")
		file.write("\n")
		for li in request.json:
			s = ""
			i = 0
			for head in headings:
				s += li[head]
				if i <= 78:
					s += ","
				i += 1
			file.write(s + "\n")
			i = 0
		file.close()

		features = pd.read_csv("static/conf.csv")
		features2 = features
		print("-----------IM HAPPENING1----------")
		# composite = features[0].to_json(orient='records')    // could print later, if wanted in console
		labels = np.array(features['actual'])
		features = features.drop('actual', axis = 1)
		feature_list = list(features.columns)
		features = np.array(features)
		train_features, test_features, train_labels, test_labels = train_test_split(features[:][0:42], labels[0:42], test_size = 0.25, random_state = 41)
		rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
		print("-----------IM HAPPENING2----------")
		rf.fit(train_features, train_labels);
		print("-----------IM HAPPENING3----------")
		predictions = rf.predict(test_features)
		errors = abs(predictions - test_labels)
		importances = list(rf.feature_importances_)
		feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
		feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
		print("-----------IM HAPPENING4----------")
		# mape = 100 * (errors / test_labels)
		# accuracy = 100 - np.mean(mape)
		months = features[:, feature_list.index('month')]
		days = features[:, feature_list.index('day')]
		years = features[:, feature_list.index('year')]
		dates = []
		for year, month, day in zip(years, months, days):
			dates.append(datetime.datetime.strptime('{0:g}'.format(year) + '-' + '{0:g}'.format(month) + '-' + '{0:g}'.format(day), '%Y-%m-%d'))
		print("-----------IM HAPPENING5----------")
		dates = ['{0:g}'.format(year) + '-' + '{0:g}'.format(month) + '-' + '{0:g}'.format(day) for year, month, day in zip(years, months, days)];
		dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates];
		print("-----------IM HAPPENING6----------")
		true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
		print("-----------IM HAPPENING7----------")
		months = test_features[:, feature_list.index('month')]
		print("-----------IM HAPPENING8----------")
		days = test_features[:, feature_list.index('day')]
		print("-----------IM HAPPENING9----------")
		years = test_features[:, feature_list.index('year')]
		print("-----------IM HAPPENING10----------")
		test_dates = ['{0:g}'.format(year) + '-' + '{0:g}'.format(month) + '-' + '{0:g}'.format(day) for year, month, day in zip(years, months, days)]
		# Convert to datetime objects
		test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
		predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})
		plt.plot(true_data['date'][0:42], true_data['actual'][0:42], 'b-', label = 'actual', color="#004D73")
		plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction', color="#ED7423")
		plt.xticks(rotation = '60')
		plt.legend()
		plt.xlabel('Date')
		plt.ylabel('Car Sales')
		plt.title('A Model for Understanding Car Sales')
		plt.savefig('static/inter1.png')
		plt.close()
		features = features2
		# composite = features[0].to_json(orient='records')    // could print later, if wanted in console
		labels = np.array(features['actual'])
		features = features.drop('actual', axis = 1)
		feature_list = list(features.columns)
		features = np.array(features)
		train_labels = np.array(labels[0:42])
		test_labels = np.array(labels[42:-1])
		train_features = np.array(features[:][0:42])
		test_features = np.array(features[:][42:-1])
		rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
		rf.fit(train_features, train_labels);
		predictions = rf.predict(test_features)
		errors = abs(predictions - test_labels)
		importances = list(rf.feature_importances_)
		feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
		feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
		# mape = 100 * (errors / test_labels)
		# accuracy = 100 - np.mean(mape)
		months = features[:, feature_list.index('month')]
		days = features[:, feature_list.index('day')]
		years = features[:, feature_list.index('year')]
		dates = []
		# for year, month, day in zip(years, months, days):
		# 	if month != 'Untitled':
		# 		dates.append(str(int(year)) + '-' + str(month) + '-' + str(day))
		dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
		# dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
		true_data = pd.DataFrame(data = {'date': dates[:42], 'actual': labels[:42]})
		months = test_features[:, feature_list.index('month')]
		days = test_features[:, feature_list.index('day')]
		years = test_features[:, feature_list.index('year')]
		test_dates = []
		# for year, month, day in zip(years, months, days):
		# 	if year != 'Untitled':
		# 		test_dates.append(str(int(year)) + '-' + str(month) + '-' + str(day))
		test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
		# test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
		predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})
		plt.plot(true_data['date'][0:42], true_data['actual'][0:42], 'b-', label = 'actual', color="#004D73")
		plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction', color="#ED7423")
		print(predictions_data)
		plt.xticks(rotation = '60')
		plt.legend()
		plt.xlabel('Date')
		plt.ylabel('Car Sales')
		plt.title('A Model for Predicting Car Sales')

		plt.savefig('static/extra1.png')
		plt.close()

	if not init:
		data = pd.read_csv('survey.csv')
		init = True
	else:
		data = pd.read_html("static/data.html")[0]

	if status == 'best':
		data = pd.read_csv('static/best_data.csv')
		under = 'best_under.png'
		pred = 'best_pred.png'
	elif status == 'worst':
		data = pd.read_csv('static/worst_data.csv')
		under = 'best_under.png'
		pred = 'worst_pred.png'
	elif status == 'recalibrate':
		data = pd.read_csv('survey.csv')
		under = 'understanding.png'
		pred = 'prediction.png'
	elif status == 'composite':
		data = pd.read_csv("static/conf.csv")
		under = 'inter1.png'
		pred = 'extra1.png'
	else:
		under = 'understanding.png'
		pred = 'prediction.png'

	data = data[0:50]
	rendered = render_template('index.html', data=data, under=under, pred=pred)
	file = open("static/data.html", "w")
	file.write(rendered)
	return rendered
 
if __name__ == "__main__":
	application.run(debug=True)







