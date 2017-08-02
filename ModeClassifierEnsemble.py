#Jenny Steffens 
#7.12.17
#An ensemble of classification algorithms to be used for the e-missions project
# import logging

# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG, filename='MCEFunctionsoutput.log')
from warnings import warn
import pandas as pd
import numpy as np
from time import clock
import sys


class ModeClassifier:
	def __init__(self, Model=None, training_function=None, scoring_function=None, prediction_function=None, probability_function=None, update_with_personal_data=True):
		if Model is None:
			raise TypeError("Model cannot be none")

		self.Model = Model
		self.update_with_personal_data = update_with_personal_data

		if not training_function: 
			warn("classifier not given a training function, reverting to Sci-kit Learn defaults....")
			self.training_function = self.Model.fit
		else:
			self.training_function = training_function

		if not scoring_function: 
			warn("classifier not given a scoring function, reverting to Sci-kit Learn defaults....")
			self.scoring_function = self.Model.score
		else:
			self.scoring_function = scoring_function

		if not prediction_function: 
			warn("classifier not given a prediction function, reverting to Sci-kit Learn defaults....")
			self.prediction_function = self.Model.predict
		else:
			self.prediction_function = prediction_function

		if not probability_function: 
			warn("classifier not given a prediction function, reverting to Sci-kit Learn defaults....")
			self.probability_function = self.Model.predict_proba
		else:
			self.probability_function = probability_function

	def train(self, *args):
		self.Model = self.training_function(*args)

	def predict(self, *args):
		return self.prediction_function(*args)

	def test(self, *args):
		return self.testing_function(*args)

	def predict_proba(self, *args):
		predictions_prob_df = {}
		proba = self.probability_function(*args)
		i = 0
		for classlabel in self.Model.classes_:
			predictions_prob_df[classlabel] = proba[:,i]
			i += 1
		return predictions_prob_df

	def getModelClasses(self, return_type = 'list'):
		if return_type is 'list':
			return list(self.Model.classes_)
		elif return_type is 'numpy_array':
			return np.array(self.Model.classes_)
		else:
			raise ValueError("return_type must be 'list' or 'numpy_array'. Recieved %s " % return_type)
			return

class ModeClassifierEnsemble:
	def __init__(self):
		self._Classifiers 						= {}
		self._Aggregators 						= {}
		self._target_category_label_indexes 	= {}
		self._algorithm_category_label_indexes	= {}
		self._ensemble_training_set_dict		= {}
		self._algorithm_category_labels 		= []
		self._target_category_labels 			= []
		self._classifier_count			 		= 0
		self._naive_mean_weights 				= None
		self._IntelligentWeighter 				= None
		self._BestAggregator 					= None
		self._bus_cluster						= None
		self._train_cluster						= None

	def getClassifiers(self):
		return self._Classifiers

	def getClassifierNames(self):
		return self._Classifiers.keys()

	def getLabels(self):
		return self._algorithm_category_labels
		
	def addClassifier(self, Classifier, classifier_name): 
		if not isinstance(Classifier, ModeClassifier):
			raise TypeError("Classifier %s is not of type ModeClassifier" % classifier_name)
		if classifier_name in self._Classifiers:
			raise ValueError("classifier name %s is already in the Ensemble" % classifier_name)
			return
		self._Classifiers[classifier_name] = Classifier
		self._classifier_count += 1

	def deleteClassifier(self, classifier_name):
		if classifier_name in self._Classifiers:
			del self._Classifiers[classifier_name]
			self._classifier_count -= 1

	def updateClassifier(self, new_Classifier, classifier_name):
		self._Classifiers[classifier_name] = new_Classifier

	def standardizeClassLabels(self, labels = None):
		if labels is not None: 
			self._target_category_labels = labels

		else:
			all_labels = []
			for classifier_name in self._Classifiers.iterkeys():
				if all_labels == []:
					all_labels = self._Classifiers[classifier_name].getModelClasses(return_type='numpy_array')
				else:
					all_labels = np.append(all_labels, self._Classifiers[classifier_name].getModelClasses(return_type='numpy_array'))
			self._target_category_labels = np.unique(all_labels)

		for classifier_name in self._Classifiers.iterkeys():
			self._Classifiers[classifier_name].setModelClasses(self._target_category_labels)

		self._target_category_label_indexes = {}
		index = 0
		
		for label in self._target_category_labels:
			self._target_category_label_indexes[label] = index
			index += 1

	def addAggregator(self, AggregatorFunction, aggregator_name):
		if not callable(AggregatorFunction):
			raise TypeError("Aggregator %s is not a function" % aggregator_name)
		if aggregator_name in self._Aggregators:
			raise ValueError("Aggregator %s is already in the ensemble" % aggregator_name)
			return
		self._Aggregators[aggregator_name] = AggregatorFunction

	def deleteAggregator(self, aggregator_name):
		if aggregator_name in self._Aggregators:
			del self._Aggregators[aggregator_name]

	def setBestAggregator(self, aggregator_name):
		if aggregator_name not in self._Aggregators:
			warn( "%s is not in Aggregators. Please add before setting. Best Aggregator is currently %s " % (aggregator_name, self._BestAggregator))
		self._BestAggregator = aggregator_name

	def setEnsembleTrainingData(self, training_examples, training_target):
		for classifier in self._Classifiers.iterkeys():
			self._ensemble_training_set_dict[classifier] = [training_examples, training_target]
		
	def addToEnsembleTrainingData(self, feature_df_row, ground_truth):
		for classifier in self._Classifiers.iterkeys():
			self._ensemble_training_set_dict[classifier_name][0] = self._ensemble_training_set_dict[classifier_name][0].append(feature_df_row)
			self._ensemble_training_set_dict[classifier_name][1] = self._ensemble_training_set_dict[classifier_name][1].append(pd.Series(data=ground_truth, index=feature_df_row.index))

	def setTrainingDataForClassifier(self, classifier_name, training_examples, training_target):
		if classifier_name not in self._Classifiers.keys():
			raise KeyError("%s is not in the list of ModeClassifiers. Must be added before training data is set")
			return
		self._ensemble_training_set_dict[classifier_name] = [training_examples, training_target]

	def addToClassifierTrainingData(self, feature_df_row, ground_truth, classifier_name):
		if classifier_name not in self._ensemble_training_set_dict:
			raise KeyError("training data for %s not initialized" % classifier_name)
			return
		self._ensemble_training_set_dict[classifier_name][0] = self._ensemble_training_set_dict[classifier_name][0].append(feature_df_row)
		self._ensemble_training_set_dict[classifier_name][1] = self._ensemble_training_set_dict[classifier_name][1].append(pd.Series(data=ground_truth, index=feature_df_row.index))


	def trainEnsemble(self, classifier_names=None, training_examples=None, training_target=None, standardize_class_labels=True):
		#The training data must be in the form of a pandas dataframe
		all_labels = []
		if classifier_names is None:
			classifier_names = self._Classifiers.keys()
		
		for classifier_name in classifier_names:
			if training_examples is None:
				training_examples = self._ensemble_training_set_dict[classifier_name][0]
			if training_target is None:
				training_target = self._ensemble_training_set_dict[classifier_name][1]

			self._Classifiers[classifier_name].train(training_examples, training_target)
			classes = self._Classifiers[classifier_name].getModelClasses(return_type='numpy_array')
			if all_labels == []:
				all_labels = classes
			else:
				all_labels = np.append(all_labels, classes)

		self._target_category_labels = np.unique(all_labels)

	def predict(self, prediction_examples, return_df=True):
		predictions = {}
		for classifier_name in self._Classifiers.iterkeys():
			predictions[classifier_name] = self._Classifiers[classifier_name].predict(prediction_examples)
		if return_df:
			return pd.DataFrame(predictions)
		return predictions

	def predictProb(self, prediction_examples):
		predictions_prob_df = pd.DataFrame(index=prediction_examples.index)	

		for classifier_name in self._Classifiers.iterkeys():
			classifier_prob_dict = self._Classifiers[classifier_name].predict_proba(prediction_examples)
			classifier_prob_df = pd.DataFrame(classifier_prob_dict, index=prediction_examples.index)
			new_column_names = list(classifier_prob_df.columns.values)
			for idx in range(len(new_column_names)):
				new_column_names[idx] = classifier_name + "_" + str(new_column_names[idx]) 

			classifier_prob_df.columns = new_column_names
			predictions_prob_df = pd.concat([predictions_prob_df, classifier_prob_df], axis=1)

		return predictions_prob_df

################

	def generateWinnerColumn(self, training_examples, training_target):
		logging.debug("Generating winner column")
		predictions_prob_df = self.predictProb(training_examples)
		winner_list = []
		row_winner = ""
		target_name = training_target.name
		logging.debug("training target name: %s" % target_name)

		for index, correct_class in training_target.iteritems(): 
			max_confidence = -1
			for classifier_name in self._Classifiers.iterkeys():
				try:
					row_confidence_correct = predictions_prob_df[classifier_name+"_"+str(correct_class)].loc[index]
				except KeyError:
					row_confidence_correct = 0.0

				if row_confidence_correct > max_confidence:
					max_confidence = row_confidence_correct
					row_winner = classifier_name
			winner_list.append(row_winner)

		winner_list = np.array(winner_list)
		count_winners = 0
		for idx, label in enumerate(np.unique(winner_list)):
			self._algorithm_category_label_indexes[idx] = label
			self._algorithm_category_labels.append(label)
			count_winners += 1

		if count_winners != self._classifier_count:
			warn("Not all algorithms accounted for in winner list. Proceed with caution.")
			logging.debug("Not all algorithms accounted for in winner list. Proceed with caution.")

		return winner_list

	def trainIntelligentWeighter(self, Model, training_examples, training_target):
		self._IntelligentWeighter = Model.fit(training_examples, self.generateWinnerColumn(training_examples, training_target))

	def generateWeightsForRows(self, example):
		weights_df = pd.DataFrame(self._IntelligentWeighter.predict_proba(example), columns=self._IntelligentWeighter.classes_, index=example.index)
		for classifier_name in self._Classifiers.iterkeys():
			if classifier_name not in weights_df:
				filler_df = pd.DataFrame(np.zeros(len(example)), index=weights_df.index, columns=[classifier_name])
				weights_df = pd.concat([filler_df, weights_df], axis=1)
		return weights_df

	def generateNaiveWeights(self, testing_examples, target_column):
		algorithm_prediction_dataframe = self.predict(testing_examples)
		weights = pd.DataFrame()
		total_correct = 0.0
		for classifier_name in algorithm_prediction_dataframe:	
			logging.debug("In generateNaiveWeights")
			logging.debug("algorithm prediction dataframe for %s is %s" % (classifier_name, algorithm_prediction_dataframe[classifier_name]))
			logging.debug("shape is %s" % str(algorithm_prediction_dataframe[classifier_name].shape))
			logging.debug("Target column is %s:" % target_column)	
			logging.debug("shape is %s" % str(target_column.shape))
			logging.debug("algorithm_prediction_dataframe with reset index is %s" % algorithm_prediction_dataframe[classifier_name].reset_index(drop=True))
			logging.debug("target_column with reset index is %s" % target_column.reset_index(drop=True))
			column_data = (algorithm_prediction_dataframe[classifier_name].reset_index(drop=True)) == (target_column.reset_index(drop=True))

			num_correct = np.sum(column_data.astype(float))
			logging.debug("num_correct is %s" % str(num_correct))
			weights[classifier_name] = np.array([num_correct])
			total_correct += num_correct
			logging.debug("algorithm prediction dataframe for %s is %s" % (classifier_name, algorithm_prediction_dataframe[classifier_name]))

			logging.debug("Target column is %s:" % target_column)	
		logging.debug("total_correct is %s" % str(total_correct))

		for classifier_name in self._Classifiers.iterkeys():
			weights[classifier_name] = np.around((weights[classifier_name]/total_correct), decimals=3)

		weights['total_correct'] = total_correct
		self._naive_mean_weights = weights

		return weights


################ Main Aggregation Functions

	def unweightedMeanProb(self, prediction_examples, predictions_prob_df=None, sign_with_label=False):	
		if predictions_prob_df is None:
			predictions_prob_df = self.predictProb(prediction_examples)
		mean_prob = pd.DataFrame(index=prediction_examples.index)
		
		for label in self._target_category_labels:
			columns_to_add = []
			for classifier_name in self._Classifiers.iterkeys():
				column = classifier_name+"_"+str(label)
				if column in predictions_prob_df:
					columns_to_add.append(column)
			
			label_mean = ((predictions_prob_df[columns_to_add].sum(axis = 1))/self._classifier_count).round(decimals=3)
			mean_columns = list(mean_prob.columns.values)
			mean_columns.append(label)
			mean_prob = pd.concat([mean_prob, label_mean], axis=1)
			mean_prob.columns = mean_columns
		
		if sign_with_label:
			columns = list(mean_prob.columns.values)
			for idx, column_name in enumerate(columns):
				columns[idx] = sign_with_label + "_" + columns[idx]
			mean_prob.columns = columns
			logging.debug("mean prob: %s" % mean_prob)

			return mean_prob

		logging.debug("mean prob: %s" % mean_prob)
		return mean_prob

	def intelligentWeightedMeanProb(self, prediction_examples, predictions_prob_df=None, sign_with_label=False):
		if self._IntelligentWeighter is None:
			raise TypeError("IntelligentWeighter must be trained before intelligentWeightedMeanProb can be called.")
		
		if predictions_prob_df is None:
			predictions_prob_df = self.predictProb(prediction_examples)

		intelligent_weights_df = self.generateWeightsForRows(prediction_examples)
		weighted_prob = pd.DataFrame(index=predictions_prob_df.index)

		for label in self._target_category_labels:	
			weighted_prob[label] = np.zeros(len(prediction_examples))
			for classifier_name in self._Classifiers.iterkeys():
				column = classifier_name+"_"+str(label)
				if column in predictions_prob_df:
					logging.debug("processing intelligent weighted prob %s for classifier %s" % (str(label), classifier_name))
					weighted_prob[label] = np.round(weighted_prob[label] + ((intelligent_weights_df[classifier_name])*(predictions_prob_df[classifier_name+"_"+str(label)])), decimals=3)
			
		if sign_with_label:
			columns = list(weighted_prob.columns.values)
			for idx, column_name in enumerate(columns):
				columns[idx] = sign_with_label + "_" + columns[idx]
			weighted_prob.columns = columns
			logging.debug("weighted prob: %s" % weighted_prob)

			return weighted_prob

		logging.debug("weighted prob: %s" % weighted_prob)
		return weighted_prob
		
	def naiveWeightedMeanProb(self, prediction_examples, predictions_prob_df=None, sign_with_label=False):
		if self._naive_mean_weights is None:
			raise TypeError("Naive mean weights must first be calculated before naiveWeightedMeanProb can be called.")

		if predictions_prob_df is None:
			predictions_prob_df = self.predictProb(prediction_examples)

		logging.debug("Naive mean weights is %s" % self._naive_mean_weights)
		naive_prob = pd.DataFrame(index=predictions_prob_df.index)
		for label in self._target_category_labels:		
			naive_prob[label] = np.zeros(len(prediction_examples))
			for classifier_name in self._Classifiers.iterkeys():
				column = classifier_name+"_"+str(label)
				if column in predictions_prob_df:
					logging.debug("processing naive prob %s for classifier %s" % (str(label), classifier_name))
					naive_prob[label] = np.round(naive_prob[label] + ((self._naive_mean_weights[classifier_name].iloc[0])*(predictions_prob_df[classifier_name+"_"+str(label)])), decimals=3)
		
		if sign_with_label:
			columns = list(naive_prob.columns.values)
			for idx, column_name in enumerate(columns):
				columns[idx] = sign_with_label + "_" + columns[idx]
			naive_prob.columns = columns
			logging.debug("naive prob: %s" % naive_prob)

			return naive_prob
		logging.debug("naive prob: %s" % naive_prob)
		return naive_prob

################## Functions for the Confidence Test

	def generateAveragesDataframe(self, final_df):
		averages_df = pd.DataFrame()
		for column_name in final_df:
			new_column_name = str(column_name) + "_mean"
			averages_df[new_column_name] = final_df[column_name].mean()

		return averages_df

	def testConfidenceMeasures(self, testing_examples, target_column, predictions_prob_df=None, attach_predictions_prob_df = False, attach_agg_df = False, path_or_buf=None, timeit=True):
		start = clock()
		if predictions_prob_df is None:
			predictions_prob_df = self.predictProb(testing_examples)

		dataframe_list = [self.testConfidenceCorrectLabeling(testing_examples, target_column, predictions_prob_df=predictions_prob_df),
						  self.testConfidenceMagnitudeCorrectConfidence(testing_examples, target_column, predictions_prob_df=predictions_prob_df),
						  self.testConfidenceFirstAndSecondGuessDifference(testing_examples, target_column, predictions_prob_df=predictions_prob_df)]

		if attach_predictions_prob_df: 
			dataframe_list.append(predictions_prob_df)
		if attach_agg_df:
			for aggregator_name in self._Aggregators.iterkeys():
				agg_df = self._Aggregators[aggregator_name](testing_examples, predictions_prob_df=predictions_prob_df)
				new_columns = list(agg_df.columns.values)
				for idx in range(len(new_columns)):
					new_columns[idx] = str(aggregator_name) + "_" + str(new_columns[idx])
				logging.debug("new columns: %s" % new_columns)
				agg_df.columns = new_columns
				dataframe_list.append(agg_df)

		final_df = pd.concat(dataframe_list, axis=1)
		if path_or_buf is not None:
			final_df.to_csv(path_or_buf=path_or_buf)
			logging.debug("final dataframe stored in %s" % path_or_buf)
		if timeit: 
			dur = clock()-start
			print "testConfidenceMeasures took %s seconds" % str(dur)
			logging.debug("testConfidenceMeasures took %s seconds" % str(dur))

		return final_df

	def testConfidenceCorrectLabeling(self, testing_examples, target_column, predictions_prob_df=None):
		correctness_df 			= pd.DataFrame(columns=['answer'], index=testing_examples.index)
		predictions_prob_df     = self.predictProb(testing_examples)
		logging.debug("In testConfidenceCorrectLabeling")
		for aggregator_name in self._Aggregators.iterkeys():
			agg_df = self._Aggregators[aggregator_name](testing_examples, predictions_prob_df=predictions_prob_df)
			guess_df = agg_df.idxmax(axis=1)
			column_name = str(aggregator_name) + "_correct"
			column_data = (guess_df.reset_index(drop=True)) == (target_column.reset_index(drop=True))
			logging.debug("guess_df is %s" % guess_df)
			logging.debug("target_column is %s" % target_column)
			logging.debug("column data: %s" % column_data)
			astypefloat = column_data.astype(float)
			logging.debug("as type float %s" % astypefloat)
			asaseries = pd.Series(data=astypefloat.values, index=target_column.index, dtype=float)
			logging.debug("As a series: %s" % asaseries)
			correctness_df[column_name] = asaseries
			logging.debug("Column in correctness_df %s is %s" % (column_name, correctness_df[column_name]))


		correctness_df['answer'] = target_column
		logging.debug("correctness_df: %s" % correctness_df)
		
		return correctness_df


	def testConfidenceMagnitudeCorrectConfidence(self, testing_examples, target_column, predictions_prob_df=None):
		if predictions_prob_df is None:
			predictions_prob_df     = self.predictProb(testing_examples)	
		agg_df_dict		= {}
		column_names  	= ['agg_with_highest_confidence', 'highest_confidence']

		for aggregator_name in self._Aggregators.iterkeys():
			agg_df_dict[aggregator_name] = self._Aggregators[aggregator_name](testing_examples, predictions_prob_df=predictions_prob_df)
			column_names.append((str(aggregator_name) + "_confidence"))

		confidence_df	= pd.DataFrame(columns=column_names, index=testing_examples.index)	
		for example_idx, correct_class in target_column.iteritems():
			for aggregator_name in agg_df_dict:
				try:
					agg_confidence = agg_df_dict[aggregator_name][correct_class].loc[example_idx]		
				except KeyError:
					warn("Target category %s was not found for Aggregator %s" % str(correct_class), str(aggregator_name))
					logging.debug("Target category %s was not found for Aggregator %s" % (str(correct_class), str(aggregator_name)))
					agg_confidence = 0.0
				column_name = str(aggregator_name) + "_confidence"
				confidence_df[column_name].loc[example_idx] = agg_confidence

		max_confidence_alg 								= confidence_df.idxmax(axis=1)
		confidence 										= confidence_df.max(axis=1)
		confidence_df['agg_with_highest_confidence']	= max_confidence_alg
		confidence_df['highest_confidence']				= confidence

		logging.debug("confidence_df: %s" % confidence_df)
		return confidence_df

	def testConfidenceFirstAndSecondGuessDifference(self, testing_examples, target_column, predictions_prob_df=None):

		if predictions_prob_df is None:
			predictions_prob_df     = self.predictProb(testing_examples)
		difference_df = pd.DataFrame(index=testing_examples.index)

		for aggregator_name in self._Aggregators.iterkeys():
			agg_df 	= self._Aggregators[aggregator_name](testing_examples, predictions_prob_df=predictions_prob_df)
			a 		= agg_df.values

			first_and_second 									= a[np.arange(len(agg_df))[:,None],np.argpartition(-a,np.arange(2),axis=1)[:,:2]]
			difference_df[(str(aggregator_name)+"_first")] 		= first_and_second[:,0]
			difference_df[(str(aggregator_name)+"_second")] 	= first_and_second[:,1]
			difference_df[(str(aggregator_name)+"_difference")] = difference_df[(str(aggregator_name)+"_first")] - difference_df[(str(aggregator_name)+"_second")]

		logging.debug("difference_df: %s" % difference_df)
		return difference_df

############# Functions for the Threshold Test

	def testThresholdsAllAggregators(self, examples_dataframe, threshold_list, examples_target=None, path_or_buf=None, attach_target=False, time_since=None, trips_since=None, timeit=False, sign_with_label=True, column_names = ["distance", "duration", "first_filter_mode", "sectionId", "avg_speed",
						  "speed_EV", "speed_variance", "max_speed", "max_accel", "isCommute",
						  "heading_change_rate", "stop_rate", "velocity_change rate",
						  "start_lat", "start_lng", "stop_lat", "stop_lng",
						  "start_hour", "end_hour", "close_to_bus_stop", "close_to_train_stop",
						  "close_to_airport", "dataframe_index"]):

		start_all = clock()
		all_agg_thres_df 		 = pd.DataFrame(index=examples_dataframe.index)

		for aggregator_name, best_aggregator_func in self._Aggregators.iteritems():
			print "testing aggregator %s" % aggregator_name
			self.setBestAggregator(aggregator_name)
			agg_thres_df 		 = self.testThresholds(examples_dataframe, threshold_list, examples_target=examples_target, best_aggregator_func=best_aggregator_func, time_since=time_since, trips_since=trips_since, sign_with_aggregator=True, column_names=column_names )
			all_agg_thres_df	 = pd.concat([all_agg_thres_df, agg_thres_df], axis=1)
			
		if attach_target:
			all_agg_thres_df['target'] = examples_target

		if timeit:
			print "testThresholdsAllAggregators took %s seconds" % (clock()-start_all)

		if path_or_buf:
			all_agg_thres_df.to_csv(path_or_buf=path_or_buf)

		return all_agg_thres_df

	def testThresholds(self, examples_dataframe, threshold_list, examples_target=None, path_or_buf=None, attach_target=False, best_aggregator_func=None, time_since=None, trips_since=None, timeit=False, sign_with_label=True, sign_with_aggregator=False, column_names = ["distance", "duration", "first_filter_mode", "sectionId", "avg_speed",
						  "speed_EV", "speed_variance", "max_speed", "max_accel", "isCommute",
						  "heading_change_rate", "stop_rate", "velocity_change rate",
						  "start_lat", "start_lng", "stop_lat", "stop_lng",
						  "start_hour", "end_hour", "close_to_bus_stop", "close_to_train_stop",
						  "close_to_airport", "dataframe_index"]):

		start = clock()
		base_training_set_dict = self._ensemble_training_set_dict.copy()

		thresholds_df = pd.DataFrame(index=examples_dataframe.index)
		if best_aggregator_func is None:
			best_aggregator_func = self._Aggregators[self._BestAggregator]

		logging.debug("Best aggregator function is %s" % best_aggregator_func.__name__)
		for threshold in threshold_list:
			print "testing threshold %s " % str(threshold)
			logging.debug("Testing threshold %s" % threshold)
			thres_request_df = self.processSectionsForThreshold(examples_dataframe, threshold, examples_target=examples_target, best_aggregator_func=best_aggregator_func, time_since=time_since, trips_since=trips_since, sign_with_label=True, column_names=column_names)
			logging.debug("threshold df for threshold %s: %s" % (threshold, thres_request_df))
			thresholds_df = pd.concat([thresholds_df, thres_request_df], axis=1)
			logging.debug("New thresholds_df shape: %s" % str(thresholds_df.shape))
			for classifier_name in base_training_set_dict:
				self.setTrainingDataForClassifier(classifier_name, base_training_set_dict.copy()[classifier_name][0], base_training_set_dict.copy()[classifier_name][1])
				logging.debug("training data set for classifier %s" % classifier_name)

		if attach_target:
			thresholds_df['target'] = examples_target

		if sign_with_aggregator:
			new_column_names = list(thresholds_df.columns.values)
			for idx in range(len(new_column_names)):
				new_column_names[idx] = str(new_column_names[idx]) + "_" + str(self._BestAggregator)
			logging.debug("Sign with aggregator column names: %s" % new_column_names)
		
		if path_or_buf is not None:
			thresholds_df.to_csv(path_or_buf=path_or_buf)

		if timeit:
			end = clock()
			print "testThresholds took %s seconds" % (end-start)
			logging.debug("testThresholds took %s seconds" % (end-start))

		return thresholds_df

	def processSectionsForThreshold(self, examples_dataframe, threshold, examples_target=None, path_or_buf=None, best_aggregator_func=None, time_since=None, trips_since=None, sign_with_label=True, column_names = ["distance", "duration", "first_filter_mode", "sectionId", "avg_speed",
						  "speed_EV", "speed_variance", "max_speed", "max_accel", "isCommute",
						  "heading_change_rate", "stop_rate", "velocity_change rate",
						  "start_lat", "start_lng", "stop_lat", "stop_lng",
						  "start_hour", "end_hour", "close_to_bus_stop", "close_to_train_stop",
						  "close_to_airport", "dataframe_index"]):

		req_column_names = ['generated_prediction', 'stored_prediction', 'confidence', 'requested']
		thres_request_df = pd.DataFrame(index = examples_dataframe.index, columns = req_column_names)
		
		if best_aggregator_func is None:
			best_aggregator_func = self._Aggregators[self._BestAggregator]
		
		gpred = pd.Series()
		spred = pd.Series()
		conf  = pd.Series()
		req   = pd.Series()

		for example in examples_dataframe.itertuples():
			gp, sp, c, r = self.processSection(example, threshold, confirmed_mode=examples_target.loc[example.Index],  dtype='tuple', best_aggregator_func=best_aggregator_func, time_since=time_since, trips_since=trips_since, column_names=column_names )
			
			gpred = gpred.append(gp)
			spred = spred.append(sp)
			conf  = conf.append(c)
			req   = req.append(r)

		logging.debug("generated predictions: %s" % gpred)
		logging.debug("stored predictions: %s" % spred)
		logging.debug("confidences: %s" % conf)
		logging.debug("request: %s" % req)

		thres_request_df['generated_prediction'] = gpred
		thres_request_df['stored_prediction'] 	 = spred
		thres_request_df['confidence']		 	 = conf
		thres_request_df['requested'] 			 = req

		if sign_with_label:
			new_column_names = ['generated_prediction_threshold_%s' % threshold,'stored_prediction_threshold_%s' % threshold, 'confidence_threshold_%s' % threshold, 'requested_threshold_%s' % threshold]
			thres_request_df.columns = new_column_names
			logging.debug("new column names: %s" % new_column_names)

		return thres_request_df


	def processSection(self, section, threshold, dtype='dataframe', confirmed_mode=None, best_aggregator_func=None, time_since=None, trips_since=None, column_names=["distance", "duration", "first_filter_mode", "sectionId", "avg_speed",
						  "speed_EV", "speed_variance", "max_speed", "max_accel", "isCommute",
						  "heading_change_rate", "stop_rate", "velocity_change rate",
						  "start_lat", "start_lng", "stop_lat", "stop_lng",
						  "start_hour", "end_hour", "close_to_bus_stop", "close_to_train_stop",
						  "close_to_airport", "dataframe_index"]):

		if best_aggregator_func is None:
			best_aggregator_func = self._Aggregators[self._BestAggregator]

		if dtype.lower() == 'dataframe':
			lo
			featureMatrix, resultVector = generateFeatureMatrixAndResultVectorStep(section, featureLabels=column_names)
			feature_df_row 				= pd.DataFrame(featureMatrix, index=[section.index], columns=column_names)

		elif dtype.lower() == 'tuple':
			featureMatrix 			    = generateFeatureMatrixStepForTuple(section, featureLabels=column_names)
			
			feature_df_row 				= pd.DataFrame(featureMatrix, index=[section.Index], columns=column_names)
		else:
			raise ValueError("dtype for processSection must be 'dataframe' or 'tuple'. %s was recieved" % dtype)

		predictions_prob_df 	= self.predictProb(feature_df_row)
		agg_prob_df 			= best_aggregator_func(feature_df_row, predictions_prob_df=predictions_prob_df)
		guess_confidence 		= agg_prob_df.max(axis=1)
		generated_prediction 	= agg_prob_df.idxmax(axis=1)
		stored_prediction 		= generated_prediction
		request 				= self.decideRequestForThreshold(guess_confidence.iloc[0], threshold)

		if request:

			if confirmed_mode is not None:
				logging.debug("confirmed mode not None")
				stored_prediction = pd.Series(data=confirmed_mode, index=generated_prediction.index, dtype=generated_prediction.dtype)
			else:
				try:
					if dtype.lower() == 'dataframe':
						stored_prediction = pd.Series(data=section['confirmed_mode'], index=generated_prediction.index, dtype=generated_prediction.dtype)
					else:
						stored_prediction = pd.Series(data=section.confirmed_mode, index=generated_prediction.index, dtype=generated_prediction.dtype)

				except KeyError:
					logging.debug("For simulating request, a confirmed mode is needed. Couldn't find confirmed_mode in %s" % section)
					stored_prediction = pd.Series(data=None, index=generated_prediction.index, dtype=generated_prediction.dtype)

			if stored_prediction is not None:	
				to_train = []
				for classifier_name in self._Classifiers:
					if self._Classifiers[classifier_name].update_with_personal_data:
						self.addToClassifierTrainingData(feature_df_row, stored_prediction, classifier_name)
						to_train.append(classifier_name)
				logging.debug("Classifiers to train: %s" % to_train)
				self.trainEnsemble(classifier_names=to_train)

		return generated_prediction, stored_prediction, guess_confidence, pd.Series(data=request, index=generated_prediction.index)

	def decideRequestForThreshold(self, guess, threshold, time_since = None, trips_since = None):

		return int(guess < threshold)

	def addHighestGuessColumn(self, aggregated_prob_df):
		aggregated_prob_df['highest_confidence'] = aggregated_prob_df.max(axis=1)

	def generateDecideRequestDataframe(self, threshold_list, prediction_examples, aggregated_prob_df=None, predictions_prob_df=None, time_since=False, trips_since=False):
		if aggregated_prob_df is None:

			if predictions_prob_df is None:
				predictions_prob_df = self.predictProb(prediction_examples)
				agg_func = self._Aggregators[self._BestAggregator]
				aggregated_prob_df = agg_func(prediction_examples, predictions_prob_df=predictions_prob_df)

			else:
				agg_func = self._Aggregators[self._BestAggregator]
				aggregated_prob_df = agg_func(prediction_examples, predictions_prob_df=predictions_prob_df)

		request_df = aggregated_prob_df.copy()
		self.addHighestGuessColumn(request_df)


		for threshold in threshold_list:
			request_df = self.generateDecideRequestDataframeForThreshold(request_df, threshold, time_since=time_since, trips_since=trips_since)

		return request_df

	def generateDecideRequestDataframeForThreshold(self, aggregated_prob_df_hg, threshold, time_since=False, trips_since=False):

		column_name = "threshold_" + str(threshold)
		aggregated_prob_df_hg[column_name] = (aggregated_prob_df_hg['highest_confidence'] < threshold)

		return aggregated_prob_df_hg
		
	def calculateRequestFrequency(self, request_df):

		frequecy_df = pd.DataFrame(index=[0])
		total_sections = float(request_df.shape[0])
		threshold_columns = []

		for column_name in list(request_df.columns.values):
			if 'requested_threshold' in str(column_name):
				threshold_columns.append(column_name)

		for column_name in threshold_columns:
			new_column_name = column_name + "_frequency"
			frequecy_df[new_column_name] = np.around((np.sum(request_df[column_name])/total_sections), decimals=3)

		return frequecy_df


#######

#Adjusted functions from the old mode file. Since its going to be deprecated anyway, there's no point in changing them in the actual file itself
#	we might as well just copy, paste, and adjust them here

def generateBusAndTrainStopStep():
	bus_cluster=mode_cluster(5,105,1)
	train_cluster=mode_cluster(6,600,1)
	air_cluster=mode_cluster(9,600,1)
	return (bus_cluster, train_cluster)

def generateFeatureMatrixAndResultVectorStep(examples_dataframe, dtype="dataframe", featureLabels=["distance", "duration", "first filter mode", "sectionId", "avg speed",
						  "speed EV", "speed variance", "max speed", "max accel", "isCommute",
						  "heading change rate", "stop rate", "velocity change rate",
						  "start lat", "start lng", "stop lat", "stop lng",
						  "start hour", "end hour", "close to bus stop", "close to train stop",
						  "close to airport", "dataframe index"], bus_cluster=None, train_cluster=None):

	featureMatrix = np.zeros([examples_dataframe.shape[0], len(featureLabels)])
	resultVector = np.zeros(examples_dataframe.shape[0])
	if bus_cluster is None or train_cluster is None:
		bus_cluster, train_cluster = generateBusAndTrainStopStep()
	i = 0
	for section in examples_dataframe.itertuples():
		#iterrows is really slow. itertuples is faster, but it requires all feature labels be "proper," i.e. contain no spaces
		if i % 100 == 0:
			logging.debug("Processing record %s " % i)
		try: 
			resultVector[i] = section.confirmed_mode

		except Exception, e:
			if isinstance(e, AttributeError):
				warn("confirmed_mode not found in section %s" % section.Index)
				logging.debug("confirmed_mode not found in section %s" % section.Index)
			else:
				print "result vector not set due to error %s" % e
				logging.debug("interpreted confirmed mode as %s" % section.confirmed_mode)
		try:
			updateFeatureMatrixRowWithSection(featureMatrix, i, section, Index=section.Index, bus_cluster=bus_cluster, train_cluster=train_cluster)			
		except Exception, e:
			logging.debug("Couldn't process section %s due to error %s" % (section, e))
		i += 1
	return (featureMatrix, resultVector)

def generateFeatureMatrixStepForTuple(section_tuple, featureLabels=["distance", "duration", "first filter mode", "sectionId", "avg speed",
						  "speed EV", "speed variance", "max speed", "max accel", "isCommute",
						  "heading change rate", "stop rate", "velocity change rate",
						  "start lat", "start lng", "stop lat", "stop lng",
						  "start hour", "end hour", "close to bus stop", "close to train stop",
						  "close to airport", "dataframe index"], bus_cluster=None, train_cluster = None):
	
	featureMatrix = np.zeros([1, len(featureLabels)])
	resultVector = np.zeros(1)
	if bus_cluster is None or train_cluster is None:
		bus_cluster, train_cluster = generateBusAndTrainStopStep()
	try:
		updateFeatureMatrixRowWithSection(featureMatrix, 0, section_tuple, Index=section_tuple.Index, bus_cluster=bus_cluster, train_cluster=train_cluster)
	except Exception, e:
		logging.debug("(Tuple) Couldn't process section %s due to error %s" % (section_tuple, e))
	
	# if confirmed_mode is not None:
	# 	try:
	# 		resultVector[0] = confirmed_mode
	# 	except Exception, e:
	# 		logging.debug("(Tuple) Couldn't set result vector for section %s due to error %s" % (section_tuple, e))
	# 		logging.debug("(Tuple) Interpreted confirmed mode as %s" % section_tuple.confirmed_mode)

	# else:
	# 	try:
	# 		resultVector[0] = section_tuple.confirmed_mode
	# 	except Exception, e:
	# 		logging.debug("(Tuple) Couldn't set result vector for section %s due to error %s" % (section_tuple, e))
	# 		if not isinstance(e, AttributeError):
	# 			logging.debug("(Tuple) Interpreted confirmed mode as %s" % section_tuple.confirmed_mode)
	# 		return featureMatrix, resultVector

	return featureMatrix

def updateFeatureMatrixRowWithSection(featureMatrix, i, section, Index=0, bus_cluster=None, train_cluster=None, air_cluster=None):
	featureMatrix[i, 0] = section.distance
	featureMatrix[i, 1] = (section.section_end_datetime - section.section_start_datetime).total_seconds()

	# Deal with unknown modes like "airplane"
	try:
	  featureMatrix[i, 2] = section.mode
	except ValueError:
	  featureMatrix[i, 2] = 0

	featureMatrix[i, 3] = section.section_id
	featureMatrix[i, 4] = calAvgSpeed(section)
	speeds = calSpeeds(section)
	if speeds is not None and len(speeds) > 0:
		featureMatrix[i, 5] = np.mean(speeds)
		featureMatrix[i, 6] = np.std(speeds)
		featureMatrix[i, 7] = np.max(speeds)
	else:
		# They will remain zero
		pass
	accels = calAccels(section)
	if accels is not None and len(accels) > 0:
		featureMatrix[i, 8] = np.max(accels)
	else:
		# They will remain zero
		pass
	featureMatrix[i, 9] = ('commute' in section) and (section.commute == 'to' or section.commute == 'from')
	featureMatrix[i, 10] = calHCR(section)
	featureMatrix[i, 11] = calSR(section)
	featureMatrix[i, 12] = calVCR(section)
	if 'section_start_point' in section and section.section_start_point != None:
		startCoords = section.section_start_point['coordinates']
		featureMatrix[i, 13] = startCoords[0]
		featureMatrix[i, 14] = startCoords[1]
	
	if 'section_end_point' in section and section.section_end_point != None:
		endCoords = section.section_end_point['coordinates']
		featureMatrix[i, 15] = endCoords[0]
		featureMatrix[i, 16] = endCoords[1]
	
	featureMatrix[i, 17] = section.section_start_datetime.time().hour
	featureMatrix[i, 18] = section.section_end_datetime.time().hour
   
	if bus_cluster is not None: 
	 	featureMatrix[i, 19] = mode_start_end_coverage(section, bus_cluster,105)
	if train_cluster is not None:
	 	featureMatrix[i, 20] = mode_start_end_coverage(section, train_cluster,600)
	if air_cluster is not None:
	 	featureMatrix[i, 21] = mode_start_end_coverage(section, air_cluster,600)

	featureMatrix[i, 22] = Index
	# Replace NaN and inf by zeros so that it doesn't crash later
	featureMatrix[i] = np.nan_to_num(featureMatrix[i])

from sklearn.cluster import DBSCAN

# Our imports
from emission.core.get_database import get_section_db, get_mode_db, get_routeCluster_db,get_transit_db
from emission.core.common import calDistance, Include_place_2
from emission.analysis.modelling.tour_model.trajectory_matching.route_matching import getRoute,fullMatchDistance,matchTransitRoutes,matchTransitStops
import utm
import math
import logging

Sections = get_section_db()
Modes = get_mode_db()


# The speed is in m/s
def calSpeed(trackpoint1, trackpoint2):
  from dateutil import parser
  distanceDelta = calDistance(trackpoint1['track_location']['coordinates'],
                              trackpoint2['track_location']['coordinates'])
  timeDelta = parser.parse(trackpoint2['time']) - parser.parse(trackpoint1['time'])
  #               (trackpoint1, trackpoint2, distanceDelta, timeDelta))
  if timeDelta.total_seconds() != 0:
    return distanceDelta / timeDelta.total_seconds()
  else:
    return None

# This formula is from:
# http://www.movable-type.co.uk/scripts/latlong.html
# It returns the heading between two points using 
def calHeading(point1, point2):
    # points are in GeoJSON format, ie (lng, lat)
    phi1 = math.radians(point1[1])
    phi2 = math.radians(point2[1])
    lambda1 = math.radians(point1[0])
    lambda2 = math.radians(point2[0])

    y = math.sin(lambda2-lambda1) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - \
        math.sin(phi1)*math.cos(phi2)*math.cos(lambda2-lambda1)
    brng = math.degrees(math.atan2(y, x))
    return brng

def calHC(point1, point2, point3):
	HC = calHeading(point2, point3) - calHeading(point1, point2)
	return HC

def calHCR(segment):
    trackpoints = segment.track_points
    if len(trackpoints) < 3:
		return 0
    else:
        HCNum = 0
        for (i, point) in enumerate(trackpoints[:-2]):
            currPoint = point
            nextPoint = trackpoints[i+1]
            nexNextPt = trackpoints[i+2]
            HC = calHC(currPoint['track_location']['coordinates'], nextPoint['track_location']['coordinates'], \
                       nexNextPt['track_location']['coordinates'])
            if HC >= 15:
                HCNum += 1
        segmentDist = segment.distance
        if segmentDist!= None and segmentDist != 0:
            HCR = HCNum/segmentDist
            return HCR
        else:
            return 0


def calSR(segment):
    trackpoints = segment.track_points
    if len(trackpoints) < 2:
		return 0
    else:
        stopNum = 0
        for (i, point) in enumerate(trackpoints[:-1]):
            currPoint = point
            nextPoint = trackpoints[i+1]

            currVelocity = calSpeed(currPoint, nextPoint)
            if currVelocity != None and currVelocity <= 0.75:
                stopNum += 1

        segmentDist = segment.distance
        if segmentDist != None and segmentDist != 0:
            return stopNum/segmentDist
        else:
            return 0

def calVCR(segment):
    trackpoints = segment.track_points
    if len(trackpoints) < 3:
		return 0
    else:
        Pv = 0
        for (i, point) in enumerate(trackpoints[:-2]):
            currPoint = point
            nextPoint = trackpoints[i+1]
            nexNextPt = trackpoints[i+2]
            velocity1 = calSpeed(currPoint, nextPoint)
            velocity2 = calSpeed(nextPoint, nexNextPt)
            if velocity1 != None and velocity2 != None:
                if velocity1 != 0:
                    VC = abs(velocity2 - velocity1)/velocity1
                else:
                    VC = 0
            else:
                VC = 0

            if VC > 0.7:
                Pv += 1

        segmentDist = segment.distance
        if segmentDist != None and segmentDist != 0:
            return Pv/segmentDist
        else:
            return 0

def calSegmentDistance(segment):
  return segment.distance

def calSpeeds(segment):
  trackpoints = segment.track_points
  if len(trackpoints) == 0:
    return None
  return calSpeedsForList(trackpoints)

def calSpeedsForList(trackpoints):
  speeds = np.zeros(len(trackpoints) - 1)
  for (i, point) in enumerate(trackpoints[:-1]):
    currPoint = point
    nextPoint = trackpoints[i+1]
    currSpeed = calSpeed(currPoint, nextPoint)
    if currSpeed != None:
      speeds[i] = currSpeed
  return speeds

def calAvgSpeed(segment):
  timeDelta = segment.section_end_datetime - segment.section_start_datetime
  if timeDelta.total_seconds() != 0:
    return segment.distance / timeDelta.total_seconds()
  else:
    return None

# In order to calculate the acceleration, we do the following.
# point0: (loc0, t0), point1: (loc1, t1), point2: (loc2, t2), point3: (loc3, t3)
# becomes
# speed0: ((loc1 - loc0) / (t1 - t0)), speed1: ((loc2 - loc1) / (t2-t1)),
# speed2: ((loc3 - loc2) / (t3 - t2)
# becomes
# segment0: speed0 / (t1 - t0), segment1: (speed1 - speed0)/(t2-t1),
# segment2: (speed2 - speed1) / (t3-t2)

def calAccels(segment):
  from dateutil import parser

  speeds = calSpeeds(segment)
  trackpoints = segment.track_points

  if speeds is None or len(speeds) == 0:
    return None

  accel = np.zeros(len(speeds) - 1)
  prevSpeed = 0
  for (i, speed) in enumerate(speeds[0:-1]):
    currSpeed = speed # speed0
    speedDelta = currSpeed - prevSpeed # (speed0 - 0)
    # t1 - t0
    timeDelta = parser.parse(trackpoints[i+1]['time']) - parser.parse(trackpoints[i]['time'])
    #   (trackpoints[i+1], trackpoints[i], speedDelta, timeDelta))
    if timeDelta.total_seconds() != 0:
      accel[i] = speedDelta/(timeDelta.total_seconds())
    prevSpeed = currSpeed
  return accel

def getIthMaxSpeed(segment, i):
  # python does not appear to have a built-in mechanism for returning the top
  # ith max. We would need to write our own, possibly by sorting. Since it is
  # not clear whether we ever actually need this (the paper does not explain
  # which i they used), we just return the max.
  assert(i == 1)
  speeds = calSpeeds(segment)
  return np.amax(speeds)

def getIthMaxAccel(segment, i):
  # python does not appear to have a built-in mechanism for returning the top
  # ith max. We would need to write our own, possibly by sorting. Since it is
  # not clear whether we ever actually need this (the paper does not explain
  # which i they used), we just return the max.
  assert(i == 1)
  accels = calAccels(segment)
  return np.amax(accels)

def calSpeedDistParams(speeds):
  return (np.mean(speeds), np.std(speeds))

# def user_tran_mat(user):
#     user_sections=[]
#     # print(tran_mat)
#     query = {"$and": [{'type': 'move'},{'user_id':user},\
#                       {'$or': [{'confirmed_mode':1}, {'confirmed_mode':3},\
#                                {'confirmed_mode':5},{'confirmed_mode':6},{'confirmed_mode':7}]}]}
#     # print(Sections.find(query).count())
#     for section in Sections.find(query).sort("section_start_datetime",1):
#         user_sections.append(section)
#     if Sections.find(query).count()>=2:
#         tran_mat=np.zeros([Modes.find().count(), Modes.find().count()])
#         for i in range(len(user_sections)-1):
#             if (user_sections[i+1]['section_start_datetime']-user_sections[i]['section_end_datetime']).seconds<=60:
#                 # print(user_sections[i+1]['section_start_datetime'],user_sections[i]['section_end_datetime'])
#                 fore_mode=user_sections[i]["confirmed_mode"]
#                 after_mode=user_sections[i+1]["confirmed_mode"]
#                 tran_mat[fore_mode-1,after_mode-1]+=1
#         row_sums = tran_mat.sum(axis=1)
#         new_mat = tran_mat / row_sums[:, np.newaxis]
#         return new_mat
#     else:
#         return None
#
# # all model
# def all_tran_mat():
#     tran_mat=np.zeros([Modes.find().count(), Modes.find().count()])
#     for user in Sections.distinct("user_id"):
#         user_sections=[]
#         # print(tran_mat)
#         query = {"$and": [{'type': 'move'},{'user_id':user},\
#                           {'$or': [{'confirmed_mode':1}, {'confirmed_mode':3},\
#                                    {'confirmed_mode':5},{'confirmed_mode':6},{'confirmed_mode':7}]}]}
#         # print(Sections.find(query).count())
#         for section in Sections.find(query).sort("section_start_datetime",1):
#             user_sections.append(section)
#         if Sections.find(query).count()>=2:
#             for i in range(len(user_sections)-1):
#                 if (user_sections[i+1]['section_start_datetime']-user_sections[i]['section_end_datetime']).seconds<=60:
#                     # print(user_sections[i+1]['section_start_datetime'],user_sections[i]['section_end_datetime'])
#                     fore_mode=user_sections[i]["confirmed_mode"]
#                     after_mode=user_sections[i+1]["confirmed_mode"]
#                     tran_mat[fore_mode-1,after_mode-1]+=1
#     row_sums = tran_mat.sum(axis=1)
#     new_mat = tran_mat / row_sums[:, np.newaxis]
#     return new_mat

def mode_cluster(mode,eps,sam):
    mode_change_pnts=[]
    # print(tran_mat)
    query = {"$and": [{'type': 'move'},\
                      {'confirmed_mode':mode}]}
    # print(Sections.find(query).count())
    for section in Sections.find(query).sort("section_start_datetime",1):
        try:
            mode_change_pnts.append(section['section_start_point']['coordinates'])
            mode_change_pnts.append(section['section_end_point']['coordinates'])
        except:
            pass
    # print(user_change_pnts)
    # print(len(mode_change_pnts))
    if len(mode_change_pnts) == 0:
      return np.zeros(0)

    if len(mode_change_pnts)>=1:
        # print(mode_change_pnts)
        np_points=np.array(mode_change_pnts)
        # print(np_points[:,0])
        # fig, axes = plt.subplots(1, 1)
        # axes.scatter(np_points[:,0], np_points[:,1])
        # plt.show()
    else:
        pass
    utm_x = []
    utm_y = []
    for row in mode_change_pnts:
        # GEOJSON order is lng, lat
        utm_loc = utm.from_latlon(row[1],row[0])
        utm_x = np.append(utm_x,utm_loc[0])
        utm_y = np.append(utm_y,utm_loc[1])
    utm_location = np.column_stack((utm_x,utm_y))
    db = DBSCAN(eps=eps,min_samples=sam)
    db_fit = db.fit(utm_location)
    db_labels = db_fit.labels_
    #print db_labels
    new_db_labels = db_labels[db_labels!=-1]
    new_location = np_points[db_labels!=-1]
    # print len(new_db_labels)
    # print len(new_location)
    # print new_information

    label_unique = np.unique(new_db_labels)
    cluster_center = np.zeros((len(label_unique),2))
    for label in label_unique:
        sub_location = new_location[new_db_labels==label]
        temp_center = np.mean(sub_location,axis=0)
        cluster_center[int(label)] = temp_center
    # print cluster_center
    return cluster_center

#
# print(mode_cluster(6))

def mode_start_end_coverage(segment,cluster,eps):
    mode_change_pnts=[]
    # print(tran_mat)
    num_sec=0
    centers=cluster
    # print(centers)
    try:
        if Include_place_2(centers,segment['section_start_point']['coordinates'],eps) and \
                    Include_place_2(centers,segment['section_end_point']['coordinates'],eps):
            return 1
        else:
            return 0
    except:
            return 0
# print(mode_start_end_coverage(5,105,2))
# print(mode_start_end_coverage(6,600,2))

# This is currently only used in this file, so it is fine to use only really
# user confirmed modes. We don't want to learn on trips where we don't have
# ground truth.
def get_mode_share_by_count(lst):
    # input here is a list of sections
    displayModeList = getDisplayModes()
    modeCountMap = {}
    for mode in displayModeList:
        modeCountMap[mode['mode_name']] = 0
        for section in lst:
            if section['confirmed_mode']==mode['mode_id']:
                modeCountMap[mode['mode_name']] +=1
            elif section['mode']==mode['mode_id']:
                modeCountMap[mode['mode_name']] +=1
    return modeCountMap

# This is currently only used in this file, so it is fine to use only really
# user confirmed modes. We don't want to learn on trips where we don't have
# ground truth.
def get_mode_share_by_count(list_idx):
    Sections=get_section_db()
    ## takes a list of idx's
    AllModeList = getAllModes()

    MODE = {}
    MODE2= {}
    for mode in AllModeList:
        MODE[mode['mode_id']]=0
    for _id in list_idx:
        section=Sections.find_one({'_id': _id})
        mode_id = section['confirmed_mode']
        try:
            MODE[mode_id] += 1
        except KeyError:
            MODE[mode_id] = 1
    # print(sum(MODE.values()))
    if sum(MODE.values())==0:
        for mode in AllModeList:
            MODE2[mode['mode_id']]=0
        # print(MODE2)
    else:
        for mode in AllModeList:
            MODE2[mode['mode_id']]=MODE[mode['mode_id']]/sum(MODE.values())
    return MODE2

def cluster_route_match_score(segment,step1=100000,step2=100000,method='lcs',radius1=2000,threshold=0.5):
    userRouteClusters=get_routeCluster_db().find_one({'$and':[{'user':segment['user_id']},{'method':method}]})['clusters']
    route_seg = getRoute(segment['_id'])

    dis=999999
    medoid_ids=userRouteClusters.keys()
    if len(medoid_ids)!=0:
        choice=medoid_ids[0]
        for idx in userRouteClusters.keys():
            route_idx=getRoute(idx)
            try:
                dis_new=fullMatchDistance(route_seg,route_idx,step1,step2,method,radius1)
            except RuntimeError:

                dis_new=999999
            if dis_new<dis:
                dis=dis_new
                choice=idx
    # print(dis)
    # print(userRouteClusters[choice])
    if dis<=threshold:
        cluster=userRouteClusters[choice]
        cluster.append(choice)
        ModePerc=get_mode_share_by_count(cluster)
    else:
        ModePerc=get_mode_share_by_count([])

    return ModePerc

def transit_route_match_score(segment,step1=100000,step2=100000,method='lcs',radius1=2500,threshold=0.5):
    Transits=get_transit_db()
    transitMatch={}
    route_seg=getRoute(segment['_id'])
    for type in Transits.distinct('type'):
        for entry in Transits.find({'type':type}):
            transitMatch[type]=matchTransitRoutes(route_seg,entry['stops'],step1,step2,method,radius1,threshold)
            if transitMatch[entry['type']]==1:
                break
    return transitMatch

def transit_stop_match_score(segment,radius1=300):
    Transits=get_transit_db()
    transitMatch={}
    route_seg=getRoute(segment['_id'])
    for type in Transits.distinct('type'):
        for entry in Transits.find({'type':type}):
            transitMatch[type]=matchTransitStops(route_seg,entry['stops'],radius1)
            if transitMatch[entry['type']]==1:
                break
    return transitMatch




























