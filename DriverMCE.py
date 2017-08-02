#Jenny Steffens
#Driver for running the ModeClassifierEnsemble tests
#Warning, this code will probably take a while to run.

import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from ModeClassifierEnsemble import *
import warnings
import emission.core.get_database as edb
import sys
import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.decorations.analysis_timeseries_queries as esda
import emission.core.wrapper.entry as ecwe
import emission.storage.decorations.trip_queries as esdt
import emission.tests.common as etc
from emission.core.get_database import get_db, get_mode_db, get_section_db
from pymongo import MongoClient
from sklearn.tree import DecisionTreeClassifier
from time import clock
from uuid import UUID
# import emission.analysis.classification.inference.mode.oldMode as om

##########
THRESHOLD_TEST_CSV			= "threshold.csv"  #This driver outputs two csvs with the results. These are their filepaths
CONFIDENCE_TEST_CSV 		= "confidence.csv"
INTELLIGENT_TRAIN_DATA_SIZE = 0.1
MY_DATA_TEST_SIZE			= 0.25
TARGET 						= 'confirmed_mode'
# MY_USER_ID 				   	= UUID('b0d937d0-70ef-305e-9563-440369012b39') #So in this instance, we are Tom.
##########																We should be whoever has the most data

warnings.simplefilter("ignore")
# tom_uuid    = UUID('b0d937d0-70ef-305e-9563-440369012b39')  
# yin_uuid    = UUID('e211dd91-423f-31ff-a1f8-89e5fdecc164')
# culler_uuid = UUID('6a488797-5be1-38aa-9700-5532be978ff9') 

ModesColl = get_mode_db()
SectionsColl = get_section_db()

backupSections = MongoClient('localhost').Backup_database.Stage_Sections


start = clock()

SECTION_DATA 				= pd.DataFrame(list(SectionsColl.find({'$and' : [{ 'confirmed_mode' : {'$exists' : True }}, {'confirmed_mode' : {'$ne' : ''}}]})))
BACKUP_DATA 			 	= pd.DataFrame(list(backupSections.find({'$and' : [{ 'confirmed_mode' : {'$exists' : True }}, {'confirmed_mode' : {'$ne' : ''}}]})))

ALL_DATA = BACKUP_DATA.append(SECTION_DATA)


# We are only using confirmed data for this test. Even the threshold test
#	because we would need to simulate "prompting "
all_target_values 		= ALL_DATA[TARGET].unique()
all_user_uuids 			= ALL_DATA['user_id'].unique()

if len(all_user_uuids) <= 100:
	user_uuids_to_test = all_user_uuids
else:
	vc = ALL_DATA['user_id'].value_counts()
	vc_list = list(vc.index)
	user_uuids_to_test = vc_list[:100]

num_uuids_to_test = len(user_uuids_to_test)
all_threshold_df = pd.DataFrame()
all_confidence_df = pd.DataFrame()

trial_start = clock()
current_uuid_number = 0

for MY_USER_ID in user_uuids_to_test:

	current_uuid_number += 1
	print "processing uuid number %s out of %s" % (current_uuid_number, num_uuids_to_test)
	
	
	EVERYONE_BUT_ME_DATA 		  = ALL_DATA[ALL_DATA['user_id'] != MY_USER_ID]
	everyone_but_me_target_values = EVERYONE_BUT_ME_DATA[TARGET].unique()

	for value in all_target_values:
		if value not in everyone_but_me_target_values:
			to_sample 				= ALL_DATA[ALL_DATA[TARGET] == value]
			new_row 				= to_sample.sample(n=1)
			EVERYONE_BUT_ME_DATA	= EVERYONE_BUT_ME_DATA.append(new_row)

	MY_DATA 					  = ALL_DATA[ALL_DATA['user_id'] == MY_USER_ID].sort_values(['section_end_datetime'], ascending=True).reset_index(drop=True) 
	#We have to sort only MY_DATA here because we are using it to test as well as train.

	MY_DATA_TRAIN				  = MY_DATA.iloc[:int((MY_DATA.shape[0])*MY_DATA_TEST_SIZE*-1)]
	my_data_train_target_values   = MY_DATA_TRAIN[TARGET].unique()

	for value in all_target_values:
		if value not in my_data_train_target_values:
			to_sample 				= ALL_DATA[ALL_DATA[TARGET] == value]
			new_row 				= to_sample.sample(n=1)
			MY_DATA_TRAIN 			= MY_DATA_TRAIN.append(new_row)

	WEIGHTER_TRAIN 			= MY_DATA.iloc[int((MY_DATA.shape[0])*MY_DATA_TEST_SIZE*-1):int((MY_DATA.shape[0])*INTELLIGENT_TRAIN_DATA_SIZE*-1)]

	MY_DATA_TEST 			= MY_DATA.iloc[int((MY_DATA.shape[0])*INTELLIGENT_TRAIN_DATA_SIZE*-1):]
	MY_DATA_TEST_TARGET		= MY_DATA_TEST[TARGET]
	MY_DATA_TEST 			= MY_DATA_TEST.drop(TARGET, axis=1)

	EVERYONE_DATA 			= pd.concat([EVERYONE_BUT_ME_DATA, MY_DATA_TRAIN], axis=0)
	# We want to make sure there's no overlap between EVERYONE_DATA and MY_DATA_TEST, or else that's cheating on the hand of the
	#	everyone classifier


	fixed_predictors = ["distance", "duration", "first_filter_mode", "sectionId", "avg_speed",
							  "speed_EV", "speed_variance", "max_speed", "max_accel", "isCommute",
							  "heading_change_rate", "stop_rate", "velocity_change rate",
							  "start_lat", "start_lng", "stop_lat", "stop_lng",
							  "start_hour", "end_hour", "close_to_bus_stop", "close_to_train_stop",
							  "close_to_airport", "dataframe_index"]


	EVERYONE_MATRIX, EVERYONE_RV 				= generateFeatureMatrixAndResultVectorStep(EVERYONE_DATA)
	EVERYONE_DF 								= pd.DataFrame(EVERYONE_MATRIX, index=EVERYONE_DATA.index, columns=fixed_predictors)
	EVERYONE_TARGET								= pd.Series(data=EVERYONE_RV, index=EVERYONE_DATA.index)
	EVERYONE_DF[TARGET]							= EVERYONE_RV

	EVERYONE_BUT_ME_MATRIX, EVERYONE_BUT_ME_RV 	= generateFeatureMatrixAndResultVectorStep(EVERYONE_BUT_ME_DATA)
	EVERYONE_BUT_ME_DF							= pd.DataFrame(EVERYONE_BUT_ME_MATRIX, index=EVERYONE_BUT_ME_DATA.index, columns=fixed_predictors)
	EVERYONE_BUT_ME_TARGET 					 	= pd.Series(data=EVERYONE_BUT_ME_RV, index=EVERYONE_BUT_ME_DATA.index)
	EVERYONE_BUT_ME_DF[TARGET]					= EVERYONE_BUT_ME_RV

	MY_TRAIN_MATRIX, MY_TRAIN_RV				= generateFeatureMatrixAndResultVectorStep(MY_DATA_TRAIN)
	MY_TRAIN_DF 								= pd.DataFrame(MY_TRAIN_MATRIX, index=MY_DATA_TRAIN.index, columns=fixed_predictors)
	MY_TRAIN_TARGET	 							= pd.Series(data=MY_TRAIN_RV, index=MY_DATA_TRAIN.index)
	MY_TRAIN_DF[TARGET]							= MY_TRAIN_RV

	WEIGHTER_TRAIN_MATRIX, WEIGHTER_TRAIN_RV	= generateFeatureMatrixAndResultVectorStep(WEIGHTER_TRAIN)
	WEIGHTER_TRAIN_DF 							= pd.DataFrame(WEIGHTER_TRAIN_MATRIX, index=WEIGHTER_TRAIN.index, columns=fixed_predictors)
	WEIGHTER_TRAIN_DF[TARGET]					= WEIGHTER_TRAIN_RV


	MY_TEST_MATRIX, MY_TEST_RV 					= generateFeatureMatrixAndResultVectorStep(MY_DATA_TEST)
	MY_TEST_DF 									= pd.DataFrame(MY_TEST_MATRIX, index=MY_DATA_TEST.index, columns=fixed_predictors)

	rf 				= RandomForestClassifier()
	rf2 			= RandomForestClassifier()
	rf3 			= RandomForestClassifier()

	everyone 		= ModeClassifier(rf)
	everyone_but_me = ModeClassifier(rf2, update_with_personal_data=False)  #The new data we get will be "our" data, so we wouldn't update this one
	just_me 		= ModeClassifier(rf3)

	MCE = ModeClassifierEnsemble()

	MCE.addClassifier(everyone, 		'Everyone')
	MCE.addClassifier(everyone_but_me,  'EveryoneButMe')
	MCE.addClassifier(just_me, 			'JustMe')

	MCE.setTrainingDataForClassifier('Everyone', 	  EVERYONE_DF[fixed_predictors], 		EVERYONE_DF[TARGET])
	MCE.setTrainingDataForClassifier('EveryoneButMe', EVERYONE_BUT_ME_DF[fixed_predictors], EVERYONE_BUT_ME_DF[TARGET])
	MCE.setTrainingDataForClassifier('JustMe', 		  MY_TRAIN_DF[fixed_predictors], 		MY_TRAIN_DF[TARGET])

	MCE.trainEnsemble()

	intelligentWeighter = RandomForestClassifier()

	MCE.trainIntelligentWeighter(intelligentWeighter, WEIGHTER_TRAIN_DF[fixed_predictors], WEIGHTER_TRAIN_DF[TARGET])
	MCE.generateNaiveWeights(WEIGHTER_TRAIN_DF[fixed_predictors], WEIGHTER_TRAIN_DF[TARGET])

	MCE.addAggregator(MCE.unweightedMeanProb, 			'unweighted')
	MCE.addAggregator(MCE.naiveWeightedMeanProb, 		'naive')
	MCE.addAggregator(MCE.intelligentWeightedMeanProb,  'intelligent')

	MCE.setBestAggregator('unweighted')

	threshold_df  = MCE.testThresholds(MY_DATA_TEST, [.3, .4, .5, .6, .7], examples_target=MY_DATA_TEST_TARGET,  attach_target=True, timeit=True)
	confidence_df = MCE.testConfidenceMeasures(MY_TEST_DF, MY_DATA_TEST_TARGET, attach_predictions_prob_df=True, attach_agg_df=True, timeit=True)

	user_series_thres = pd.Series(data=MY_USER_ID, index=threshold_df.index)
	user_series_conf  = pd.Series(data=MY_USER_ID, index=confidence_df.index)

	threshold_df['user_id']  = user_series_thres
	confidence_df['user_id'] = user_series_conf

	all_threshold_df = all_threshold_df.append(threshold_df)
	all_confidence_df = all_confidence_df.append(confidence_df)


print all_threshold_df
print all_confidence_df
all_threshold_df.to_csv(path_or_buf=THRESHOLD_TEST_CSV)
all_confidence_df.to_csv(path_or_buf=CONFIDENCE_TEST_CSV)

print "total time: %s" % (clock()-trial_start)
print "files:"
print CONFIDENCE_TEST_CSV
print THRESHOLD_TEST_CSV








