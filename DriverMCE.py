#Jenny Steffens
#Driver for running the ModeClassifierEnsemble tests
#Warning, this code will probably take a while to run.

import logging
# import emission.analysis.classification.inference.mode.oldMode as om
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG, filename='newMCEoutput.log')

import numpy as np
import pandas as pd 
import traceback
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
serverName = 'localhost'
ModesColl = get_mode_db()

SectionsColl = get_section_db()
# etc.loadTable(serverName, "Stage_Modes", "emission/tests/data/modes.json")
# etc.loadTable(serverName, "Stage_Sections", "../culler_all_trips")
# etc.loadTable(serverName, "Stage_Sections", "../tom_all_trips")
# etc.loadTable(serverName, "Stage_Sections", "../yin_all_trips")

logger = logging.getLogger()
# Configure logger to write to a file...

def myexcepthook(*exc_info):
    text = "".join(traceback.format_exception(*exc_info))
    logging.error("Unhandled exception: %s", text)


sys.excepthook = myexcepthook


backupSections = MongoClient('localhost').Backup_database.Stage_Sections

start = clock()

SECTION_DATA 				= pd.DataFrame(list(SectionsColl.find({'$and' : [{ 'confirmed_mode' : {'$exists' : True }}, {'confirmed_mode' : {'$ne' : ''}}, {'confirmed_mode': {'$ne' : 'Please Specify:'}}]})))
BACKUP_DATA 			 	= pd.DataFrame(list(backupSections.find({'$and' : [{ 'confirmed_mode' : {'$exists' : True }}, {'confirmed_mode' : {'$ne' : ''}}, {'confirmed_mode': {'$ne' : 'Please Specify:'}}]})))

logging.debug("Size of section dataframe is %s" % str(SECTION_DATA.shape))
logging.debug("Size of backup dataframe is %s" % str(BACKUP_DATA.shape))

ALL_DATA = BACKUP_DATA.append(SECTION_DATA)
logging.debug("Confirmed_mode column is %s" % ALL_DATA['confirmed_mode'])

# ALL_DATA = ALL_DATA[ALL_DATA['confirmed_mode'] != 'Please Specify:']
# We are only using confirmed data for this test. Even the threshold test
#	because we would need to simulate "prompting "
all_target_values 		= list(ALL_DATA[TARGET].unique())
logging.debug("All target values found in dataset: %s" % all_target_values)
for idx in range(len(all_target_values)):
	try: 
		float(all_target_values[idx])
	except:
		all_target_values.pop(idx)

ALL_DATA = ALL_DATA[ALL_DATA['confirmed_mode'].isin(list(all_target_values))]
all_user_uuids 			= ALL_DATA['user_id'].unique()

logging.debug("all target values after cleaning: %s" % str(all_target_values))
logging.debug("all user uuids: %s" % str(all_user_uuids))
logging.debug("Confirmed_mode column is %s" % ALL_DATA['confirmed_mode'])


if len(all_user_uuids) <= 100:
	user_uuids_to_test = all_user_uuids
else:
	vc = ALL_DATA['user_id'].value_counts()
	vc_list = list(vc.index)
	user_uuids_to_test = vc_list[:100]

num_uuids_to_test = len(user_uuids_to_test)

logging.debug("testing %s uuids" % str(num_uuids_to_test))

all_threshold_df = pd.DataFrame()
all_confidence_df = pd.DataFrame()

trial_start = clock()
current_uuid_number = 0

for MY_USER_ID in user_uuids_to_test:

	current_uuid_number += 1
	print "processing uuid number %s out of %s" % (current_uuid_number, num_uuids_to_test)
	logging.debug("processing uuid %s out of %s" % (current_uuid_number, num_uuids_to_test))
	logging.debug("uuid: %s" % str(MY_USER_ID))

	try:

		EVERYONE_BUT_ME_DATA 		  = ALL_DATA[ALL_DATA['user_id'] != MY_USER_ID]
		everyone_but_me_target_values = EVERYONE_BUT_ME_DATA[TARGET].unique()

		logging.debug("Everyone but me data shape: %s" % str(EVERYONE_BUT_ME_DATA.shape))
		logging.debug("everyone_but_me_target_values: %s" % str(everyone_but_me_target_values))
		
		for value in all_target_values:
			if value not in everyone_but_me_target_values:
				to_sample 				= ALL_DATA[ALL_DATA[TARGET] == value]
				new_row 				= to_sample.sample(n=1)
				EVERYONE_BUT_ME_DATA	= EVERYONE_BUT_ME_DATA.append(new_row)
				logging.debug("added target value %s to EVERYONE_BUT_ME_DATA" % str(value))
			logging.debug("New unique values for EVERYONE_BUT_ME_DATA are %s" % str(list(EVERYONE_BUT_ME_DATA[TARGET].unique())))


		MY_DATA 					  = ALL_DATA[ALL_DATA['user_id'] == MY_USER_ID].sort_values(['section_end_datetime'], ascending=True).reset_index(drop=True) 
		#We have to sort only MY_DATA here because we are using it to test as well as train.

		MY_DATA_TRAIN				  = MY_DATA.iloc[:int((MY_DATA.shape[0])*MY_DATA_TEST_SIZE*-1)]

		if MY_DATA_TRAIN.shape[0] < 1:
			logging.debug("Not enough rows in user training set. Skipping user number %s, %s" % (current_uuid_number, MY_USER_ID))
			continue

		my_data_train_target_values   = MY_DATA_TRAIN[TARGET].unique()

		logging.debug("My data train shape: %s" % str(MY_DATA_TRAIN.shape))
		logging.debug("my_data_train_target_values: %s" % str(my_data_train_target_values))


		for value in all_target_values:
			if value not in my_data_train_target_values:
				to_sample 				= ALL_DATA[ALL_DATA[TARGET] == value]
				new_row 				= to_sample.sample(n=1)
				MY_DATA_TRAIN 			= MY_DATA_TRAIN.append(new_row)
				logging.debug("added target value %s to MY_DATA_TRAIN" % str(value) )
			logging.debug("New unique values for MY_DATA_TRAIN are %s" % str(list(MY_DATA_TRAIN[TARGET].unique())))

		WEIGHTER_TRAIN 			= MY_DATA.iloc[int((MY_DATA.shape[0])*MY_DATA_TEST_SIZE*-1):int((MY_DATA.shape[0])*INTELLIGENT_TRAIN_DATA_SIZE*-1)]

		if WEIGHTER_TRAIN.shape[0] < 1:
			logging.debug("Not enough rows in weigher training set. Skipping user number %s, %s" % (current_uuid_number, MY_USER_ID))
			continue

		MY_DATA_TEST 			= MY_DATA.iloc[int((MY_DATA.shape[0])*INTELLIGENT_TRAIN_DATA_SIZE*-1):]

		if MY_DATA_TEST.shape[0] < 1:
			logging.debug("Not enough rows in user testing set. Skipping user number %s, %s" % (current_uuid_number, MY_USER_ID))
			continue

		MY_DATA_TEST_TARGET		= MY_DATA_TEST[TARGET]
		MY_DATA_TEST 			= MY_DATA_TEST.drop(TARGET, axis=1)

		EVERYONE_DATA 			= pd.concat([EVERYONE_BUT_ME_DATA, MY_DATA_TRAIN], axis=0)
		# We want to make sure there's no overlap between EVERYONE_DATA and MY_DATA_TEST, or else that's cheating on the hand of the
		#	everyone classifier
		logging.debug("My data test shape: %s" % str(MY_DATA_TEST.shape))

		fixed_predictors = ["distance", "duration", "first_filter_mode", "sectionId", "avg_speed",
								  "speed_EV", "speed_variance", "max_speed", "max_accel", "isCommute",
								  "heading_change_rate", "stop_rate", "velocity_change rate",
								  "start_lat", "start_lng", "stop_lat", "stop_lng",
								  "start_hour", "end_hour", "close_to_bus_stop", "close_to_train_stop",
								  "close_to_airport", "dataframe_index"]

		logging.debug("Processing Everyone matrix")
		EVERYONE_MATRIX, EVERYONE_RV 				= generateFeatureMatrixAndResultVectorStep(EVERYONE_DATA)
		EVERYONE_DF 								= pd.DataFrame(EVERYONE_MATRIX, index=EVERYONE_DATA.index, columns=fixed_predictors)
		EVERYONE_TARGET								= pd.Series(data=EVERYONE_RV, index=EVERYONE_DATA.index)
		EVERYONE_DF[TARGET]							= EVERYONE_RV

		logging.debug("Processing Everyone but me matrix")
		EVERYONE_BUT_ME_MATRIX, EVERYONE_BUT_ME_RV 	= generateFeatureMatrixAndResultVectorStep(EVERYONE_BUT_ME_DATA)
		EVERYONE_BUT_ME_DF							= pd.DataFrame(EVERYONE_BUT_ME_MATRIX, index=EVERYONE_BUT_ME_DATA.index, columns=fixed_predictors)
		EVERYONE_BUT_ME_TARGET 					 	= pd.Series(data=EVERYONE_BUT_ME_RV, index=EVERYONE_BUT_ME_DATA.index)
		EVERYONE_BUT_ME_DF[TARGET]					= EVERYONE_BUT_ME_RV

		logging.debug("Processing my matrix")
		MY_TRAIN_MATRIX, MY_TRAIN_RV				= generateFeatureMatrixAndResultVectorStep(MY_DATA_TRAIN)
		MY_TRAIN_DF 								= pd.DataFrame(MY_TRAIN_MATRIX, index=MY_DATA_TRAIN.index, columns=fixed_predictors)
		MY_TRAIN_TARGET	 							= pd.Series(data=MY_TRAIN_RV, index=MY_DATA_TRAIN.index)
		MY_TRAIN_DF[TARGET]							= MY_TRAIN_RV

		logging.debug("Processing weighter matrix")
		WEIGHTER_TRAIN_MATRIX, WEIGHTER_TRAIN_RV	= generateFeatureMatrixAndResultVectorStep(WEIGHTER_TRAIN)
		WEIGHTER_TRAIN_DF 							= pd.DataFrame(WEIGHTER_TRAIN_MATRIX, index=WEIGHTER_TRAIN.index, columns=fixed_predictors)
		WEIGHTER_TRAIN_DF[TARGET]					= WEIGHTER_TRAIN_RV

		logging.debug("Processing weighter test matrix; Don't worry if we can't find these confirmed modes--We're not supposed to here")
		MY_TEST_MATRIX, MY_TEST_RV 					= generateFeatureMatrixAndResultVectorStep(MY_DATA_TEST)
		MY_TEST_DF 									= pd.DataFrame(MY_TEST_MATRIX, index=MY_DATA_TEST.index, columns=fixed_predictors)

		rf 				= RandomForestClassifier()
		rf2 			= RandomForestClassifier()
		rf3 			= RandomForestClassifier()

		logging.debug("Models created")

		everyone 		= ModeClassifier(rf)
		everyone_but_me = ModeClassifier(rf2, update_with_personal_data=False)  #The new data we get will be "our" data, so we wouldn't update this one
		just_me 		= ModeClassifier(rf3)

		logging.debug("Models converted")

		MCE = ModeClassifierEnsemble()

		MCE.addClassifier(everyone, 		'Everyone')
		MCE.addClassifier(everyone_but_me,  'EveryoneButMe')
		MCE.addClassifier(just_me, 			'JustMe')

		logging.debug("Models added")

		MCE.setTrainingDataForClassifier('Everyone', 	  EVERYONE_DF[fixed_predictors], 		EVERYONE_DF[TARGET])
		MCE.setTrainingDataForClassifier('EveryoneButMe', EVERYONE_BUT_ME_DF[fixed_predictors], EVERYONE_BUT_ME_DF[TARGET])
		MCE.setTrainingDataForClassifier('JustMe', 		  MY_TRAIN_DF[fixed_predictors], 		MY_TRAIN_DF[TARGET])

		logging.debug("Training data set")

		MCE.trainEnsemble()

		logging.debug("Ensemble trained")

		intelligentWeighter = RandomForestClassifier()

		MCE.trainIntelligentWeighter(intelligentWeighter, WEIGHTER_TRAIN_DF[fixed_predictors], WEIGHTER_TRAIN_DF[TARGET])
		
		logging.debug("Intelligent weighter trained")

		MCE.generateNaiveWeights(WEIGHTER_TRAIN_DF[fixed_predictors], WEIGHTER_TRAIN_DF[TARGET])

		logging.debug("Naive weights generated")

		MCE.addAggregator(MCE.unweightedMeanProb, 			'unweighted')
		MCE.addAggregator(MCE.naiveWeightedMeanProb, 		'naive')
		MCE.addAggregator(MCE.intelligentWeightedMeanProb,  'intelligent')

		logging.debug("Aggregators added")

		MCE.setBestAggregator('unweighted')

		logging.debug("Best aggregator set")

		threshold_df  = MCE.testThresholds(MY_DATA_TEST, [.4, .5, .6, .7], examples_target=MY_DATA_TEST_TARGET,  attach_target=True, timeit=True)
		logging.debug("Threshold_df: %s" % threshold_df)
		confidence_df = MCE.testConfidenceMeasures(MY_TEST_DF, MY_DATA_TEST_TARGET, attach_predictions_prob_df=True, attach_agg_df=True, timeit=True)
		logging.debug("Confidence_df: %s" % confidence_df)

		user_series_thres = pd.Series(data=MY_USER_ID, index=threshold_df.index)
		logging.debug("user series thres %s" % user_series_thres)
		user_series_conf  = pd.Series(data=MY_USER_ID, index=confidence_df.index)
		logging.debug("user series conf %s" % user_series_conf)

		threshold_df['user_id']  = user_series_thres
		logging.debug("Threshold_df: %s" % threshold_df)
		confidence_df['user_id'] = user_series_conf
		logging.debug("Confidence_df: %s" % confidence_df)


		all_threshold_df = all_threshold_df.append(threshold_df)
		logging.debug("appended threshold dataframe for user %s" % MY_USER_ID)
		all_confidence_df = all_confidence_df.append(confidence_df)
		logging.debug("appended confidence dataframe for user %s" % MY_USER_ID)

	except Exception, e:
		logging.debug("Skipping processing user number %s, %s due to error %s" (current_uuid_number, MY_USER_ID, e))
		continue

print all_threshold_df
print all_confidence_df

logging.debug("all threshold df is %s" % all_threshold_df)
logging.debug("all_confidence_df is %s" % all_confidence_df)

all_threshold_df.to_csv(path_or_buf=THRESHOLD_TEST_CSV)
logging.debug("all threshold df stored in %s" % THRESHOLD_TEST_CSV)
all_confidence_df.to_csv(path_or_buf=CONFIDENCE_TEST_CSV)
logging.debug("all confidence df stored in %s" % CONFIDENCE_TEST_CSV)


print "total time: %s" % (clock()-trial_start)
print "files:"
print CONFIDENCE_TEST_CSV
print THRESHOLD_TEST_CSV








