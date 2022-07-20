from surprise import SVD, SVDpp
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import KNNBasic, KNNWithMeans
from surprise import accuracy
from surprise.model_selection import train_test_split

import pandas as pd
import numpy as np


def convert_traintest_dataframe_forsurprise(training_dataframe, testing_dataframe):
    reader = Reader(rating_scale=(0, 5))
    trainset = Dataset.load_from_df(training_dataframe[['userId', 'tmdbId', 'rating']], reader)
    testset = Dataset.load_from_df(testing_dataframe[['userId', 'tmdbId', 'rating']], reader)
    trainset = trainset.construct_trainset(trainset.raw_ratings)
    testset = testset.construct_testset(testset.raw_ratings)
    return trainset, testset


def recommendation(algo, trainset, testset):
    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)

    # # Predictions on training set
    # train_predictions = algo.test(trainset)
    # train_rmse = accuracy.rmse(train_predictions)
    # train_mae = accuracy.mae(train_predictions)

    # Predictions on testing set
    test_predictions = algo.test(testset)
    test_rmse = accuracy.rmse(test_predictions)
    test_mae = accuracy.mae(test_predictions)

    return test_rmse, test_mae, test_predictions


file_path_train = '/Users/nguyenhieu/Documents/GitHub/Movie-Recommendation-System/0_data/processed/training_data.csv'
file_path_test = '/Users/nguyenhieu/Documents/GitHub/Movie-Recommendation-System/0_data/processed/testing_data.csv'

traindf = pd.read_csv(file_path_train)
testdf = pd.read_csv(file_path_test)
trainset, testset = convert_traintest_dataframe_forsurprise(traindf, testdf)

print("1")
sim_options = {'name': 'pearson_baseline',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBasic(sim_options=sim_options)
test_knn_rmse, test_knn_mae, test_knn_pred = recommendation(algo, trainset, testset)

print("2")
sim_options = {'name': 'pearson_baseline',
               'user_based': False  # compute  similarities between items
               }
algo = KNNWithMeans(sim_options=sim_options)
test_knnwm_rmse, test_knnwm_mae, test_knnwm_pred = recommendation(algo, trainset, testset)

print("3")
# SVD
algo = SVD()
test_svd_rmse, test_svd_mae, test_svd_pred = recommendation(algo, trainset, testset)

print("4")
# SVDpp
algo = SVDpp()
test_svdpp_rmse, test_svdpp_mae, test_svdpp_pred = recommendation(algo, trainset, testset)

print("5")
test_pred_df = pd.DataFrame(
    columns=['uid', 'iid', 'og_rating', 'svd_rating', 'knn_rating', 'svdpp_rating', 'slopeone_rating',
             'baseline_rating'])

test_svd_df = pd.DataFrame(
    columns=['uid', 'iid', 'og_rating', 'est_rating'])
test_svdpp_df = pd.DataFrame(
    columns=['uid', 'iid', 'og_rating', 'est_rating'])
test_knn_df = pd.DataFrame(
    columns=['uid', 'iid', 'og_rating', 'est_rating'])
test_knnwm_df = pd.DataFrame(
    columns=['uid', 'iid', 'og_rating', 'est_rating'])

num_test = len(test_svd_pred)

for i in range(num_test):
    
    svd = test_svd_pred[i]
    svdpp = test_svdpp_pred[i]
    knn = test_knn_pred[i]
    knnwm = test_knnwm_pred[i]

    df = pd.DataFrame([[svd.uid, svd.iid, svd.r_ui, svd.est, svdpp.est, knn.est, knnwm.est]],
                      columns=['uid', 'iid', 'og_rating', 'svd_rating', 'svdpp_rating', 'knn_rating', 'knnwm_rating'])

    df_svd = pd.DataFrame([[svd.uid, svd.iid, svd.r_ui, svd.est]],
                          columns=['uid', 'iid', 'og_rating', 'est_rating'])
    df_svdpp = pd.DataFrame([[svd.uid, svd.iid, svd.r_ui, svdpp.est]],
                            columns=['uid', 'iid', 'og_rating', 'est_rating'])
    df_knn = pd.DataFrame([[svd.uid, svd.iid, svd.r_ui, knn.est]],
                           columns=['uid', 'iid', 'og_rating', 'est_rating'])
    df_knnwm = pd.DataFrame([[svd.uid, svd.iid, svd.r_ui, slopeone.est]],
                            columns=['uid', 'iid', 'og_rating', 'est_rating'])

    test_pred_df = pd.concat([df, test_pred_df], ignore_index=True)
    test_svd_df = pd.concat([df_svd, test_svd_df], ignore_index=True)
    test_svdpp_df = pd.concat([df_svdpp, test_svdpp_df], ignore_index=True)
    test_knn_df = pd.concat([df_knn, test_knn_df], ignore_index=True)
    test_knnwm_df = pd.concat([df_knnwm, test_knnwm_df], ignore_index=True)

print("6")
test_pred_df.to_csv('test_prediction_HP.csv')
# test_svd_df.to_csv('test_predictions_svd.csv')
# test_svdpp_df.to_csv('test_predictions_svdpp.csv')
# test_knn_df.to_csv('test_predictions_knn.csv')
# test_knnwm_df.to_csv('test_predictions_knnwm.csv')