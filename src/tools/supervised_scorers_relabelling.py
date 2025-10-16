from scipy.spatial import distance
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler


def eval_hyp_supervised(calibration_faces_observed, target_face, calibration_eeg_observed, observation_faces_observed, observation_eeg_observed, face_hypothesis, scoring_function):
    import warnings
    warnings.filterwarnings("ignore")
    distances_hypothesis = distance.cdist([face_hypothesis], observation_faces_observed, metric='euclidean')[0]
    scores = scoring_function(calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis)
    return scores


def eval_all_hyp_supervised(calibration_faces_observed, target_face, calibration_eeg_observed, observation_faces_observed, observation_eeg_observed, face_hypothesis, scoring_function):
    from joblib import Parallel, delayed
    return Parallel(n_jobs=-1)(delayed(eval_hyp_supervised)(calibration_faces_observed, target_face, calibration_eeg_observed, observation_faces_observed, observation_eeg_observed, face_hypothesis[i, :], scoring_function) for i in range(face_hypothesis.shape[0]))
    # return [eval_hyp(faces_observed, eegs_observed, face_hypothesis[i, :], scoring_function) for i in range(face_hypothesis.shape[0])]


def get_relative_scores_supervised(estimator, calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis):
    
    labelled_distances = distance.cdist([target_face], calibration_faces_observed, metric='euclidean')[0]

    observation_eeg = np.concatenate([calibration_eeg_observed, observation_eeg_observed], axis=0)
    relabelled_distances = np.concatenate([labelled_distances, distances_hypothesis], axis=0)

    results = get_relative_scores(estimator, observation_eeg, relabelled_distances)

    return results


def get_relative_scores(estimator, eegs_observed, distances_hypothesis):
    
    n_splits = 10
    test_size = 0.1
    random_state = np.random.randint(1000)
        
    scaler = StandardScaler()
    # Silly way to apply scaler to a list and have output as a list. Silly but just to keep working with list here and not have to change anything downstream.
    scaled_distances_hypothesis = scaler.fit_transform(np.array(distances_hypothesis).reshape(-1, 1)).flatten().tolist()

    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    # scorings=['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
    scorings=['neg_root_mean_squared_error']

    # raw scores
    raw_scores = cross_validate(estimator, eegs_observed, scaled_distances_hypothesis, cv=ss, scoring=scorings, n_jobs=-1)
    
    # shuffled eeg to use as baseline
    rng = np.random.RandomState(random_state)
    shuffled_eeg = eegs_observed[rng.permutation(len(eegs_observed))]
    shuffled_scores = cross_validate(estimator, shuffled_eeg, scaled_distances_hypothesis, cv=ss, scoring=scorings, n_jobs=-1)

    # packing it together
    results = {}
    results['raw_scores'] = raw_scores
    results['shuffled_scores'] = shuffled_scores
    
    results['scores'] = {}
    results['mean_scores'] = {}
    results['std_scores'] = {}
    for scoring_method in scorings:
        s = 'test_{}'.format(scoring_method)
        results['scores'][s] = raw_scores[s] - shuffled_scores[s]
        results['mean_scores'][s] = np.mean(results['scores'][s])
        results['std_scores'][s] = np.std(results['scores'][s])
    
    # results['distances_hypothesis'] = distances_hypothesis
    # results['scaled_distances_hypothesis'] = scaled_distances_hypothesis

    return results

def DummyScoring_Mean(calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis):
    estimator = DummyRegressor(strategy="mean")
    return get_relative_scores_supervised(estimator, calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis)

def LinRegScoring(calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis):
    estimator = make_pipeline(StandardScaler(), LinearRegression())
    return get_relative_scores_supervised(estimator, calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis)

def RandomForestScoring(calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis):
    estimator = make_pipeline(StandardScaler(), RandomForestRegressor())
    return get_relative_scores_supervised(estimator, calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis)

def SVRScoring(calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis):
    estimator = make_pipeline(StandardScaler(), SVR())
    return get_relative_scores_supervised(estimator, calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis)

def BestSVRScoring(calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis):
    estimator = make_pipeline(StandardScaler(), SVR(C= 0.01, 
                                                    gamma='scale', 
                                                    kernel='linear'))
    return get_relative_scores_supervised(estimator, calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis)

def MLPScoring(calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis):
    estimator = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50, 25)))
    return get_relative_scores_supervised(estimator, calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis)

def BestMLPScoring(calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis):
    estimator = make_pipeline(StandardScaler(), MLPRegressor(activation='identity', 
                                                             alpha=0.1, 
                                                             hidden_layer_sizes=(100, 100), 
                                                             learning_rate='adaptive'))
    return get_relative_scores_supervised(estimator, calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis)

supervised_method_to_function_mapping = {}
supervised_method_to_function_mapping['DummyScoring_Mean'] = DummyScoring_Mean
supervised_method_to_function_mapping['LinearRegression'] = LinRegScoring
supervised_method_to_function_mapping['Shuffle_LinearRegression'] = LinRegScoring
supervised_method_to_function_mapping['RandomForest'] = RandomForestScoring
supervised_method_to_function_mapping['SVR'] = SVRScoring
supervised_method_to_function_mapping['BestSVR'] = BestSVRScoring
supervised_method_to_function_mapping['MLP'] = MLPScoring
supervised_method_to_function_mapping['BestMLP'] = BestMLPScoring
