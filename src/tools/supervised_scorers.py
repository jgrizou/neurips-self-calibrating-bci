from scipy.spatial import distance
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error



def eval_hyp_supervised(calibration_faces_observed, target_face, calibration_eeg_observed, observation_faces_observed, observation_eeg_observed, face_hypothesis, scoring_function):
    import warnings
    warnings.filterwarnings("ignore")
    distances_hypothesis = [distance.euclidean(face, face_hypothesis) for face in observation_faces_observed]
    scores = scoring_function(calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis)
    return scores


def eval_all_hyp_supervised(calibration_faces_observed, target_face, calibration_eeg_observed, observation_faces_observed, observation_eeg_observed, face_hypothesis, scoring_function):
    from joblib import Parallel, delayed
    return Parallel(n_jobs=-1)(delayed(eval_hyp_supervised)(calibration_faces_observed, target_face, calibration_eeg_observed, observation_faces_observed, observation_eeg_observed, face_hypothesis[i, :], scoring_function) for i in range(face_hypothesis.shape[0]))
    # return [eval_hyp(faces_observed, eegs_observed, face_hypothesis[i, :], scoring_function) for i in range(face_hypothesis.shape[0])]


def get_relative_scores_supervised(estimator, calibration_faces_observed, target_face, calibration_eeg_observed, observation_eeg_observed, distances_hypothesis):

    random_state = np.random.randint(1000)
    
    # training the esitmator
    calibration_distances = [distance.euclidean(face, target_face) for face in calibration_faces_observed]

    calibration_scaler = StandardScaler()
    # Silly way to apply scaler to a list and have output as a list. Silly but just to keep working with list here and not have to change anything downstream.
    scaled_calibration_distances = calibration_scaler.fit_transform(np.array(calibration_distances).reshape(-1, 1)).flatten().tolist()
    estimator.fit(calibration_eeg_observed, scaled_calibration_distances)
        
    # eval the new EEG aligned
    scaled_pred_distance = estimator.predict(observation_eeg_observed)
    pred_distance = calibration_scaler.inverse_transform(np.array(scaled_pred_distance).reshape(-1, 1)).flatten().tolist()

    # eval the new EEG shuffled
    rng = np.random.RandomState(random_state)
    shuffled_eeg = observation_eeg_observed[rng.permutation(len(observation_eeg_observed))]
    scaled_shuffled_distance = estimator.predict(shuffled_eeg)
    shuffled_distance = calibration_scaler.inverse_transform(np.array(scaled_shuffled_distance).reshape(-1, 1)).flatten().tolist()

    # Define your scoring methods
    scorings = ['neg_root_mean_squared_error']

    # Compute scores and structure them like cross_validate output
    raw_scores = {}
    shuffled_scores = {}

    for scoring_method in scorings:
        test_key = f'test_{scoring_method}'
        
        if scoring_method == 'neg_root_mean_squared_error':
            # Raw score (single value, but put in array to match cross_validate format)
            raw_score = -np.sqrt(mean_squared_error(distances_hypothesis, pred_distance))
            raw_scores[test_key] = np.array([raw_score])
            
            # Shuffled scores
            shuffled_score = -np.sqrt(mean_squared_error(distances_hypothesis, shuffled_distance))
            shuffled_scores[test_key] = np.array([shuffled_score])
    
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
