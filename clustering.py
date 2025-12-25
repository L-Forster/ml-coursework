from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
import os
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_covtype

os.environ["LOKY_MAX_CPU_COUNT"] = "-1"

'''
FROM INFO FILE:

  Study Code USFS ELU Code			Description
	 1	   2702		Cathedral family - Rock outcrop complex, extremely stony.
	 2	   2703		Vanet - Ratake families complex, very stony.
	 3	   2704		Haploborolis - Rock outcrop complex, rubbly.
	 4	   2705		Ratake family - Rock outcrop complex, rubbly.
	 5	   2706		Vanet family - Rock outcrop complex complex, rubbly.
	 6	   2717		Vanet - Wetmore families - Rock outcrop complex, stony.
	 7	   3501		Gothic family.
	 8	   3502		Supervisor - Limber families complex.
	 9	   4201		Troutville family, very stony.
	10	   4703		Bullwark - Catamount families - Rock outcrop complex, rubbly.
	11	   4704		Bullwark - Catamount families - Rock land complex, rubbly.
	12	   4744		Legault family - Rock land complex, stony.
	13	   4758		Catamount family - Rock land - Bullwark family complex, rubbly.
	14	   5101		Pachic Argiborolis - Aquolis complex.
	15	   5151		unspecified in the USFS Soil and ELU Survey.
	16	   6101		Cryaquolis - Cryoborolis complex.
	17	   6102		Gateview family - Cryaquolis complex.
	18	   6731		Rogert family, very stony.
	19	   7101		Typic Cryaquolis - Borohemists complex.
	20	   7102		Typic Cryaquepts - Typic Cryaquolls complex.
	21	   7103		Typic Cryaquolls - Leighcan family, till substratum complex.
	22	   7201		Leighcan family, till substratum, extremely bouldery.
	23	   7202		Leighcan family, till substratum - Typic Cryaquolls complex.
	24	   7700		Leighcan family, extremely stony.
	25	   7701		Leighcan family, warm, extremely stony.
	26	   7702		Granile - Catamount families complex, very stony.
	27	   7709		Leighcan family, warm - Rock outcrop complex, extremely stony.
	28	   7710		Leighcan family - Rock outcrop complex, extremely stony.
	29	   7745		Como - Legault families complex, extremely stony.
	30	   7746		Como family - Rock land - Legault family complex, extremely stony.
	31	   7755		Leighcan - Catamount families complex, extremely stony.
	32	   7756		Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
	33	   7757		Leighcan - Catamount families - Rock outcrop complex, extremely stony.
	34	   7790		Cryorthents - Rock land complex, extremely stony.
	35	   8703		Cryumbrepts - Rock outcrop - Cryaquepts complex.
	36	   8707		Bross family - Rock land - Cryumbrepts complex, extremely stony.
	37	   8708		Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
	38	   8771		Leighcan - Moran families - Cryaquolls complex, extremely stony.
	39	   8772		Moran family - Cryorthents - Leighcan family complex, extremely stony.
	40	   8776		Moran family - Cryorthents - Rock land complex, extremely stony.


        Note:   First digit:  climatic zone             Second digit:  geologic zones
                1.  lower montane dry                   1.  alluvium
                2.  lower montane                       2.  glacial
                3.  montane dry                         3.  shale
                4.  montane                             4.  sandstone
                5.  montane dry and montane             5.  mixed sedimentary
                6.  montane and subalpine               6.  unspecified in the USFS ELU Survey
                7.  subalpine                           7.  igneous and metamorphic
                8.  alpine                              8.  volcanic

        The third and fourth ELU digits are unique to the mapping unit 
        and have no special meaning to the climatic or geologic zones.

'''



def generate_zones(soil_type):

    clim_zones = {
        1: "lower montane dry",
        2: "lower montane",
        3: "montane dry",
        4: "montane",
        5: "montane dry and montane",
        6: "montane and subalpine",
        7: "subalpine",
        8: "alpine"
    }

    geo_zones = {
        1: "alluvium",
        2: "glacial",
        3: "shale",
        4: "sandstone",
        5: "mixed sedimentary",
        6: "unspecified",
        7: "igneous and metamorphic",
        8: "volcanic"
    }

    soil_to_elu = {
        1: 2702, 2: 2703, 3: 2704, 4: 2705, 5: 2706, 6: 2717, 7: 3501, 8: 3502, 9: 4201, 10: 4703, 11: 4704, 12: 4744, 13: 4758, 14: 5101,
        15: 5151, 16: 6101, 17: 6102, 18: 6731, 19: 7101, 20: 7102, 21: 7103, 22: 7201, 23: 7202, 24: 7700, 25: 7701, 26: 7702, 27: 7709, 28: 7710,
        29: 7745, 30: 7746, 31: 7755, 32: 7756, 33: 7757, 34: 7790, 35: 8703, 36: 8707, 37: 8708, 38: 8771, 39: 8772, 40: 8776
    }

    elu_code = soil_to_elu[soil_type]
    climatic = int(str(elu_code)[0])
    geologic = int(str(elu_code)[1])

    return clim_zones[climatic], geo_zones[geologic]


def add_features(data):

    n_features = data.shape[1] - 40
    feature_columns = [f'Feature_{i}' for i in range(n_features)]
    soil_columns = [f'Soil_Type{i + 1}' for i in range(40)]
    df = pd.DataFrame(data, columns=feature_columns + soil_columns)

    climatic_zones = [
        "lower montane dry", "lower montane", "montane dry", "montane",
        "montane dry and montane", "montane and subalpine", "subalpine", "alpine"
    ]

    geologic_zones = [
        "alluvium", "glacial", "shale", "sandstone", "mixed sedimentary",
        "unspecified", "igneous and metamorphic", "volcanic"
    ]

    for zone in climatic_zones + geologic_zones:
        df[zone] = 0

    for i, row in df.iterrows():
        for j in range(1, 41):
            if row[f'Soil_Type{j}'] == 1:
                climatic, geologic = generate_zones(j)
                if climatic in df.columns: # set the zones for each item
                    df.at[i, climatic] += 1
                if geologic in df.columns:
                    df.at[i, geologic] += 1

    return df.values



k = 7
covtype = fetch_covtype()

print(min(covtype.data[:,1]), max(covtype.data[:,1])) #aspect 0 and 360 are close
seed = 1
np.random.seed(seed)
indices = np.random.choice(covtype['data'].shape[0], 10000, replace=False)
X_sample = covtype['data'][indices]
y_sample = covtype['target'][indices]


# adding the additional geological and climatic features from soil type
X_sample = add_features(X_sample)

# fixing aspect stuff
aspects = X_sample[:, 1]
print(max(aspects))
# Adjust aspects so it loops around angle
# aspects_sin = np.sin(np.deg2rad(aspects)) * 360 + 180
# aspects_cos = np.cos(np.deg2rad(aspects))  * 360 + 180
# X_sample = np.hstack((
#     X_sample[:, :1],
#     aspects_sin[:, np.newaxis],
#     aspects_cos[:, np.newaxis],
#     X_sample[:, 2:]
# ))

#standardising the data
scaler = StandardScaler()
X_sample = scaler.fit_transform(X_sample)
print(X_sample.shape)

pca = PCA(n_components='mle', random_state=1)
X_sample = pca.fit_transform(X_sample)
print("shape: ", X_sample.shape)

# defining an unsupervised scoring method
def silhouette_scorer(estimator, X, y=None):
    return silhouette_score(X, estimator.predict(X))


def kmm_best_params():
    kmeans_params = {
        'init': Categorical(['k-means++', 'random']),
        'max_iter': Integer(100, 500),
        'algorithm': Categorical(['lloyd', 'elkan'],),
        'n_init': Integer(1, 50),
    }
    kmeans = KMeans(n_clusters=k, n_init=30, random_state=seed)
    bayes = BayesSearchCV(kmeans,kmeans_params,n_iter=50,scoring=silhouette_scorer,cv=5,n_jobs=-1,random_state=seed)
    bayes.fit(X_sample)
    return bayes.best_params_


def kmeans_estimator(best_params, K):
    print(best_params)
    kmm_params = {**best_params, 'n_clusters': K}
    kmm = KMeans(**kmm_params, random_state=seed)
    kmm.fit(X_sample)
    return kmm.labels_

# bayes search for best params
def gmm_best_params(subset, k):
    gmm_params = {
        'covariance_type': Categorical(['full', 'tied', 'diag', 'spherical']),
        'init_params': Categorical(['kmeans', 'random']),
        'reg_covar': Categorical([1e-6, 1e-7, 1e-5, 1e-3, 1e-4]),
        'max_iter': Integer(80, 120),
        'tol': Categorical([1e-3, 1e-4, 1e-2]),
        'n_init': Integer(1, 25)
    }
    gmm = GaussianMixture(n_components=k, random_state=seed)
    bayes = BayesSearchCV(gmm,gmm_params,n_iter=50,scoring=silhouette_scorer,cv=5,n_jobs=-1,random_state=seed)
    bayes.fit(X_sample)
    print("Best Parameters:", bayes.best_params_)
    return bayes.best_params_

# run the model with input parameters
def gmm_estimator(best_params, K):
    gmm_params = {**best_params, 'n_components': K}
    gmm = GaussianMixture(**gmm_params, random_state=seed)
    gmm.fit(X_sample)
    gmm_labels = gmm.predict(X_sample)
    print(gmm.get_params())
    return gmm_labels


# task 4
# generate random labels
rand_baseline = np.zeros(X_sample.shape)
print(X_sample.shape)
rand_baseline_labels = np.random.randint(1, k + 1, size=len(X_sample))


def calculate_error(labels):
    """
    Optimized O(n) version using numpy.
    Counts pairs that share same ground truth but are in different clusters.
    """
    # Total pairs where y_sample[i] == y_sample[j]: sum of C(n,2) per true class
    _, true_counts = np.unique(y_sample, return_counts=True)
    total_pairs = np.sum(true_counts * (true_counts - 1) // 2)
    
    # Correctly paired: same true label AND same predicted cluster
    # Create combined keys and count unique (true, pred) pairs
    combined = y_sample.astype(np.int64) * (labels.max() + 1) + labels
    _, pair_counts = np.unique(combined, return_counts=True)
    correctly_paired = np.sum(pair_counts * (pair_counts - 1) // 2)
    
    # Error = pairs that should be together (same truth) but aren't (diff cluster)
    error_count = total_pairs - correctly_paired
    
    return error_count, total_pairs

# getting the best params and best labels..

gmm_params = gmm_best_params(X_sample,k)

gmm_labels = gmm_estimator(gmm_params, k)

kmeans_params = kmm_best_params()

kmeans_labels = kmeans_estimator(kmeans_params, k)

# calculate the ARI to find a useful performance metric (random to true)
print("GMM ARI", adjusted_rand_score(gmm_labels, y_sample))
print("Random Baseline ARI", adjusted_rand_score(rand_baseline_labels, y_sample))
print("KMEANS ARI",adjusted_rand_score(kmeans_labels, y_sample))




error_count, total_pairs = calculate_error(kmeans_labels)
print("Kmeans Error: ", error_count, "Total Pairs: ", total_pairs)

error_count, total_pairs = calculate_error(rand_baseline_labels)
print("Random Baseline Error: ", error_count, "Total Pairs: ", total_pairs)
#
error_count, total_pairs = calculate_error(gmm_labels)
print("GMM Error: ", error_count, "Total Pairs: ", total_pairs)


'''
By applying hyperparameter optimisation:

Best Parameters: OrderedDict([('covariance_type', 'diag'), ('init_params', 'random'), ('max_iter', 120), ('n_init', 1), ('reg_covar', 0.001), ('tol', 0.0001)])
{'covariance_type': 'diag', 'init_params': 'random', 'max_iter': 120, 'means_init': None, 'n_components': 7, 'n_init': 1, 'precisions_init': None, 'random_state': 1, 'reg_covar': 0.001, 'tol': 0.0001, 'verbose': 0, 'verbose_interval': 10, 'warm_start': False, 'weights_init': None}
OrderedDict([('algorithm', 'lloyd'), ('init', 'random'), ('max_iter', 454), ('n_init', 23)])
GMM ARI 0.18263589526506036
KMEANS ARI 0.10551137387020575

Kmeans Error:  13191142 Total Pairs:  18906728
Random Baseline Error:  16204045 Total Pairs:  18906728
GMM Error:  7549382 Total Pairs:  18906728

'''