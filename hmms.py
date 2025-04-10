import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
rewards = np.loadtxt("rewards.txt")


'''
adapted from
https://hmmlearn.readthedocs.io/en/latest/tutorial.html
'''

def learn_dist():
    states = 9
    n_features = 3
    start_probs = np.full(states, 1 / states)
    transmat = np.zeros((9, 9))

    for x in range(transmat.shape[0]):
        probs = np.random.random(9)
        transmat[x, :] = probs / probs.sum()
    # print(transmat)
    emission_matrix = np.random.random((states, n_features))
    emission_matrix = emission_matrix / emission_matrix.sum(axis=1, keepdims=True)

    rewards = np.loadtxt("rewards.txt", dtype=int).reshape(-1, 1)
    # print(rewards)
    # print(transmat)

    hmm_model = hmm.CategoricalHMM(n_components=states,n_iter=1000,init_params="",tol=1e-4)

    hmm_model.startprob_ = start_probs
    hmm_model.transmat_ = transmat
    hmm_model.emissionprob_ = emission_matrix
    hmm_model.fit(rewards)
    return hmm_model.transmat_, hmm_model.emissionprob_, hmm_model.startprob_



def learn_true():

    states = 9
    n_features = 3
    start_probs = np.full(states, 1 / states)
    transmat = np.zeros((9, 9))

    neighbors = {
        0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
        3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
        6: [3, 7], 7: [4, 6, 8], 8: [5, 7]
    }

    # random probs
    for state, valid_moves in neighbors.items():
        prob = 1.0 / len(valid_moves)
        for next_state in valid_moves:
            transmat[state, next_state] = prob


    rewards = np.loadtxt("rewards.txt", dtype=int).reshape(-1, 1)
    # print(rewards)
    # print(transmat)

    model = hmm.CategoricalHMM(
        n_components=states,
        n_iter=1000,
        init_params="se",
        tol=1e-4,
    )
    model.params = "se"
    model.transmat_ = transmat
    model.fit(rewards)
    # print(model.transmat_, model.emissionprob_, model.startprob_)
    return model.transmat_, model.emissionprob_, model.startprob_


def plot_emission_probabilities(emission_matrix, title="Emission Probabilities"):
    num_states, num_features = emission_matrix.shape
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(num_states)
    for reward in range(num_features):
        ax.bar(x + reward / num_features, emission_matrix[:, reward], width=0.3, label=reward)

    ax.set_title(title)
    ax.set_xlabel("States")
    ax.set_ylabel("Emission Probs")
    plt.xticks(x + 0.3, [f"State {i+1}" for i in range(num_states)])
    ax.legend()
    plt.show()

def main():
    a,b,c = learn_dist()
    print(a,b,c)
    plot_emission_probabilities(b)

    a,b,c = learn_true()
    print(a,b,c)

    plot_emission_probabilities(b)





if __name__ == '__main__':
    main()