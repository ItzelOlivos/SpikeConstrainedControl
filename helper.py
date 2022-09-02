import numpy as np


class Environment:
    def __init__(self, a, q, cx, cu, p0, max_t, trials):
        self.a = a
        self.q = q
        self.cx = cx
        self.cu = cu
        self.proc_noise = np.sqrt(q) * np.random.randn(max_t, trials)
        self.x = np.zeros([max_t, trials])
        self.x[0] = p0[0] + np.sqrt(p0[1]) * np.random.randn(trials)

    def step(self, t, u):
        self.x[t] = self.a * self.x[t-1] + u + self.proc_noise[t-1]
        cost = self.x[t]**2 * self.cx + u**2 * self.cu
        return cost


class Agent:
    def __init__(self, a, g, w, stimuli_range, M, N, l, b0, k, max_t, trials):

        # Neural parameters
        self.g = g
        self.w = w
        self.M = M
        self.N = N
        self.delta = (stimuli_range[1] - stimuli_range[0])/M

        # World assumptions
        self.a = a
        self.r = w * self.delta / (g * np.sqrt(2*np.pi))
        self.k = k
        self.q = self.k * (-1 - self.k * self.a + self.a**2) * self.r / (self.k-self.a)
        self.P = (self.q + (-1 + self.a**2)*self.r + np.sqrt(4*self.q*self.r + (-self.q + (1 - self.a**2)*self.r)**2))/2

        # Control gain
        self.l = l

        # Input layer
        self.tuning_centers = np.array([stimuli_range[0] + i * self.delta for i in range(M)])
        self.ai = 1/w**2 * np.ones(M)
        self.bi = self.tuning_centers/w**2
        self.fi = np.zeros([M, trials])
        self.ri = np.zeros([M, trials])
        self.story_ri = []

        # Recurrent layer
        # self.ao = np.array([2 * (np.cos((2*i-1)*np.pi/(2*N)))**2 - 1 for i in range(N)])/N
        # self.bo = (np.array([np.cos((2*i-1)*np.pi/(2*N)) for i in range(N)]))/N
        self.ao = np.array([2 * (np.cos(2 * i * np.pi / N))**2 - 1 for i in range(N)])/N
        self.bo = (np.array([np.cos(2 * i * np.pi / N) for i in range(N)]))/N

        self.aop = self.ao/np.linalg.norm(self.ao)**2
        self.bop = self.bo/np.linalg.norm(self.bo)**2
        self.W1 = np.outer(self.aop, self.ai) + self.a * np.outer(self.bop, self.bi)
        self.W2 = np.outer(self.aop, self.ao) + (self.a + self.l) * np.outer(self.bop, self.bo)
        self.W3 = np.outer(self.bop, self.bo)

        self.c = - np.ones([N, trials])

        self.fo = b0[1] * np.outer(self.aop, np.ones(trials))
        self.ro = np.random.uniform(0, .001, [N, trials])
        self.story_ro = []

        # Auxiliary variables
        self.mu = np.zeros([max_t, trials])
        self.sig = np.zeros([max_t, trials])
        # self.sig = self.P * np.ones([max_t, trials])
        self.mu[0] = b0[0]
        self.sig[0] = b0[1]
        self.baseline = []
        self.trials = trials
        self.y = np.zeros([max_t, trials])

    def bel_update(self, t, obs):
        # This is the same k as self.k
        k = self.a * self.sig[t-1] / (self.sig[t-1] + self.r)
        self.mu[t] = (self.a + self.l) * self.mu[t-1] + self.k * (obs - self.mu[t-1])
        self.sig[t] = self.q + self.a**2 * self.sig[t-1] - k * (self.a * self.sig[t-1])
        return self.w**2/self.r + (8 + self.mu[t]**2)/(4 * self.sig[t])

    def neural_computation(self, t, stimulus):
        # Spikes in input layer
        self.fi = np.array(
            [self.g * np.exp(-(stimulus - self.tuning_centers[i]) ** 2 / (2 * self.w ** 2)) for i in range(self.M)])
        self.ri = np.random.poisson(self.fi)
        self.story_ri.append(self.ri)

        self.y[t] = np.dot(self.bi, self.ri) / np.dot(self.ai, self.ri)

        # Layer has access to its own firing rates, but the decoder uses R
        den = self.a**2 + self.q * (np.dot(self.ai, self.ri) + np.dot(self.ao, self.fo))
        self.fo = (np.dot(self.W1, self.ri) + np.dot(self.W2, self.fo) + self.l * (np.dot(self.ai, self.ri) / np.dot(self.ao, self.fo)) * np.dot(self.W3, self.fo))/den

        self.baseline.append(np.min(self.fo))
        self.fo += self.c * np.min(self.fo)
        self.ro = np.random.poisson(self.fo)
        self.story_ro.append(self.ro)

        # Decoding layer
        self.mu[t] = np.dot(self.bo, self.ro) / np.dot(self.ao, self.ro)
        self.sig[t] = 1 / np.dot(self.ao, self.ro)
        return self.l * self.mu[t]
