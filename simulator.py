from matplotlib import animation

from helper import *
from visualizer import *

# Execution params
T = 100
trials = 2

# World params
a = 1.2
q = .01
cx = 1.
cu = 4.
cn = 0.001/1000

# Agent's prams
alpha = a
stimuli_range = [-100, 100]
g = 5      # Min g = 1
w = 15      # Min .2 Max 20 to ensure a flat sum. Less than 4 changes empirical P
M = 500
N = 1000    # More N minimizes the effects of orthogonality not exactly zero
l = -(-cx + (-1 + a**2)*cu + np.sqrt(4*cx*cu + (cx + (-1 + a**2)*cu)**2))/(2 * a * cu)
delta = (stimuli_range[1] - stimuli_range[0])/M
r = w * delta / (g * np.sqrt(2*np.pi))
k = (-q + (-1 + a**2)*r + np.sqrt(4*q*r + (q + (-1 + a**2)*r)**2))/(2 * a * r)
print(l)

env = Environment(a=a, q=q, cx=cx, cu=cu, p0=[0, .1], max_t=T, trials=trials)
agent = Agent(a=alpha, g=g, w=w, stimuli_range=stimuli_range, M=M, N=N, l=l, b0=[0, .1], k=k, max_t=T, trials=trials)

print("-------------", agent.r, "-------------")

total_task_cost = np.zeros([T, trials])
total_neural_cost_in = np.zeros([T, trials])
total_neural_cost_rec = np.zeros([T, trials])

u = 0
for t in range(1, T):
    total_task_cost[t] = env.step(t, u)
    u = agent.neural_computation(t, env.x[t-1])
    total_neural_cost_in[t] = np.sum(agent.ri, axis=0)
    total_neural_cost_rec[t] = np.sum(agent.ro, axis=0)

error = env.x - agent.mu
# Predicted error under assumptions matches np.mean(agent.sig) and np.mean(agent.mu**2)
P = (agent.q - agent.r + agent.a**2 * agent.r + np.sqrt(4 * agent.q * agent.r + (-agent.q + agent.r - agent.a**2 * agent.r)**2))/2

print("Task cost: ", np.mean(np.sum(total_task_cost, axis=0))/T)
print("Encoding cost: ", np.mean(np.sum(total_neural_cost_in, axis=0))/T)
print("Recoding cost: ", np.mean(np.sum(total_neural_cost_rec, axis=0))/T)

# Collect a sample trial to visualize

idx = np.random.choice(trials)
steps = np.arange(0, T)

x = env.x[:, idx]
mu = agent.mu[:, idx]
sig = agent.sig[:, idx]
sqrt_sig = np.sqrt(agent.sig[:, idx])
y = agent.y[:, idx]
sqrt_r = np.sqrt(agent.r)

neurons_in = np.arange(0, M)
ri = np.asarray([agent.story_ri[t][:, idx] for t in range(T - 1)])

neurons_rec = np.arange(0, N)
ro = np.asarray([agent.story_ro[t][:, idx] for t in range(T - 1)])

anim = create_demo(T, x, mu, sig, y, agent.r, M, ri, N, ro, agent.l)

f = r"animation-spikes.gif"
writergif = animation.PillowWriter(fps=6)
anim.save(f, writer=writergif)