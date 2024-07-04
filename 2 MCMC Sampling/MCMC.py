import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import random

# --- Probability Functions and Metropolis-Hastings Algorithm ---

@jit(nopython=True)
def norm_pdf(x, mean, std):
    """
    Calculate the normal distribution's probability density function (PDF) at point x.
    
    Parameters:
    - x: Point at which to evaluate the PDF.
    - mean: Mean of the distribution.
    - std: Standard deviation of the distribution.
    
    Returns:
    The probability density at x.
    """
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

@jit(nopython=True)
def normMH(T, sigma):
    """
    Generate samples using the Metropolis-Hastings algorithm with normal distribution proposals.
    
    Parameters:
    - T: Total number of samples including burn-in.
    - sigma: Standard deviation of the proposal distribution.
    
    Returns:
    A numpy array of samples after the burn-in period.
    """
    pi = np.empty(T)
    pi[0] = np.random.normal(0, sigma)
    for t in range(1, T):
        pi_star = np.random.normal(pi[t - 1], sigma)
        alpha = min(1, norm_pdf(pi_star, 3, 2) / norm_pdf(pi[t - 1], 3, 2))
        if np.random.uniform() < alpha:
            pi[t] = pi_star
        else:
            pi[t] = pi[t - 1]
    return pi[1000:]

# --- Helper Functions for p-bit Coding ---

def decimal(bit_sequence, fractional, integer):
    """
    Convert a binary sequence to a decimal number using fixed-point notation.
    """
    exp = integer
    num = 0
    for bit in bit_sequence:
        if bit == 1:
            num += np.power(2.0, exp) * bit
        exp -= 1
    if exp == fractional:
        return num - np.power(2, float(integer))
    else:
        print("Error: Exponent does not match the fractional position.")
        return None

def gray_to_binary(gray_sequence):
    """
    Convert a sequence of Gray code bits to a binary sequence.
    """
    binary = [gray_sequence[0]]
    for i in range(1, len(gray_sequence)):
        binary.append(binary[i-1] ^ gray_sequence[i])
    return binary

def p_bit_coding_mh(T,max_p):
    """
    Perform the Metropolis-Hastings algorithm using p-bit coding.
    """
    G = [random.randint(0, 1) for _ in range(20)]
    t = 0
    fractional = -16
    integer = len(G) + fractional
    B = gray_to_binary(G)
    pi = [decimal(B, fractional, integer)]
    while t < T - 1:
        for i in range(len(G)):
            G_new = G.copy()
            p = random.randint(0, len(B) - 1)
            G_new[p] = 1 - G[p]
            B_new = gray_to_binary(G_new)
            num_new = decimal(B_new, fractional, integer)
            alpha = min(max_p * norm_pdf(num_new, 3, 2) / norm_pdf(pi[t], 3, 2), max_p)
            if random.random() < alpha:
                pi.append(num_new)
                G = G_new
                t += 1

    return pi[10000:]

# --- Visualization ---

def plot_samples(samples):
    """
    Plot histogram of the sampled distribution.
    """
    plt.hist(samples, bins=64, density=True, alpha=0.7, color='red')
    plt.title('Sampled Distribution')
    plt.show()

def normplot():
    """
    Plot the theoretical normal distribution curve.
    """
    x = np.arange(-8, 14, 0.1)
    plt.plot(x, norm_pdf(x, 3, 2))

# --- Example Usage ---
def example_usage():
    T = 20000 * 30  # Total number of samples including burn-in
    sigma = 1       # Standard deviation of the proposal distribution
    # samples = p_bit_coding_mh(T,1)
    # samples = p_bit_coding_mh(T,0.5)
    samples = normMH(T,sigma)
    plot_samples(samples)

# Run example usage
normplot()
example_usage()
