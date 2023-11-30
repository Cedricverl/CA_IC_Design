from HDC_library import lookup_generate, encode_HDC_RFF, vthreshold
import numpy as np
np.random.seed(seed=69) # nice

# ## Lookup generate
dim = 100
n_keys = 3 # size of imput vector
maxval = 64 # LUT size
mode = 1
B = 8
# random weights mode 0
result_weights = lookup_generate(dim, n_keys, mode=0)
# print("weights", np.average(result_weights))  # should be about 0 if p = 0.5
assert(abs(np.average(result_weights)) <= 0.1)
assert(result_weights.shape == (n_keys, dim))

# LUT mode 1
result_lut = lookup_generate(dim, maxval, mode=1)
result_lut_avg = np.average(result_lut, 1)
# print("LUT", result_lut_avg)  # should go from -1 to 1 in a quasi-linear way

assert(result_lut.shape == (maxval, dim))
odds = [index/(maxval-1)*1 + (1-index/(maxval-1))*-1 for index in range(maxval)]
diff = np.abs(np.subtract(odds, result_lut_avg))

assert(all(diff < 0.3))   # check wether probabilities are what we intended

## Testing encoding: encode_HDC_RFF
# generate random input vector
input_vector = np.random.randint(maxval, size=3)
# L(x)
Lx = result_lut[input_vector,:]

# HD: xor Lx with weights
HD = Lx*result_weights

# Get the sums
sums = np.sum(HD, axis=0)

# check wether encode_HDC_RFF gives the same result:
functio_output = encode_HDC_RFF(input_vector, result_weights, result_lut, dim)
assert(all(encode_HDC_RFF(input_vector, result_weights, result_lut, dim) == sums))

### Testing the threshold function
t = 40
assert(vthreshold(-5, t, B) == -1)
assert(vthreshold(100, t, B) == 0)
assert(vthreshold(150, t, B) == 0)
assert(vthreshold(200, t, B) == 1)
assert(vthreshold(250, t, B) == 1)


# Training HDC encoder
## TODO
a = 1

print("All tests finished succesfully!")
