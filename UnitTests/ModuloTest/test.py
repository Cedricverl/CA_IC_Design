import numpy as np

beta = np.float64(9.57237156)

overflowvarneg = np.int8(-12)
overflowvarpos = np.int8(6)

acctotstepneg = np.uint8(0)
acctotsteppos = np.int64(0)

acctotstepneg = (overflowvarneg*beta).astype(np.uint8)
acctotstepneg1 = (overflowvarneg*beta)
acctotsteppos = (overflowvarpos*beta).astype(np.uint8)
acctotsteppos1 = (overflowvarpos*beta)

acctotmodneg = np.mod(acctotstepneg1, 256)
acctotmodpos = np.mod(acctotsteppos1, 256)
print("No rounding result of negative multiplication: ", str(acctotstepneg1))
print("No rounding result of positive multiplication: ", str(acctotsteppos1))
print("After modulo operation (for negative values): ", str(acctotmodneg))
print("After modulo operation (for positive values): ", str(acctotmodpos))
print("\n")
expected_value_ofneg = acctotmodneg
expected_value_ofpos = acctotsteppos1

print("expected value for the negative starting point: ",str(expected_value_ofneg) + ", Actual result after uint8 casting:", str(acctotstepneg))
print("expected value for the positive starting point: ",str(expected_value_ofpos) + ", Actual result after uint8 casting:", str(acctotsteppos))




