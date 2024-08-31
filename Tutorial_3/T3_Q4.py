#(i)
d={(i,j):i+j for i in range(1,7) for j in range(1,7)}

#(ii) collect all of the (a, b) pairs that sum to each of the possible values from two to twelve
from collections import defaultdict
dinv = defaultdict(list)
for i,j in d.items(): dinv[j].append(i)

# Compute the probability measured for each of these items
# including the sum equals seven
X={i:len(j)/36. for i,j in dinv.items() }
print(X)

d={(i,j,k):((i*j*k)/2>i+j+k) for i in range(1,7)
                                for j in range(1,7)
                                    for k in range(1,7)}

dinv = defaultdict(list)
for i,j in d.items(): dinv[j].append(i)

X={i:len(j)/6.0**3 for i,j in dinv.items() }
print(X)
