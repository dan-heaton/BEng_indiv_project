from math import sqrt, pi, exp

Y = [[0,0,0], [0.5,0,0], [0.5,-1,0], [0,0,-0.5],
        [1,0,0], [1,0,0.5], [0,0.5,0], [0,-0.5,0]]

c1_means = [[0,0,0], [1,0,0], [1,1,1]]
c1_vars  = [[[1,0,0], [0,2,0], [0,0,1]],
            [[1,0,0], [0,1,0], [0,0,1]],
            [[2,0,0], [0,2,0], [0,0,1]]]
c1_weights = [0.5, 0.3, 0.2]
c2_means = [[0,-1,0], [1,-1,0], [0,-1,-1]]
c2_vars = [[[1,0,0], [0,1,0], [0,0,1]],
           [[1,0,0], [0,1,0], [0,0,2]],
           [[2,0,0], [0,2,0], [0,0,1]],]
c2_weights = [0.4, 0.3, 0.3]

p_y_c1 = 1
p_y_c2 = 1
for i in range(8):              #for each yt feature vector
    weighted_sum_c1 = 0
    weighted_sum_c2 = 0
    for j in range(3):          #for each component
        temp_c1 = 0
        sigma_c1 = 1
        temp_c2 = 0
        sigma_c2 = 1
        for k in range(3):      #for each dimension
            temp_c1 += ((Y[i][k] - c1_means[j][k])**2) / c1_vars[j][k][k]
            sigma_c1 *= c1_vars[j][k][k]
            temp_c2 += ((Y[i][k] - c2_means[j][k])**2) / c2_vars[j][k][k]
            sigma_c2 *= c2_vars[j][k][k]
        p_yt_c1 = (1/sqrt(((2 * pi)**3)*sigma_c1))*exp(-0.5*temp_c1)
        weighted_sum_c1 += p_yt_c1 * c1_weights[j]
        p_yt_c2 = (1/sqrt(((2 * pi)**3)*sigma_c2))*exp(-0.5*temp_c2)
        weighted_sum_c2 += p_yt_c2 * c2_weights[j]
    p_y_c1 *= weighted_sum_c1
    p_y_c2 *= weighted_sum_c2

print("P(Y|C1) = " + str(p_y_c1))
print("P(Y|C2) = " + str(p_y_c2))
print()

p_c1 = 0.5
p_c2 = 0.5
p_y = p_c1*p_y_c1 + p_c2*p_y_c2

p_c1_y = p_c1*p_y_c1 / p_y
p_c2_y = p_c2*p_y_c2 / p_y

if p_c1_y > p_c2_y:
    print("Most likely class: C1")
    print("Estimated probability: " + str(p_c1_y))
else:
    print("Most likely class: C2")
    print("Estimated probability: " + str(p_c2_y))