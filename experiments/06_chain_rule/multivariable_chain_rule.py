def compute_y(x1, x2):
    return x1**2 + 3*x2

def compute_z(y):
    return 2*y

def compute_dy_dx1(x1):
    return 2*x1

def compute_dy_dx2():
    return 3

def compute_dz_dy():
    return 2

def compute_dz_dx1(x1):
    dz_dy = compute_dz_dy()
    dy_dx1 = compute_dy_dx1(x1)
    return dz_dy * dy_dx1

def compute_dz_dx2():
    dz_dy = compute_dz_dy()
    dy_dx2 = compute_dy_dx2()
    return dz_dy * dy_dx2

def compute_direct_dz_dx1(x1):
    return 4 * x1

def compute_direct_dz_dx2():
    return 6

x1 = 2
x2 = 3

# 2. Calculate original y and z values
y = compute_y(x1, x2)
z = compute_z(y)

# 3. Calculate both partial derivatives
chain_dz_dx1 = compute_dz_dx1(x1)
chain_dz_dx2 = compute_dz_dx2()

# Direct calculations
direct_dz_dx1 = compute_direct_dz_dx1(x1)
direct_dz_dx2 = compute_direct_dz_dx2()

# Compare results
print(f"dz/dx1 -> Chain Rule: {chain_dz_dx1} | Direct: {direct_dz_dx1}")
print(f"dz/dx2 -> Chain Rule: {chain_dz_dx2} | Direct: {direct_dz_dx2}")