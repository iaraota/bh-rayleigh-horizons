# constants
Gconst = 6.67259e-11  # m^3/(kg*s^2)
clight = 2.99792458e8  # m/s
MSun = 1.989e30  # kg
parsec = 3.08568025e16  # m

# in seconds
tSun = MSun * Gconst / clight ** 3  # from Solar mass to seconds
Dist = 1.0e6 * parsec / clight  # from Mpc to seconds
