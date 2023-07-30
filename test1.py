from sympy import symbols, Eq, solve

currentVal = [0]
updatedVal = 0
valX = 0
valY = 0

valXtemp = 0
valYtemp = 0
valXtemp1 = 0
valYtemp1 = 0
c = 0

def find3DPosition(r1, r2, r3, a1, b1, c1, a2, b2, c2, a3, b3, c3):

    

    x = symbols('x', real=True)
    y = symbols('y', real=True)
    z = symbols('z', real=True)

    a = Eq((x-a1)**2 + (y-b1)**2 + (z-c1)**2, r1**2)
    b = Eq((x-a2)**2 + (y-b2)**2 + (z-c2)**2, r2**2)
    c = Eq((x-a3)**2 + (y-b3)**2 + (z-c3)**2, r3**2)

    sol = solve((a, b, c), (x, y, z))
    # print(sol)
    global valX
    global valY
    global valXtemp
    global valYtemp

    if len(sol) == 0:
        valX = valXtemp
        valY = valYtemp

    else:
        valX = sol[0][0]
        valY = sol[0][1]
        valXtemp = valX
        valYtemp = valY

    # valX = sol[0][0]
    # valY = sol[0][1]
    if (valX > 1000):
        valX = 1000
    if (valY > 1000):
        valY = 1000

    print("x = ", valX)
    print("y = ", valY)




find3DPosition(8*100, 8*100, 5*100, 0, 0, 0, 837, 0, 0, 0, 665, 0)