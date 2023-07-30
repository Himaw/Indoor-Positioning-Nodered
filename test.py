import time
from sympy import symbols, Eq, solve
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import json


currentVal = [0]
updatedVal = 0
valX = 0
valY = 0

valXtemp = 0
valYtemp = 0
valXtemp1 = 0
valYtemp1 = 0
c = 0

distance_a1_a2 = 3.0
meter2pixel = 100
range_offset = 0.9


def uwb_range_offset(uwb_range):

    temp = uwb_range
    return temp


# def find3DPosition(r1, r2, r3, a1, b1, c1, a2, b2, c2, a3, b3, c3):
#     x = symbols('x', real=True)
#     y = symbols('y', real=True)
#     z = symbols('z', real=True)

#     a = Eq((x-a1)**2 + (y-b1)**2 + (z-c1)**2, r1**2)
#     b = Eq((x-a2)**2 + (y-b2)**2 + (z-c2)**2, r2**2)
#     c = Eq((x-a3)**2 + (y-b3)**2 + (z-c3)**2, r3**2)

#     sol = solve((a, b, c), (x, y, z))

#     global valX
#     global valY
#     global valXtemp
#     global valYtemp

#     if len(sol) == 0:
#         valX = valXtemp
#         valY = valYtemp
#     else:
#         valX = sol[0][0]
#         valY = sol[0][1]
#         valXtemp = valX
#         valYtemp = valY

#     if valX > 1000:
#         valX = 1000
#     if valY > 1000:
#         valY = 1000

    # print("x =", valX)
    # print("y =", valY)
   

def kalman_xy(x, P, measurement, R,
              motion = np.matrix('0. 0. 0. 0.').T,
              Q = np.matrix(np.eye(4))):
    """
    Parameters:    
    x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    measurement: observed position
    R: measurement noise 
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    """
    return kalman(x, P, measurement, R, motion, Q,
                  F = np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      '''),
                  H = np.matrix('''
                      1. 0. 0. 0.;
                      0. 1. 0. 0.'''))


def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H 
    '''
    # UPDATE x, P based on measurement m    
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    x = x + K*y
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K*H)*P

    # PREDICT x, P based on motion
    x = F*x + motion
    P = F*P*F.T + Q

    return x, P

import numpy as np

def trilateration(positions, distances):
    # Get the number of anchors
    num_anchors = len(positions)

    # Construct the matrix A and vector b for the linear system of equations
    A = np.zeros((num_anchors - 1, 2))
    b = np.zeros(num_anchors - 1)

    for i in range(num_anchors - 1):
        A[i, 0] = 2 * (positions[i + 1, 0] - positions[0, 0])
        A[i, 1] = 2 * (positions[i + 1, 1] - positions[0, 1])
        b[i] = distances[0]**2 - distances[i + 1]**2 - positions[0, 0]**2 + positions[i + 1, 0]**2 - positions[0, 1]**2 + positions[i + 1, 1]**2

    # Solve the linear system of equations
    result = np.linalg.lstsq(A, b, rcond=None)

    # Calculate the estimated position
    x = result[0][0]
    y = result[0][1]

    return x, y



def main():
    

    

    x_kalman = np.matrix('0. 0. 0. 0.').T 
    P = np.matrix(np.eye(4))*1000 # initial uncertainty
    R = 0.01**2 
    valXkalman = 0
    valYkalman = 0
    global valXtemp
    global valYtemp

    # generating random data values
    x = []
    y = []

    
    # enable interactive mode
    plt.ion()
    
    # creating subplot and figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, marker='o', color='g',markersize=15)
    line2, = ax.plot(0, 0, marker='o', color='r',markersize=14)
    line3, = ax.plot(837, 0, marker='o', color='r',markersize=14)
    line4, = ax.plot(0, 665, marker='o', color='r',markersize=14)


    a1_range = 0.0
    a2_range = 0.0
    a3_range = 0.0
    range_anc84 = 0.0
    range_anc82 = 0.0
    range_anc83 = 0.0

   
    
 
    # mlist = [[{'A': '82', 'R' : '0.97'},{'A': '83', 'R' : '0.97'},{'A': '84', 'R' : '0.97'}],[{'A': '82', 'R' : '0.5'},{'A': '83', 'R' : '0.5'},{'A': '84', 'R' : '0.9'}]]
   
 

    # node_count = 0
    # list = read_data()
    # print(list)
    # list = mlist[0]
    # Set the range of x-axis
    # plt.xlim(0, 837)
    # Set the range of y-axis
    # plt.ylim(0, 665)


    while True:

         # Set the range of x-axis
        plt.xlim(0, 837)
        # Set the range of y-axis
        plt.ylim(0, 665)


        list = sys.stdin.readline()
        # list = int(list)
        # print(list)

        json_object = json.loads(list)
        links = json_object["links"]
        if len(links) == 3:
            
            if float(links[0]["A"]) == 82:
                a1_range = uwb_range_offset(float(links[0]["R"]))
            elif float(links[0]["A"]) == 83:
                a2_range = uwb_range_offset(float(links[0]["R"]))
            elif float(links[0]["A"]) == 84:
                a3_range = uwb_range_offset(float(links[0]["R"]))


            if float(links[1]["A"]) == 82:
                a1_range = uwb_range_offset(float(links[1]["R"]))
            elif float(links[1]["A"]) == 83:
                a2_range = uwb_range_offset(float(links[1]["R"]))
            elif float(links[1]["A"]) == 84:
                a3_range = uwb_range_offset(float(links[1]["R"]))

            
            if float(links[2]["A"]) == 82:
                a1_range = uwb_range_offset(float(links[2]["R"]))
            elif float(links[2]["A"]) == 83:
                a2_range = uwb_range_offset(float(links[2]["R"]))
            elif float(links[2]["A"]) == 84:
                a3_range = uwb_range_offset(float(links[2]["R"]))

        else:
            valX = valXtemp
            valY = valYtemp
        
       

        # for one in list:
        #     if one["A"] == "82":
        #         a1_range = uwb_range_offset(float(one["R"]))
                
        #         # a1_range = random.uniform(3,7)
        #         # print(a1_range)
        #         # print()
        #         node_count += 1
                

        #     if one["A"] == "83":
        #         a2_range = uwb_range_offset(float(one["R"]))
            
        #         # a2_range = random.uniform(3,7)
        #         # print(a2_range)
        #         node_count += 1
                

        #     if one["A"] == "84":
        #         a3_range = uwb_range_offset(float(one["R"]))
                
        #         # a3_range = random.uniform(3,7)
        #         # print(a3_range)
        #         node_count += 1
                
        range_anc82 = a1_range
        range_anc83 = a2_range
        range_anc84 = a3_range
        # range_anc1 = random.uniform(3.00,10.00)
        # range_anc2 = random.uniform(3.00,10.00)
        # range_anc3 = random.uniform(3.00,10.00)


        #         # print(range_anc3)

        
        # if node_count == 3:
        
        # find3DPosition(range_anc1*100, range_anc2*100,
        #                 range_anc3*100, 0, 0, 0, 837, 0, 0, 0, 665, 0)
        
        # Example usage
        anchors = np.array([(0, 0), (835, 0), (0, 665)])  # Anchor positions
        distances = np.array([range_anc82*100, range_anc83*100, range_anc84*100])  # Distance measurements
        # distances = np.array([343, 597, 897])  # Distance measurements


        estimated_position = trilateration(anchors, distances)
        # print("Estimated position:", estimated_position)
                
        # print(x_kalman)

        valX = estimated_position[0]
        valY = estimated_position[1]

        if valX > 835:
            valX = valXtemp
        elif valX < 0:
            valX = valXtemp
        if valY > 665:
            valY = valYtemp
        elif valY < 0:
            valY = valYtemp


        
        x_kalman, P = kalman_xy(x_kalman, P, (valX,valY), R)
        valXkalman = int(x_kalman[:2][0])
        valYkalman = int(x_kalman[:2][1])


        if valXkalman > 835:
            valXkalman = valXtemp
        elif valXkalman < 0:
            valXkalman = valXtemp
        if valYkalman > 665:
            valYkalman = valYtemp
        elif valYkalman < 0:
            valYkalman = valYtemp

        # print(a1_range)
        # print(a2_range)
        # print(a3_range)

        # positionData = { "x" : valXkalman, "y" : valYkalman }
        # print(int(valXkalman))
        # print(int(valYkalman))
        # testx = random.randint(0,835)
        # testy = random.randint(0,667)

        valXtemp = valXkalman
        valYtemp = valYkalman



        line1.set_xdata(valX)
        line1.set_ydata(valY)
        fig.canvas.draw()
    
        fig.canvas.flush_events()

        
        print(valXkalman, valYkalman)
        # print(testx,testy)
        time.sleep(0.1)

        # print(valXkalman)
        

    

    print("\n")
    print("\n")
    print("\n")

     
            

if __name__ == '__main__':
    main()


# "{"links":[{"A":"83","R":"2.5"},{"A":"84","R":"2.0"},{"A":"82","R":"2.3"}]}"
