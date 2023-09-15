import time
import numpy as np
import matplotlib.pyplot as plt
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

    x = []
    y = []

    
    # enable interactive mode
    plt.ion()
    
    # creating subplot and figure
    img = plt.imread("/Users/himasarawarnakulasuriya/Desktop/IndoorNodered/Indoor-Positioning-Nodered/Map_matplotlib/room12.jpg")
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 835, 0, 665])


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, marker='o', color='g',markersize=9)
    line2, = ax.plot(0, 0, marker='o', color='r',markersize=14)
    line3, = ax.plot(837, 0, marker='o', color='r',markersize=14)
    line4, = ax.plot(0, 665, marker='o', color='r',markersize=14)


    a1_range = 0.0
    a2_range = 0.0
    a3_range = 0.0
    range_anc84 = 0.0
    range_anc82 = 0.0
    range_anc83 = 0.0



    while True:

         # Set the range of x-axis
        plt.xlim(0, 837)
        # Set the range of y-axis
        plt.ylim(0, 665)


        list = sys.stdin.readline()
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

    
                    
            range_anc82 = a1_range
            range_anc83 = a2_range
            range_anc84 = a3_range
   
            # Example usage
            anchors = np.array([(0, 0), (835, 0), (0, 665)])  # Anchor positions
            distances = np.array([range_anc82*100, range_anc83*100, range_anc84*100])  # Distance measurements


            estimated_position = trilateration(anchors, distances)
         

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

        else:
            valX = valXtemp
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

 

        valXtemp = valXkalman
        valYtemp = valYkalman



        line1.set_xdata(valX)
        line1.set_ydata(valY)
        fig.canvas.draw()
    
        fig.canvas.flush_events()

        
        print(valXkalman, valYkalman)
        time.sleep(0.1)

        

    

    print("\n")
    print("\n")
    print("\n")

     
            

if __name__ == '__main__':
    main()


