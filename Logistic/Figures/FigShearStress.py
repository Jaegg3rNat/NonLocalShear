import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from matplotlib import rc
from matplotlib.patches import Rectangle
from scipy.special import j1 as Bj1
import fontstyle
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import seaborn as sns
from tqdm import tqdm

mu = 800
# Define the initial data set
def ufp(mu, pe):
    """
    Import the external data set of the patterned configuration
    :param mu: growth rate
    :param pe: peclet number
    :return: np.array of the density values
    """
    f = h5py.File(f'../Data/HeatMap/R_0.2/mu{mu:.2f}_w{2 * np.pi:.2f}_Pe{pe:.1f}/dat.h5', 'r')
    time = f['time'][-1]
    D = 1e-4
    comp_rad = 0.2

    u = f[f't{time}'][:] / (mu * D / comp_rad ** 2)

    return u


def kill_spots(u, xa, xb, ya, yb):
    """
    Define the window around a single spot and kill all the rest
    :param u:
    :param xa:
    :param xb:
    :param ya:
    :param yb:
    :return: np.array of the system with only one spot
    """
    for i in range(0, len(u[0, :])):
        for j in range(0, len(u[0, :])):
            if i < xa or i > xb:
                u[i, j] = 0
            if j < ya or j > yb:
                u[i, j] = 0
    return u


def set_boundary(u):
    unew = np.zeros_like(u)  #
    pts = []  # DEFINE THE LIST FOR ALL TUPLES OF COORDINATES OF BOUNDARY POINTS
    for i in range(1, len(u[0, :]) - 1):
        for j in range(1, len(u[0, :]) - 1):
            ucentral = u[i, j]
            utop = u[i, j + 1]
            ubottom = u[i, j - 1]
            uleft = u[i + 1, j]
            urigth = u[i - 1, j]
            usum = utop + ubottom + uleft + urigth
            if ucentral == 1 and usum < 4:
                unew[i, j] = ucentral
                pts.append((y[i], y[j]))
            else:
                pass
    return unew, pts


pe1,pe2,pe3,pe4 = 10,120,230,240

def plotleft_figure():

    norm = matplotlib.colors.Normalize(vmin=0., vmax=umax)
    bounds = np.array([-0.5, 0.5])
    y_bounds = np.linspace(-0.5,0.5,128)
    print(y_bounds)
    bounds2 = np.array([y_bounds[32],y_bounds[64]])
    print(bounds2)





    pcm1 = axL[0].imshow(ufp(mu, pe=pe1).T[32:64,:], norm=norm, cmap="gnuplot", origin="lower",
               extent=np.concatenate((bounds, bounds2)))

    # axL[0].set_ylim([y_bounds[32],y_bounds[64]])
    bounds2 = np.array([y_bounds[32+2], y_bounds[64+2]])
    print(bounds2)
    pcm1 = axL[1].imshow(ufp(mu, pe=pe2).T[34:64+2,:], norm=norm, cmap="gnuplot", origin="lower",
               extent=np.concatenate((bounds, bounds2)))
    # axL[1].set_ylim([y_bounds[32+2], y_bounds[64+2]])
    bounds2 = np.array([y_bounds[32+16], y_bounds[64+16]])
    print(bounds2)
    pcm = axL[2].imshow(ufp(mu, pe=pe3).T[32+16:64+16,:], norm=norm, cmap="gnuplot", origin="lower",
               extent=np.concatenate((bounds, bounds2)))
    #
    pcm = axL[3].imshow(ufp(mu, pe=pe4).T[32+16:64+16,:], norm=norm, cmap="gnuplot", origin="lower",
               extent=np.concatenate((bounds, bounds2)))

    cbar = fig.colorbar(pcm,ax=axL.ravel().tolist(), shrink = 0.9)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Normalized population density', rotation=270, fontsize=12, labelpad=22)

    for i in range(4):
        axL[i].tick_params(axis='both', which='both', labelsize=12)

    axL[0].text(-0.6, 0.1, s='A', fontsize=15,
                  bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.2'))
    axL[0].set_title('$\overline{\mbox{Pe}}$= ' f'{pe1}')
    axL[1].set_title('$\overline{\mbox{Pe}}$= ' f'{pe2}')
    axL[2].set_title('$\overline{\mbox{Pe}}$= ' f'{pe3}')
    axL[3].set_title('$\overline{\mbox{Pe}}$= ' f'{pe4}')
    axL[3].set_xlabel('$x$',fontsize = 18)
    axL[0].set_ylabel('$y$', fontsize=18)
    axL[1].set_ylabel('$y$', fontsize=18)
    axL[2].set_ylabel('$y$', fontsize=18)
    axL[3].set_ylabel('$y$', fontsize=18)


######################

def plotright_figure():
    #find line of best fit
    p= np.polyfit(pe, Delta, 2)
    ppe = np.array(pe)
    D = np.array(Delta)
    # Fit a regression line
    x_reshaped = ppe.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_reshaped, D)
    y_pred = model.predict(x_reshaped)

    # Calculate residuals
    residuals = D - y_pred
    threshold = 0.3  # Define a threshold for residuals

    # Filter out points with large residuals
    mask = np.abs(residuals) < threshold
    x_filtered = ppe[mask]
    y_filtered = D[mask]
    # plt.plot(x_filtered,y_filtered)


    slope, intercept, r_value, p_value, std_err = linregress(x_filtered, y_filtered)
    line = slope * x_filtered + intercept
    # # Plot the data and regression line with error filled region
    sns.scatterplot(x=x_filtered[1:], y=y_filtered[1:], color='k', label='Measurement')
    sns.lineplot(x=x_filtered, y=line, color='k', label='Least Squares Fit', ls= '--',alpha = 0.6)

    # Calculate error bounds
    print('std_err:', std_err)
    confidence_interval = 1.95 * std_err  # 95% confidence interval
    upper_bound = line + confidence_interval
    lower_bound = line - confidence_interval

    # Fill error region
    plt.fill_between(x_filtered, upper_bound, lower_bound, color='red', alpha=0.2)


    axR.tick_params(axis='both', which='both', labelsize=12)
    axR.set_xlabel(r'Characteristic Péclet, $\overline{\mbox{Pe}}$', fontsize=14)
    axR.set_ylabel('Taylor deformation parameter, $\Delta$', fontsize=14)
    axL[0].text(0.9, 0.1, s='B', fontsize=15,
                bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.2'))
    # axR.plot([int(pe1),int(pe2),int(pe3)],[ Delta[1],Delta[-12],Delta[-1]], 'o', color = 'r')
    # axR.legend(fontsize = 12)
    # axR.set_xlim([0,60])

    # Modify the indices to correspond to pe1, pe2, pe3, and pe4
    indices = [pe.index(pe1), pe.index(pe2), pe.index(pe3)]
    selected_Delta = [Delta[idx] for idx in indices]
    selected_pe = [pe1, pe2, pe3]

    # Plot the selected points
    axR.plot(selected_pe, selected_Delta[:3], 'o', color='r')
    axR.legend(fontsize=12)



#######################################################################################################################
#######################################################################################################################
# Data
B_list = []
A_List = []
pe = []
for i in tqdm(range(24)):
    print('####################################################')##############
    print('############ FIRST PLOT ############################')##############
    print('####################################################')##############
    #######################################################################################################################3
    ########################################################################################################################

    umax = np.max(ufp(250, pe=340))  # TRESHOLD VALUE

    norm = matplotlib.colors.Normalize(vmin=0., vmax=1.5)
    bounds = np.array([-0.5, 0.5])  # COMPLETE ARRAY SIZE
    bounds2 = np.array([-0.1, 0.1])  # SEGMENTED ARRAY
    y = np.linspace(*bounds, 128 + 1)[1:]  # Y- ARRAY DISCRETIZATION

    # u = ufp(800, int(input('Choose Peclet number: ')))
    u = ufp(mu, i*10)
    # pcm = plt.imshow(u.T[:, :], norm=norm, cmap="gnuplot", origin="lower",
    #                  extent=np.concatenate((bounds, bounds)))
    # plt.colorbar()
    # plt.show()
    # plt.close()

    #

    # ################################################################################################################################################################################################################################################

    print('#########################################################################')##############
    print('############ SECOND PLOT::   ISOLATION OF THE SPOT ######################')####################
    print('# #######################################################################')
    # #
    # #
    # print(np.max(u))
    max_value = np.max(u[32:66, 40:64 + 16])  # Find the maximum value in the matrix
    result = np.where(u == max_value)  # Get the indices where the matrix equals the maximum value
    #
    # The output from np.where is a tuple of arrays (rows, columns)
    max_indices = list(zip(result[0], result[1]))  # Combine row and column indices
    #
    print('##############################################################')
    print("Maximum density value:", max_value)
    print("Indices of maximum density value (row, column):", max_indices)
    print('##############################################################')
    xa, xb, ya, yb = max_indices[0][0] - 14, max_indices[0][0] + 14, max_indices[0][1] - 13, max_indices[0][1] + 18
    print()
    kill_spots(u, xa, xb, ya, yb)
    u[u <= umax] = 0
    u[u > umax] = 1
    #
    # pcm = plt.imshow(u.T[:, :], norm=norm, cmap="gnuplot", origin="lower",
    #                  extent=np.concatenate((bounds, bounds)))
    # plt.show()
    # plt.close()
    print('##############################################################')
    print('############ THIRD PLOT::   CREATE THE CIRCLE################################')########################################
    print('##############################################################')

    # DEFINE THE BOUNDARY
    u_new, points = set_boundary(u)
    #
    from itertools import combinations
    from math import sqrt
    #
    #
    # Function to calculate Euclidean distance between two points
    def euclidean_distance(point1, point2):
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)





    # Initialize variables to store the maximum distance and the points that yield it
    max_distance = 0
    max_pair = None  # This will store the points that yield the maximum distance

    # Iterate over all combinations of pairs of points
    for point1, point2 in combinations(points, 2):
        # Calculate the distance between the pair of points
        dist = euclidean_distance(point1, point2)
        # Update the maximum distance and the points that yield it, if needed
        if dist > max_distance:
            max_distance = dist
            max_pair = (point1, point2)
    #
    print("Maximum distance between points (or B of Taylor):", max_distance)
    print("Points that yield the maximum distance:", max_pair)
    print('##############################################################')
    # #
    #
    # # #
    # pcm = plt.imshow(u_new.T[:, :], norm=norm, cmap="gnuplot", origin="lower",
    #                  extent=np.concatenate((bounds, bounds)))
    # plt.show()
    # plt.close()


    # Find the maximum distance pair among points
    max_dist = 0
    max_pair = None

    for p1, p2 in combinations(points, 2):
        dist = euclidean_distance(p1, p2)
        if dist > max_dist:
            max_dist = dist
            max_pair = (p1, p2)

    print(f"Maximum distance pair: {max_pair} with distance {max_dist}")

    # Calculate alpha and beta for the orthogonal line equation
    if max_pair:
        (x1, y1), (x2, y2) = max_pair
        alpha = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Avoid division by zero
        beta = y1 - alpha * x1



    # # Optional plotting of points and orthogonal line
    # plt.scatter(*zip(*points), label="Data Points")
    # if max_pair:
    #     plt.plot([x1, x2], [y1, y2], 'r-', label="Max Distance Pair")
    # plt.title('Points and Orthogonal Line')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend()
    # plt.show()
    # # #
    # # ################################################################################################################################################################################################################################################
    # # ######################################FOURTH PLOT::   FIND THE TAYLOR PARAMETERS########################################################################
    # # #######################################################################################################################3
    # #
    # pcm = plt.imshow(u_new.T[:, :], norm=norm, cmap="gnuplot", origin="lower",
    #                  extent=np.concatenate((bounds, bounds)))
    # plt.plot([max_pair[0][0], max_pair[1][0]], [max_pair[0][1], max_pair[1][1]], color='y', lw=1)
    # plt.plot(max_pair[0][0], max_pair[0][1], 'o', color='y')
    # plt.plot(max_pair[1][0], max_pair[1][1], 'o', color='y')
    # plt.show()
    # plt.close()
    #
    #
    # #
    # #
    # Function to calculate the slope of the line between two points
    def calculate_slope(x1, y1, x2, y2):
        # Handle the case of a vertical line (infinite slope)
        if x2 - x1 == 0:
            return None  # Represents infinite slope (vertical line)
        return (y2 - y1) / (x2 - x1)


    #
    #
    # Function to calculate the y-intercept given a point and a slope
    def calculate_y_intercept(x, y, slope):
        if slope is None:
            return None  # No y-intercept for vertical lines
        return y - slope * x


    #
    #
    # Function to return the general line equation given slope and y-intercept
    def line_equation(slope, y_intercept):
        if slope is None:
            return f"x = {y_intercept}"  # Equation for vertical line
        return f"y = {slope}x + {y_intercept}"


    #
    #
    # Function to get the orthogonal slope
    def orthogonal_slope(slope):
        if slope is None:
            return 0  # Orthogonal to vertical line is horizontal (slope=0)
        elif slope == 0:
            return None  # Orthogonal to horizontal line is vertical (infinite slope)
        else:
            return -1 / slope  # Negative reciprocal for orthogonal slope


    #
    #
    # Given two points, calculate the original line equation and the orthogonal line equation
    def line_and_orthogonal(x1, y1, x2, y2):
        # Calculate the original slope and y-intercept
        slope = calculate_slope(x1, y1, x2, y2)
        y_intercept = calculate_y_intercept(x1, y1, slope)
        #
        # Get the equation of the original line
        original_line = line_equation(slope, y_intercept)

        # Calculate the orthogonal slope and y-intercept
        orth_slope = orthogonal_slope(slope)

        # Get the equation of the orthogonal line (passing through the first point)
        orth_y_intercept = calculate_y_intercept(x1, y1, orth_slope)
        orthogonal_line = line_equation(orth_slope, orth_y_intercept)

        return original_line, orthogonal_line


    def values_orthogonal_line(x1, y1, x2, y2):
        # Calculate the original slope and y-intercept
        slope = calculate_slope(x1, y1, x2, y2)
        y_intercept = calculate_y_intercept(x1, y1, slope)
        # Calculate the orthogonal slope and y-intercept
        orth_slope = orthogonal_slope(slope)
        # Get the equation of the orthogonal line (passing through the first point)
        orth_y_intercept = calculate_y_intercept(x1, y1, orth_slope)

        return orth_slope, orth_y_intercept, slope, y_intercept


    # Example points
    maxBy = max(max_pair[0][1], max_pair[1][1])
    if maxBy == max_pair[0][1]:
        x1, y1 = max_pair[0][0], max_pair[0][1]
        x2, y2 = max_pair[1][0], max_pair[1][1]
    else:
        x2, y2 = max_pair[0][0], max_pair[0][1]
        x1, y1 = max_pair[1][0], max_pair[1][1]

    # Get the line equations
    original_line, orthogonal_line = line_and_orthogonal(x1, y1, x2, y2)

    alpha, beta, alpha2, beta2 = values_orthogonal_line(x1, y1, x2, y2)


    Dxb = abs(x1 - x2)
    Dyb = abs(y1 - y2)
    B = np.sqrt(Dxb ** 2 + Dyb ** 2)
    print('B taylor is =',B)

    print("Original line equation:", original_line)
    print("Orthogonal line equation:", orthogonal_line)
    print('#######################################################################')


    # Define a function to compute the maximum distance between found points
    def compute_max_distance(points):
        max_distance = 0
        max_pair = (None, None)
        for i, pt1 in enumerate(points):
            for j, pt2 in enumerate(points):
                if i >= j:
                    continue
                distance = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                if distance > max_distance:
                    max_distance = distance
                    max_pair = (pt1, pt2)
        return max_distance, max_pair


    print('##############################DEFINE ALL POINTS IN THE LINE#########################################')

    Dd = 0.001  # Initial definition of line
    step = 0.001  # Increment step for Dd
    min_points = 2  # Minimum number of points needed
    max_points = 3  # Maximum number of points to find
    found_points = []

    # Loop until at least two points are found and their x-coordinate distance is greater than 0.1
    while (len(found_points) < min_points or (len(found_points) >= min_points and abs(
            max(found_points, key=lambda p: p[0])[0] - min(found_points, key=lambda p: p[0])[0]) <= 0.1)) and len(found_points) <= max_points:
        found_points.clear()  # Clear the list of found points for each Dd iteration
        for pt in points:
            # Adjusted delta calculation using the orthogonal line equation
            delta = abs(pt[1] - alpha * (pt[0] + abs(max_pair[0][0] - max_pair[1][0]) / 2) - beta + abs(
                max_pair[0][1] - max_pair[1][1]) / 2)
            if delta <= Dd:
                found_points.append(pt)

        # Print points if at least two are found
        if len(found_points) >= min_points and abs(
                max(found_points, key=lambda p: p[0])[0] - min(found_points, key=lambda p: p[0])[0]) > 0.1:
            for pt in found_points:
                print("orthogonal line pt", pt)
        else:
            # Increment Dd if fewer than two points were found or x-coordinate distance is not sufficient
            Dd += step

    # Compute the maximum distance and the pair of points
    max_distance, max_pair = compute_max_distance(found_points)

    # # Optional plotting of points and orthogonal line
    # plt.figure(figsize=(8, 6))
    #
    # # Plot all points
    # plt.scatter(*zip(*points), label="All Data Points", color='blue')
    #
    # # Highlight points that were found
    # if found_points:
    #     plt.scatter(*zip(*found_points), label="Highlighted Points", color='red', s=100, edgecolor='black')
    #
    # # Plot the line between the max distance pair
    # if max_pair[0] and max_pair[1]:
    #     (x1, y1), (x2, y2) = max_pair
    #     plt.plot([x1, x2], [y1, y2], 'g--', label="Max Distance Pair", linewidth=2)  # Line in green with dashed style
    #
    # plt.xlim([-0.5, 0.5])
    # plt.ylim([-0.5, 0.5])
    #
    # plt.title('Points and Orthogonal Line')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    print('Best Dd: ',Dd)
    print('Distance between found points:' , max_distance)

    B_list.append(B)
    A_List.append(max_distance)
    pe.append(i*10)

print('AList',A_List)
print('BList', B_list)

#
#
print('####################################################################################################')
print('##############################BEGIN FIGURE SHEAR STRESS FIGURE######################################')




Delta = [abs(A - B) / (A + B) for A, B in zip(A_List, B_list)]
print(pe)
# # # Begin figure
fig = plt.figure(dpi = 900, figsize=(10, 5))
#
# # Create division
subfigs = fig.subfigures(1, 2, hspace=0.0, wspace=0, width_ratios=[1, 0.9])
# #
# # Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
# #
#
#
print('Left figure: density distribution. 4 rows x 1 column')
axL = subfigs[0].subplots(4, 1, sharey=False,sharex= True)
subfigs[0].subplots_adjust(hspace=0.5)
# subfigs[0].set_facecolor('ghostwhite')
#
umax = np.max(ufp(250, pe=0))
print('Normalization: umax = 250 mu, 0 pe')
plotleft_figure()  #Call figure
print('row 1: mu =800, pe = 10')
print(f'row 2: mu =800, pe = 100')
print(f'row 3: mu = 800, pe =230')
print(f'row 4: mu = 800, pe =240')


print('##########################################')
print('Right figure: Taylor deformation parameter and Linear Regression')
axR = subfigs[1].subplots(1, 1, sharey=True)
plotright_figure()


plt.savefig('FigTaylor.png', bbox_inches="tight")  #
plt.savefig('FigTaylor.pdf', bbox_inches="tight")  #