import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import sys
matplotlib.use('TkAgg')

class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value

class Network:

	def __init__(self, nodes=None):
		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes

	def get_mean_degree(self):
		if not self.nodes: # If there are no nodes in the network
			return 0  # Return mean degree as 0
		
        	# Calculate the degree of each node
		degrees = [sum(1 for conn in node.connections if conn == 1) for node in self.nodes]
		mean_degree = np.mean(degrees) # Calculate the mean degree
		return mean_degree


	def get_mean_clustering(self):
		clustering_coeffs = []
		
        	# Iterate through each node in the network
		for node in self.nodes:
			neighbors = [i for i, connected in enumerate(node.connections) if connected]
			if len(neighbors) < 2: # If the node has less than 2 neighbors
				clustering_coeffs.append(0)  # Append 0 to clustering coefficients list

           		 # Calculate the possible triangles for the node
			possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2
			actual_triangles = 0
			# Count the actual triangles
			for i in range(len(neighbors)):
				for j in range(i + 1, len(neighbors)):
					if self.nodes[neighbors[i]].connections[neighbors[j]]:
						actual_triangles += 1

            		# Calculate clustering coefficient
			clustering_coeffs.append(actual_triangles / possible_triangles)
			
        	# Calculate the mean clustering coefficient
		mean_clustering_coefficient = np.mean(clustering_coeffs)
		return mean_clustering_coefficient

    	# calculate the mean path length by using breadth-first-search
	def get_mean_path_length(self):
		num_nodes = len(self.nodes)
		path_lengths = []

        	# Iterate through each node and calculate shortest paths
		for start in range(num_nodes):
			visited = [False] * num_nodes
			distances = [0] * num_nodes
			queue = [start]

			visited[start] = True

            		# Breadth-first search to calculate shortest paths
			while queue:
				current = queue.pop(0)
				for neighbour_index, connected in enumerate(self.nodes[current].connections):
					if connected and not visited[neighbour_index]:
						visited[neighbour_index] = True
						distances[neighbour_index] = distances[current] + 1
						queue.append(neighbour_index)
			path_lengths.extend([dist for dist in distances if dist > 0])
			
		if not path_lengths: # If there are no valid path lengths calculated
			return float('inf')# Return infinity
		
		# Calculate the mean path length
		mean_path_length = sum(path_lengths) / (num_nodes * (num_nodes - 1))
		return round(mean_path_length, 15) # Round the mean path length to 15 decimal places

	def make_random_network(self, N, connection_probability=0.5):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	def make_ring_network(self, N, neighbour_range=1):
		self.nodes = []
		for node_number in range(N):
			connections = [0 for _ in range(N)]
			for i in range(-neighbour_range, neighbour_range + 1):
				if i != 0:  # Avoid self-connection
					neighbour_index = (node_number + i) % N
					connections[neighbour_index] = 1
			# When initializing Node, provide a default value for `value` as well
			new_node = Node(value=np.random.random(), number=node_number, connections=connections)
			self.nodes.append(new_node)

	def make_small_world_network(self, N, re_wire_prob=0.1, neighbour_range=2):
		# Create a ring network with connections up to 'neighbour_range' distance
		self.nodes = [Node(value=np.random.random(), number=i) for i in range(N)]
		for node in self.nodes:
			node.connections = [0] * N
			for offset in range(1, neighbour_range + 1):
				right = (node.index + offset) % N
				left = (node.index - offset) % N
				node.connections[right] = 1
				node.connections[left] = 1

		# Rewire connections with given probability 're_wire_prob'
		for node in self.nodes:
			for neighbour in range(1, neighbour_range + 1):
				right = (node.index + neighbour) % N
				if np.random.random() < re_wire_prob:
					# Choose a node to rewire to that isn't already a neighbour and isn't self
					potential_targets = [i for i in range(N)
										 if i != node.index
										 and self.nodes[node.index].connections[i] == 0]
					if potential_targets:
						new_target = np.random.choice(potential_targets)
						# Disconnect the original connection and make the new one
						node.connections[right] = 0
						self.nodes[right].connections[node.index] = 0
						node.connections[new_target] = 1
						self.nodes[new_target].connections[node.index] = 1

	def plot(self):

			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.set_axis_off()

			num_nodes = len(self.nodes)
			network_radius = num_nodes * 10
			ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
			ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

			for (i, node) in enumerate(self.nodes):
				node_angle = i * 2 * np.pi / num_nodes
				node_x = network_radius * np.cos(node_angle)
				node_y = network_radius * np.sin(node_angle)

				circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
				ax.add_patch(circle)

				for neighbour_index in range(i+1, num_nodes):
					if node.connections[neighbour_index]:
						neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
						neighbour_x = network_radius * np.cos(neighbour_angle)
						neighbour_y = network_radius * np.sin(neighbour_angle)

						ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')


def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_mean_clustering()==0), network.get_mean_clustering()
	assert(network.get_mean_path_length()==2.777777777777778), network.get_mean_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_mean_clustering()==0),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==5), network.get_mean_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_mean_clustering()==1),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==1), network.get_mean_path_length()

	print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0, alpha=1.0):
    #  Calculate the agreement at a given position.
    """
    population: the current state of the Ising model.
    row: the row index of the position.
    col: the column index of the position.
    external: the magnitude of any external "pull" on opinion.
    alpha: system's tolerance for disagreement.
    """

    n_rows, n_cols = population.shape
    sum_neighbors = 0

    # Neighbors' coordinates (Up, Right, Down, Left)
    neighbors = [(row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1)]
    for x, y in neighbors:
        if 0 <= x < n_rows and 0 <= y < n_cols:
            sum_neighbors += population[x, y]
    # Agreement considers the external influence
    agreement = sum_neighbors * population[row, col] + external * population[row, col]

    return agreement
    # The agreement value at the given position


def ising_step(population, alpha=1.0, external=0.0):
    #  Single update of the Ising model.

    """
    This function will perform a single update of the Ising model.
    Inputs: population (numpy array)
            alpha (float) - system's tolerance for disagreement
            external (float) - the magnitude of any external "pull" on opinion
    """
    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, alpha, external)

    prob_flip = np.exp(-agreement) / alpha if agreement > 0 else 1

    if np.random.random() < prob_flip or agreement < 0:
        population[row, col] *= -1

    return population


def plot_ising(im, population):
    # Plot the Ising model.

    """
    im: matplotlib image object.
    population: the current state of the Ising model.
    """
    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)


def test_ising():
    # Test calculations.

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1,1)==3), "Test 7"
    assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
    assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
    assert(calculate_agreement(population,1,1, -10)==14), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    # Main function for the Ising model.

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

def defuant_model(opinions, threshold, beta,iterations):
    # Implementation of the Defuant model.
    """
    opinions: initial opinions.
    threshold: threshold for interaction.
    beta: updating rate.
    iterations: number of iterations.
    """
    opinions_over_time = [[] for i in range(len(opinions))]
    for t in range(iterations):
        i = np.random.randint(len(opinions))
        j = (i + 1) % len(opinions) if np.random.rand() > 0.5 else (i - 1) % len(opinions)

        if abs(opinions[i] - opinions[j]) < threshold:
            bias = beta* (opinions[j] - opinions[i])
            opinions[i] += bias
            opinions[j] -= bias
        for i in range(len(opinions)):
            opinions_over_time[i].append(opinions[i])
    return opinions_over_time
    # list of lists, opinions over time

def run_defuant(beta, threshold, population_size, iterations, testing=False):
    """
    beta: updating rate.
    threshold: threshold for interaction.
    population_size: number of agents.
    iterations: number of iterations.
    testing: flag to plot opinions over time if True.
    """
    initial_population = np.random.rand(population_size)
    opinions_over_time = defuant_model(initial_population, threshold, beta,iterations)
    # Plot the final population distribution
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    opinions_transposed = list(zip(*opinions_over_time))
    plt.hist(opinions_transposed[-1], bins=np.linspace(0, 1, 20), color='blue', edgecolor='black')
    plt.title('Final Population Distribution')
    plt.xlabel('Opinion')
    plt.ylabel('Frequency')

    # If testing flag is set, plot the opinions over time
    # if testing:
    plt.subplot(1, 2, 2)
    plt.title('Opinions Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Opinion')
    for opinions in opinions_over_time:
        plt.plot(opinions, 'o', markersize=3, label=f'Person {opinions}')
    plt.tight_layout()
    plt.show()

'''
==============================================================================================================
This section contains code for the Ising and Defuant Model modified with network model - task 5 in the assignment
==============================================================================================================
'''

# Plot a net
def NetPlot(fig, ax, net, current_frame):
    ax.clear()
    ax.set_title(f'Frame {current_frame}')
    num_nodes = len(net.nodes)
    network_radius = num_nodes * 10
    ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
    ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

    for (i, node) in enumerate(net.nodes):
        node_angle = i * 2 * np.pi / num_nodes
        node_x = network_radius * np.cos(node_angle)
        node_y = network_radius * np.sin(node_angle)

        circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
        ax.add_patch(circle)

        for neighbour_index in range(i + 1, num_nodes):
            if node.connections[neighbour_index]:
                neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                neighbour_x = network_radius * np.cos(neighbour_angle)
                neighbour_y = network_radius * np.sin(neighbour_angle)

                ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

# Display the model in an animation graph
current_frame = 0
def animationNet(nets_seq):
    global current_frame
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    def update(frame):
        global current_frame
        current_frame += 1
        current_frame = min(current_frame, len(nets_seq)-1)
        NetPlot(fig, ax, nets_seq[current_frame], current_frame)
    # Draw the animation
    anim = FuncAnimation(fig, update, 
                         frames=len(nets_seq), repeat=False,
                         interval=500)
    anim.save("animation.gif")
    plt.show()
    current_frame = 0

def ising_network(population, alpha=None, external=0.0):
    # Build a small-world model first
    net = Network()
    net.make_small_world_network(population, re_wire_prob=0.2)
    
    # Traverse each node and calculate the agreement to decide the final value
    nodes = net.nodes
    net_seqs = [net]
    for node in nodes:
        # Compute the agreement first by D_i=Sum(S_i*S_j)+H*S_i
        agreement = external*node.value
        for k in range(len(node.connections)):
            if node.connections[k] == 1:
                agreement += node.value*nodes[k].value
        # Flip the value if need
        prob_flip = np.exp(-agreement) / alpha if agreement > 0 else 1
        if np.random.random() < prob_flip or agreement < 0:
            node.value *= -1
        # Add the current net
        net_seqs.append(net)
    
    # Display the processing
    animationNet(net_seqs)
    
    # Return the network
    return net
    
def defuant_network(population, threshold, beta, iterations):
    # Build a small-world model first
    net = Network()
    net.make_small_world_network(population, re_wire_prob=0.2)
    
    # Traverse each node and calculate the agreement to decide the final value
    nodes = net.nodes
    net_seqs = [net]
    for t in range(iterations):
        i = np.random.randint(len(nodes))
        j = (i + 1) % len(nodes) if np.random.rand() > 0.5 else (i - 1) % len(nodes)

        if abs(nodes[i].value - nodes[j].value) < threshold:
            bias = beta * (nodes[j].value - nodes[i].value)
            nodes[i].value += bias
            nodes[j].value -= bias
        
        net_seqs.append(net)
        
    # Display the processing
    animationNet(net_seqs)
    
    # Return the network
    return net

'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
	#You should write some code for handling flags here
	global testt

	# task 5 has been added into both task 1 and task 2
	
	# task 1:
	H = 0.0
	alpha = 1.0
	grid_size = 10
	if "-ising_model" in sys.argv:
		if "-external" in sys.argv:
			external_index = sys.argv.index("-external") + 1
			H = float(sys.argv[external_index])

		if "-alpha" in sys.argv:
			alpha_index = sys.argv.index("-alpha") + 1
			alpha = float(sys.argv[alpha_index])
		population = np.random.choice([-1, 1], size=(grid_size, grid_size))
		# Using network or gird
        	if "-use_network" in sys.argv:
            		# Get the given size
            		pop_size = int(sys.argv[-1])
            		ising_network(pop_size, alpha, H)
        	else:
            		ising_main(population, alpha, H)
	elif "-test_ising" in sys.argv:
		test_ising()
	# task 2
	beta = 0.2
	threshold = 0.2
	testing = False
	testt = 0
	if "-defuant" in sys.argv:
		testt = 1
		if "-beta" in sys.argv:
			beta_index = sys.argv.index("-beta") + 1
			beta = float(sys.argv[beta_index])
		if "-threshold" in sys.argv:
			threshold_index = sys.argv.index("-threshold") + 1
			threshold = float(sys.argv[threshold_index])
		if "-use_network" in sys.argv:
            		# Get the given size
            		pop_size = int(sys.argv[-1])
            		defuant_network(pop_size, threshold, beta, iterations=100)
            		# Do not test the original defuant model
            		testing = 0

	elif "-test_defuant" in sys.argv:
		testing = 1
	if testing == True or testt == True:
		run_defuant(beta=beta, threshold=threshold, population_size=100, iterations=10000, testing=testing)
	
	# task 3
	if "-network" in sys.argv: # check if '-network' flag is present in command line arguments
		try:
			index = sys.argv.index("-network") + 1 # find the index of 'number of nodes'
			nodes = int(sys.argv[index]) # number of nodes
			conn_arg = sys.argv[index + 1] if index + 1 < len(sys.argv) else None
			try:
				conn = float(conn_arg) # connection probability
			except(ValueError, TypeError):
				conn = 0.5 # connection probability default
			
			network = Network() 
			network.make_random_network(N=nodes, connection_probability=conn)# create a random network
			network.plot() # plot it
			plt.show # show the plot
			
			print(f"Mean degree: {network.get_mean_degree()}")
			print(f"Average path length: {network.get_mean_path_length()}")
			print(f"Clustering coefficient: {network.get_mean_clustering()}")
			
		except (IndexError, ValueError):
			print('Please enter an integer (number of nodes) after "-network"')
			
	elif "-test_network" in sys.argv: # check if '-test_network' flag is present in command line arguments
		test_networks() # call the test function to test networks
	else:
		# print usage instructions if no valid flags are provided
		print("Usage for task3:")
		print("python FCP_assignment_v3.py -network <N> <probability>")
		print("python FCP_assignment_v3.py -test_network")
	
	#task 4
	if "-ring_network" in sys.argv:
		index = sys.argv.index("-ring_network") + 1
		N = int(sys.argv[index])
		network = Network()
		network.make_ring_network(N, neighbour_range=1)  # Set range as 1
		network.plot()
	elif "-small_world" in sys.argv:
		index = sys.argv.index("-small_world") + 1
		N = int(sys.argv[index])
		re_wire_prob = 0.2  # default re-wiring probability
		if "-re_wire" in sys.argv:
			re_wire_index = sys.argv.index("-re_wire") + 1
			re_wire_prob = float(sys.argv[re_wire_index])
		network = Network()
		network.make_small_world_network(N, re_wire_prob)
		network.plot()
	else:
		print("Usage for task4:")
		print("python FCP_assignment_v3.py -ring_network <N>")
		print("python FCP_assignment_v3.py -small_world <N> [-re_wire <probability>]")
	plt.show()

if __name__=="__main__":
	main()
