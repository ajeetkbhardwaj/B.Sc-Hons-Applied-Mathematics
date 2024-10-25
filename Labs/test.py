# Test

#### Python : Numpy + Matplotlib
import numpy as np
import matplotlib.pyplot as plt

arr1=np.array([1,2,3,4,5,6])
print(arr1)

arr2=np.array([[1,2,3],[4,5,6]])
print(arr2)

arr=np.arange(24)
print(arr)
arr=arr.reshape((6,4))
print(arr)

arr[1:4,2:4]
arr.flatten()
arr.mean()

arr=np.arange(25)

np.array_split(arr,3)


xpoints=np.array([11,15,16,19,13,19])
ypoints=np.array([0,2,5,0,5,3])

plt.plot(ypoints,'*-.b',ms=20,mec='#4CAF50',mfc='g',linewidth='10') #marker | \line | color \types of lines: -, :, --, -. #MEC: Marker Edge Colour #MFC: Marker Face Colour

carA=np.array([2,4,6,4,2,7])
carB=np.array([5,3,4,6,7,2])

plt.plot(carA)
plt.plot(carB)
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.title("Sports Watch Data")
plt.grid(axis='y')
plt.show()

#%%

x=np.array([0,1,2,3])
y=np.array([3,8,1,10])
z=np.array([10,20,30,40])


plt.subplot(3,2,1)
plt.plot(x,y)

plt.subplot(3,2,2)
plt.plot(x,z)

plt.subplot(3,2,3)
plt.plot(x,y)

plt.subplot(3,2,4)
plt.plot(x,y)

plt.subplot(3,2,5)
plt.plot(x,y)

plt.subplot(3,2,6)
plt.plot(x,y)

plt.show()

x=np.array([0,1,2,3,4,5,6,7,8,9])
y=np.array([4,5,6,7,2,3,4,6,5,2])
z=np.array([6,4,3,6,7,4,3,6,7,8])


plt.scatter(x,y,color='r')
plt.scatter(x,z,color='g')
plt.show()


x=np.array([0,1,2,3])
y=np.array([4,5,6,7,])
color=np.array(['red','green','yellow','blue'])
plt.scatter(x,y,c=color)
plt.show()


#BarGraph
x=np.array(["A","B","C","D"])
y=np.array([3,8,1,10])
plt.bar(x,y,color='hotpink',width=0.1)
plt.title('Classes')
plt.ylabel('count')
plt.xlabel('sections')
plt.show()

#Histogram
x=np.array([0,1,2,3])
y=np.array([3,8,1,10])
plt.hist(x,y)
plt.show()

x=np.random.normal(170,10,250)
plt.hist(x)
plt.show()

y=np.array([35,25,25,15])
plt.pie(y)
plt.show()

# Graph Theory
#%%


def is_regular(graph):
    degree = len(graph[0])
    for node in graph:
        if len(node) != degree:
            return False
    return True
"""
graph = [
    [1, 2],
    [0, 3],
    [0, 3],
    [1, 2]
]
"""
graph = [
    [1],
    [0, 2, 3],
    [1, 3],
    [1, 2]
]
if is_regular(graph):
    print("Graph is regular")
else:
    print("Graph is irregular")
    
    
  
#%%
  def deg_sequence(graph):
    degree = [len(neighbors) for neighbors in graph]
    return sorted(degree, reverse = True)

graph1 = [
    [1, 2],
    [0, 3],
    [0, 3],
    [1, 2]
]

graph2 = [
    [1],
    [0, 2, 3],
    [1, 3],
    [1, 2]
]

graph3 = [
    [1, 2, 3],
    [0, 2],
    [0, 1, 3],
    [0, 2]
]
print("Graph1:", deg_sequence(graph1))
print("Graph2:", deg_sequence(graph2))
print("Graph3:", deg_sequence(graph3))

 
 
 
 #%%
# Function to check if a graph is planar based on Euler's formula (approximation)
def is_planar(graph):
    V = len(graph)
    E = sum(len(neighbors) for neighbors in graph) // 2
    if V < 3:
        return True


    if E <= 3 * V - 6:
        return True
    else:
        return False
graph = [
    [1, 2],  # Node 0 is connected to Node 1 and Node 2
    [0, 3],  # Node 1 is connected to Node 0 and Node 3
    [0, 3],  # Node 2 is connected to Node 0 and Node 3
    [1, 2]   # Node 3 is connected to Node 1 and Node 2
]


print("Graph is planar:", is_planar(graph))


# Non-planar graph K5 (Complete graph on 5 vertices)
K5_graph = [
    [1, 2, 3, 4],  # Node 0 is connected to Node 1, 2, 3, 4
    [0, 2, 3, 4],  # Node 1 is connected to Node 0, 2, 3, 4
    [0, 1, 3, 4],  # Node 2 is connected to Node 0, 1, 3, 4
    [0, 1, 2, 4],  # Node 3 is connected to Node 0, 1, 2, 4
    [0, 1, 2, 3]   # Node 4 is connected to Node 0, 1, 2, 3
]

# Function to check if a graph is planar based on Euler's formula (approximation)
def is_planar(graph):
    V = len(graph)
    E = sum(len(neighbors) for neighbors in graph) // 2

    if V < 3:
        return True


    if E <= 3 * V - 6:
        return True


    else:
        return False


print("K5 graph is planar:", is_planar(K5_graph))
#%%
# Function to check if the graph has an Eulerian path or circuit
def has_eulerian_path(graph):

    odd_count = sum(1 for neighbors in graph.values() if len(neighbors) % 2 != 0)


    return odd_count == 0 or odd_count == 2


graph = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1, 3],
    3: [2]
}

print("Graph has Eulerian path or circuit:", has_eulerian_path(graph))



def create_adjacency_matrix(V, edges):
    # Initialize an empty V x V matrix with all zeros
    matrix = [[0] * V for _ in range(V)]


    for edge in edges:
        u, v = edge
        matrix[u][v] = 1
        matrix[v][u] = 1

    return matrix


# Example 1
V1 = 3
edges1 = [(0, 1), (1, 2), (2, 0)]
adj_matrix1 = create_adjacency_matrix(V1, edges1)
for row in adj_matrix1:
    print(row)
print()

# Example 2
V2 = 4
edges2 = [(0, 1), (1, 2), (1, 3), (2, 3), (3, 0)]
adj_matrix2 = create_adjacency_matrix(V2, edges2)
for row in adj_matrix2:
    print(row)

#%%
def compute_degrees(graph):
    # Initialize in-degree and out-degree dictionaries
    in_degree = {node: 0 for node in range(len(graph))}
    out_degree = {node: 0 for node in range(len(graph))}

    # Calculate out-degrees and in-degrees
    for node, neighbors in enumerate(graph):
        out_degree[node] = len(neighbors)
        for neighbor in neighbors:
            in_degree[neighbor] += 1

    return in_degree, out_degree

directed_graph = [
    [1, 2],  # Node 0 points to Node 1 and Node 2
    [2],     # Node 1 points to Node 2
    [0],     # Node 2 points to Node 0
    [1, 2]   # Node 3 points to Node 1 and Node 2
]

# Compute in-degrees and out-degrees
in_degree, out_degree = compute_degrees(directed_graph)

# Print the results
print("In-Degree:", in_degree)
print("Out-Degree:", out_degree)

 
""" 
Question-1 : Given a list of vertices, create a graph.

"""

import itertools
def Create_Graph(Vertices):
    # create list of edges of graph, given the vertices
    Edges = list(itertools.combinations(Vertices, 2))
    return Edges

# Vertices
V = [9, 3, 4, 5, 2, 8]
G = Create_Graph(V)
print(G)

#%%
""" 
Question-2 : Using networkx library create and visualize the graph

"""
import itertools
import networkx as nx
import matplotlib.pyplot as plt

# Initialize a graph

def Create_Graph(Vertices, Edges):
    G = nx.Graph()
    G.add_nodes_from(Vertices)
    G.add_edges_from(Edges)
    return G

def Visualize_Graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=100)
    nx.draw_networkx_edges(G, pos, width=0.4)
    nx.draw_networkx_labels(G, pos, font_size=6)
    plt.show()
V = [1, 2, 3, 4]  
E = [(2, 1), (3, 2), (4, 3), (1, 4)]
G = Create_Graph(V, E) 
Visualize_Graph(G)
#


    
# %%
""" 
Question-3 : Write a function which check a given graph is bipartite or not ?

"""

def Is_Bipartite(G):
    G = nx.is_bipartite(G)
    if G==True:
        print("G is bipartite graph")
    else:
        print("G is not bipartite graph")
        
G = nx.cycle_graph(5)
H = nx.complete_graph(5)
I = nx.star_graph(6)
J = nx.wheel_graph(8)

def Is_Cyclic(G):
    
    
    
#%%

import itertools

def generate_edges(vertices):
    """Generate all possible edges from the list of vertices."""
    return list(itertools.combinations(vertices, 2))

def generate_graphs(vertices):
    """Generate all simple graphs from the given vertices."""
    edges = generate_edges(vertices)
    all_graphs = []
    
    # Generate all combinations of edges (including empty set)
    for i in range(len(edges) + 1):
        for edge_subset in itertools.combinations(edges, i):
            all_graphs.append(list(edge_subset))
    
    return all_graphs

def display_graph(graph_edges, vertices):
    """Display the graph in an adjacency list format."""
    adjacency_list = {vertex: [] for vertex in vertices}
    
    for edge in graph_edges:
        u, v = edge
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    
    print("Adjacency List Representation:")
    for vertex, neighbors in adjacency_list.items():
        print(f"{vertex}: {', '.join(neighbors) if neighbors else 'No neighbors'}")

# Example usage
vertices = ['A', 'B', 'C']
all_graphs = generate_graphs(vertices)

# Display each graph
for i, graph in enumerate(all_graphs):
    print(f"\nGraph {i + 1}: Edges = {graph}")
    display_graph(graph, vertices)
# %%

def create_adjacency_matrix(edges):
    """Creates an adjacency matrix from a list of edges."""
    
    # Step 1: Identify unique vertices
    vertices = set()
    for u, v in edges:
        vertices.add(u)
        vertices.add(v)
    
    # Convert set to a sorted list for consistent ordering
    vertices = sorted(vertices)
    
    # Step 2: Initialize the adjacency matrix
    n = len(vertices)
    adjacency_matrix = [[0] * n for _ in range(n)]
    
    # Step 3: Fill in the adjacency matrix
    for u, v in edges:
        # Find indices of u and v
        index_u = vertices.index(u)
        index_v = vertices.index(v)
        
        # Mark the presence of an edge
        adjacency_matrix[index_u][index_v] = 1
        adjacency_matrix[index_v][index_u] = 1  # For undirected graph
    
    return vertices, adjacency_matrix

def display_adjacency_matrix(vertices, adjacency_matrix):
    """Displays the adjacency matrix."""
    print("Adjacency Matrix:")
    print("  ", " ".join(vertices))
    
    for i, row in enumerate(adjacency_matrix):
        print(vertices[i], " ".join(map(str, row)))

# Example usage
edges = [('A', 'B'), ('A', 'C'), ('B', 'C')]
vertices, adjacency_matrix = create_adjacency_matrix(edges)

# Display the result
display_adjacency_matrix(vertices, adjacency_matrix)

# %%


def bisection_method(f, a, b, tol=1e-7, max_iter=1000):
    """Finds the root of f(x) using the Bisection Method."""
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    
    for iteration in range(max_iter):
        c = (a + b) / 2  # Midpoint
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c  # Root found
        
        if f(c) * f(a) < 0:
            b = c  # The root is in the left subinterval
        else:
            a = c  # The root is in the right subinterval
    
    raise ValueError("Maximum iterations reached without convergence.")


def regular_falsi_method(f, a, b, tol=1e-7, max_iter=1000):
    """Finds the root of f(x) using the Regular Falsi Method."""
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    
    for iteration in range(max_iter):
        # Calculate the point using linear interpolation
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        
        if abs(f(c)) < tol or abs(b - a) < tol:
            return c  # Root found
        
        # Update interval
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    
    raise ValueError("Maximum iterations reached without convergence.")


# Example usage
if __name__ == "__main__":
    # Define a function for which we want to find the root
    def func(x):
        return x**3 - x - 2  # Example: x^3 - x - 2 = 0

    # Define interval [a, b]
    a = 1
    b = 2

    # Find root using Bisection Method
    try:
        root_bisection = bisection_method(func, a, b)
        print(f"Root found using Bisection Method: {root_bisection}")
    except ValueError as e:
        print(e)

    # Find root using Regular Falsi Method
    try:
        root_falsi = regular_falsi_method(func, a, b)
        print(f"Root found using Regular Falsi Method: {root_falsi}")
    except ValueError as e:
        print(e)
# %%

def simpsons_1_3(f, a, b, n):
    """Approximate the integral of f from a to b using Simpson's 1/3 rule."""
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's 1/3 rule.")
    
    h = (b - a) / n
    integral = f(a) + f(b)

    for i in range(1, n, 2):
        integral += 4 * f(a + i * h)

    for i in range(2, n-1, 2):
        integral += 2 * f(a + i * h)

    integral *= h / 3
    return integral

def simpsons_3_8(f, a, b, n):
    """Approximate the integral of f from a to b using Simpson's 3/8 rule."""
    if n % 3 != 0:
        raise ValueError("n must be a multiple of 3 for Simpson's 3/8 rule.")
    
    h = (b - a) / n
    integral = f(a) + f(b)

    for i in range(1, n):
        if i % 3 == 0:
            integral += 2 * f(a + i * h)
        else:
            integral += 3 * f(a + i * h)

    integral *= (3 * h) / 8
    return integral

# Example usage
if __name__ == "__main__":
    # Define a function to integrate
    def func(x):
        return x**2  # Example function: f(x) = x^2

    # Define interval [a, b]
    a = 0
    b = 1

    # Number of intervals (must be even for Simpson's 1/3 and multiple of 3 for Simpson's 3/8)
    n_1_3 = 4
    n_3_8 = 6

    # Calculate integrals
    try:
        result_1_3 = simpsons_1_3(func, a, b, n_1_3)
        print(f"Integral using Simpson's 1/3 Rule: {result_1_3}")
        
        result_3_8 = simpsons_3_8(func, a, b, n_3_8)
        print(f"Integral using Simpson's 3/8 Rule: {result_3_8}")
        
    except ValueError as e:
        print(e)
# %%
# trapezoidal integration
import numpy as np

def f(x):
   return x*np.sin(x)
# Left and Right limits
a=0
b=np.pi/2
# Number of trapezoids
n=6
# Width of trapezoid
h=(b-a)/n
# array of x
# For n trapezoids (no. of nodes=n+1)
x=np.linspace(a,b,n+1)
# First term of I
I=(f(a)+f(b))
# Summing over remaining n-2 trapezoids
for j in range(1,n):
   if j%3==0:
      I=I+2*f(x[j])
   else:
      I=I+3*f(x[j])
I=(h/8)*I*3
print(f'I = {round(I,6)}')

#%%
# Newton Rapson
def f(x):
    return x**2 - 5*x + 5
def Df(x):
    return 2*x - 5
count = 1
def NewtonMethod(x):
    h = f(x)/Df(x)
    while(np.abs(h)>=0.1):
        h = f(x)/Df(x)
        x = x - h
        
        print(x)
    print('the root value is :', x)

#%%

# Simpson 1_3
def f(x):
    return x*np.sin(x)

a = 2
b = 9
n = 100
h = np.abs(b-a)/n
x = np.linspace(a, b, n+1)

def Simpson(x):
    f_0 = f(x[0])
    f_n = f(x[n])
    even = 0
    odd = 0
    for i in range(1, n):
        if(i%2==0):
            even = even+2*f(x[i])
        else:
            odd = odd + 4*f(x[i])

    I = h/3 * (f_0 + f_n + even + odd)
    print(I)
        
Simpson(x)

#%% Simson 3/8

def Simpson(x):
    f_0 = f(x[0])
    f_n = f(x[n])
    even = 0
    odd = 0
    for i in range(1, n):
        if(i%3==0):
            even = even+2*f(x[i])
        else:
            odd = odd + 4*f(x[i])

    I = 3*h/8 * (f_0 + f_n + even + odd)
    print(I)