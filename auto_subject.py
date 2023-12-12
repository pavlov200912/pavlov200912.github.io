html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Bachelor's Program Curriculum</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
</head>
<body class="container py-5">
    <h1 class="text-center mb-4">Saint Petersburg State University, Modern Programming educational program</h1>
    <h2>Curriculum</h2>
    <p class="lead">This page provides an accessible, unofficial interpretation of the <a href="https://english.spbu.ru/admission/programms/undergraduate/modern-programming">Saint Petersburg State University's Modern Programming</a> undergraduate curriculum.</p>
"""

# Subjects from the Excel table
subjects = {
    "Fundamentals of Programming I": "Pointers. Static, heap and stack memory. Structures. Function pointers. Overview of libc: stdio. const. Overview of libc. C++. OOP. Encapsulation in C++. const, mutable, static, inline. Classes and new/delete. Inheritance. Operator overloading. Smart pointers. Polymorphism. Static and dynamic binding (virtual functions). Namespace. Input-output in C++. Inheritance.",
    
    "Fundamentals of Programming II": "Operator overloading. Design patterns: iterator, observer, visitor, adapter, builder. Computer arithmetic. Multithreading: race condition, resource sharing, message passing. Programming paradigms: imperative and declarative programming, procedural and functional programming. ",
    "Mathematical Analysis I": "Axiomatics of real numbers. Metric, topological and normed spaces. Limits and continuity, compactness. Theorems of Weierstrass and Cantor. Differential calculus of functions of one real variable (Roll's, Fermat's, Cauchy's and Lagrange's theorems, Taylor's formula). Integration of functions of one real variable (antiderivative and definite integral, existence and properties).",
    "Mathematical Analysis II": "Applications of the definite integral: area, volume, length, numerical integration, trapezoid formula, Stirling formula. Improper integrals. Analysis in normed spaces: contraction mapping principle, differentiability, implicit function theorem. Optimization problems: distance to a subspace, minimum of a quadratic form on a sphere, isoperimetric problem, brachistochrone problem, catenary problem. Theory of numerical and functional series.",
        "Mathematical Analysis III": "Measure theory: set structures, volume and measure, product of measures, standard continuation of measures, Lebesgue measure, measurable functions, various types of convergence. Lebesgue integral: construction of the Lebesgue integral and its properties, permutation of limit and integral, Cavalieri's principle, connection between multiple and iterated integrals, change of variable in a multiple integral. Integrals with a parameter and curvilinear integrals: uniform convergence, differentiation with respect to a parameter, B and Γ- functions, curvilinear integrals,differential forms of the first order, closed and exact forms, homotopic path integrals, Green's formula.",
    "Algebra I": "Algebraic structures: groups, rings, fields. Complex numbers: trigonometric form, De Moivre's formula,extraction of roots. Polynomials and rational fractions: Euclidean rings,decomposition into irreducible, multiple roots, decomposition ofa fraction into a sum of simplest ones, interpolation. Vector spaces and linear systems: linear dependence, matrixrank, Gauss method, Kronecker-Capelli theorem, linear mappings, kernel and image dimensions, sums and intersections of subspaces. Determinants: properties, minor rank, reciprocal matrix, Cramer's theorem.",
    "Algebra II": "Group theory: subgroups, homomorphisms, factor groups, symmetric groups, free group. Linear operators: eigenvalues, characteristic polynomial, canonical forms. Quadratic forms: orthogonalization, law of inertia. Analytic geometry: Euclidean spaces, orthogonal complement, canonical form of the normal operator. Finite fields: classification, cyclicity of the multiplicative group.",
    "Algebra III": """Bilinear forms, Gram matrix, Hermitian form. Euclidean and unitary spaces, Cauchy-Schwarz inequality, orthonormal basis, Gram–Schmidt process. Operator algebra. Eigenvalues and eigenvectors, diagonalizable operators. Cayley–Hamilton theorem. Jordan normal form.  Dual spaces and linear maps. Tensor algebra.""",
    "Discrete Mathematics I": "Evidence: existence, optimality, universality. Sets: cardinalities, diagonal argument, orders, chains/antichains. Enumeration and counting: subsets, permutations, placements, combinations, bracket sequences, splits, recurrence relations. Probability: events, conditional probability and independence, random variables, deviations, probabilistic method.",
    "Discrete Mathematics II": "graphs: trees, Euler and Hamiltonian cycles, connectivity, cuts and flows, matchings and coverings, coloring pages, planarity. Games and strategies. Codes with minimal redundancy and codes that correct errors. Calculation models and complexity classes.",
    "Discrete Mathematics III": "Complexity and hierarchies in time and memory. Reducibility and complete problems. P, NP, polynomial hierarchy, PSPACE. Polynomial schemes and parallel algorithms. Probabilistic calculations.",
    "C++": "templates: basic syntax, template variables,  specialization. Exceptions, RAII, guarantees. Type casting xxx_cast. RTTI . Sequential containers. Iterators . Associative containers. Algorithms . C++11: move, auto, lambda . Metaprogramming. Variadic templates, SFINAE, enable_if. STL in C++11. C++14/C++17: std::filesystem, std::variant, std::any, std:optional.",
    
    
    "Algorithms I": "Complexity of algorithms: O-symbols, recurrent relations. Basic algorithms and data structures: list, dynamic array, binary search, sorts, binary heap, hash table. Algorithms on graphs: depth-first search and its applications, algorithms for finding shortest paths. Greedy algorithms: Huffman's algorithm, Beladi's algorithm, Prim's and Kruskal's algorithms. Dynamic programming.",
    "Algorithms II": "Computational complexity theory: complexity classes P and NP, examples of NP-complete problems and their reductions to each other. Approximate algorithms and caching algorithms. Number-theoretic algorithms: comparison arithmetic, Miller-Rabin" +
                     "test, p-Pollard's algorithm, an introduction to cryptography. Advanced data structures: segment tree, binary search trees, skip-lists.",
    "Algorithms III": "Fast Fourier transform, Newton's fast division. Matching: Kuhn's algorithm, algorithm for finding a stable matching. Flows and cuts: search algorithms for maximum flow, minimum" +
                      "global cut. Algorithms on strings: search for a substring in a string, Aho-Korasik algorithm, suffix data structures.",
    
    "JVM Based Languages I": "kotlin: classes, interfaces, generics, extension functions, lambda expressions, reflection. Tools: assembly, debugging, profiling. Platforms: JVM, Android, native, JS, multiplatform",
    
    "Functional Programming": """Lambda calculus: terms, reduction, normal form, reduction strategies, Church-Rosser theorem, fixed point theorem and
recursion. Types: simple type system, Curry and Church versions, validation, inference, habitability, Hindley-Milner algorithm. Haskell language: loose semantics, algebraic data types, pattern
matching; semigroups and monoids; convolutions and traverses; functors, applicative functors and monads; monad transformers.""",
    "Computer System Architecture": """Electronic components. Conductivity control. Sequential and combinational logic. Logic. Building blocks of the processor. Single cycle processor. Multicycle processor. Design of the control device. Conveyor. Organization of memory.  Instruction coding. Interrupts and I/O.""",
    "Probability Theory": "Random event, conditional probability, independence, Bernoulli scheme, limit theorems. Random Variables: Density and Independence, Math. expectation and variance, the law of large numbers. Method of characteristic functions: inversions, various types of convergence of random variables, central limit theorem. Discrete random processes: Markov chains, random walks, branching processes.",
    "JVM Based Languages II": "Scala type system. Functional OOP. Futures and Promises API. Implicit parameters. SBT. Typeclasses, monoids, semigroups, functors, monads.",
    "Theory of Algorithms": "Computable functions, solvable and enumerable sets. Universal numbering. Rice's theorem. Fixed point theorem. A program that prints its own text. Algorithmically unsolvable problems: halting problem, Post matching problem, domino problem. Reducibility and arithmetic hierarchy, complete sets in hierarchy classes. The concept of information. Information on Hartley. Entropy of distribution. Information on Shannon. Entropy estimate for binomial coefficients. Uniquely decodable codes. Kraft–McMillan inequality. Shannon-Fano code, Huffman code, arithmetic coding. Cryptographic protocols: the problem of sharing a secret. Communication complexity. Estimates for communication protocols. Karchmer-Wigderson game. Application of information theory in communication complexity. Khrapchenko's theorem. Kolmogorov complexity",
    "Complex Analysis": "Theory of functions of a complex variable: holomorphy and equivalent properties, Cauchy integral formula, mean value theorem and maximum principle, zeros and poles of a holomorphic function, Laurent series, Cauchy's theoremabout residues, calculation of integrals and sums of series using residues, localization of roots, application of TFKT to the study of generating functions, conformal mappings. Fourier series: Lebesgue spaces, Hilbert spaces, Fourier coefficients in orthogonal system, Bessel inequality, best approximations, convergence of trigonometric Fourier series, Gibbs effect, approximation of continuous functions by polynomials, Fourier transform.",
    "Operating Systems": "Basic abstractions of the OS kernel. Processes and threads. Loading the OS and creating the first process. Planning and multitasking. Synchronization primitives. Memory management and protected mode. File systems. User environment management. Implementation of interprocess communication.",
    "Formal Language Theory": "Computability with finite memory, regular languages. Finite automata: deterministic, non-deterministic, two-way, probabilistic. Formal grammars and parsing algorithms. Computability and unsolvable problems, Turing machines.",
    "Computer Networks": "Fundamentals of data transmission networks. Five levels of the Internet (physical, channel, network, transport and applied). Wireless and mobile networks. Multimedia network technologies. Security of computer networks.",
    "Machine Learning - 1": "Linear methods of classification and regression. Logical methods of classification. Metric methods of classification and regression. Quality metrics, generalizing ability. Feature selection methods. Fundamentals of Bayesian methods.",
    "Machine Learning - 2": "Time series. Deep neural networks. Methods of teaching ranking. Collaborative filtering and matrix expansions. Thematic modeling. Reinforcement learning. Active learning. Causal impact and uplift modeling. Interpretation of machine learning models.",
    "High-load Systems Design": "Web services. Distributed systems. Service architecture. Monitoring. Application delivery.",
    "Methods and Algorithms of Heuristic Search": "Search methods for uninformed search: BFS, DFS, DFID. Algorithm A*: principle of operation, concept of heuristic function, properties of the algorithm. Heuristics: properties, relationship with the properties of the search algorithm. Modifications of the A* algorithm: search with iterative depth(IDA*), search for sub-optimal solutions (WA*, FocalA*), bidirectional search. Application of heuristic search algorithms for solving planning problems. Search methods for games with two players.",
    "Linux Programming": "Overview of Executable and Linkable Format (ELF). Dynamic libraries and plugins. Files and file systems. IO Scheduler, File System Events. Virtual File Systems. Processes and Access Rights. Rights. Linux Capabilities. Linux Namespaces. Interprocess Communication. Signals. Interprocess Communication. Queues, Channels, Shared Memory. Interprocess Communication. Ptrace. eBPF. Process Scheduling. Resource Limits.",

    "Computer Graphics": "Orthographic and perspective projections, affine transformations, homogeneous coordinates, quaternions. Graphics pipeline: primitive assembly, shaders, rasterization. Basic OpenGL primitives: buffers, VAOs, shaders, textures. Rendering types and options: indexed rendering, instancing, depth test, stencil test, clipping of invisible surfaces, translucency. Textures: views, filtering options, mipmaps. Lighting: lighting models, types of light sources, normal/bump/material mapping, baking. Rendering to texture and post-processing: depth blur, SSAO, gamma correction, anti-aliasing. Shadows: shadow volumes, shadow mapping, soft shadows. Deferred rendering. Rendering optimizations: batching, LOD, frustum culling, occlusion culling.",
    "Software Engineering": "Software project management. Tools for team development. Design and architecture of software systems. Principles of building highly loaded distributed systems. Services and microservice architecture. Working with databases, ORM systems. Unit testing and integration testing. Refactoring and clean code.",
    "Deep Learning": "Backpropagation. Stochastic gradient descent (SGD). Adaptive SGD: Adagrad, Adadelta, Adam. Dropout, initialization, batchnorm. Convolutional neural networks: VGG, Inception, ResNet, DenseNet, EfficientNet. Object detection: R-CNN, YoLo, SSD. Fully convolutional networks. Image segmentation: DeconvNet, SegNet, U-Net. Recurrent neural networks: LSTM, GRU. Recurrent models of visual attention, DRAW. Machine translation: encoder-decoder architecture. Google NMT, Transformer. BERT, GPT. Generative models: WaveNet. Generative adversarial networks: DCGAN, ProGan.",
    "Parallel Programming": "Processes/threads: architecture selection criteria, life in the OS kernel. Synchronization: primitives, algorithms, lock-free, wait-free, transactional memory, consensus. Parallel programming patterns: basic patterns, asynchronous I/O. Other performance enhancement technologies: compilers, SSE, OpenCL. Profiling and searching for errors. Technologies: OS threads, OpenMP, Intel TBB, Java.util.concurrent, Coroutines.",
    "Mobile Applications Development": "Android app development: architectural principles, android platform guidelines. Views and layouts. AndroidX library. Activities: fragments and lifecycle. Network requests. Unit testing.",
    "GPGPU Computing": "Introduction to OpenCL. GPU architecture. Examples of optimizations with local memory. Matrix transposition, matrix multiplication, merge sort. Sparse matrices. Bitonic sort, radix sort. Patch match, poisson reconstruction. Morton Code, LBVH Construction, TVL1 Surface Reconstruction. Rasterization: OpenGL, Larrabee, cudaraster. Ray Marching (SDF, shadertoy). Octree Textures.",
    "JVM Architecture": "Jvm architecture overview. Bytecode, classes and methods. Memory model, method calls. Interpreter and JIT compiler. Garbage collection.",
    "Numerical Methods": "Numericals errors. Libraries for numerical methods: blitz++. Local optimization, global optimization. Interpolation on a regular grid. Interpolation on an irregular grid. Numerical integration. Ordinary differential equations. Numerical differentiation. Partial differential equations on regular grids. Partial differential equations on irregular grids. Linear algebraic equations. Least squares techniques. Modeling fluid surface motion. Parallel computing. Introduction to computational geometry.",
    "Natural Language Processing": "TF-IDF, sentiment analysis, introduction to PyTorch. Natural language processing in information retrieval. String processing and text representation. String similarity. Spellcheck. First neural language models. Recurrent neural networks. Markov chains. Markov models and basics of information theory. Applications in NLP. LMs: N-Grams vs RNNs. Vector semantics. Unsupervised learning: clustering. Brown clustering. Duplicate detection. Thematic modeling. Sequence labeling. Attention mechanism, encoding-decoding, transformers.",

    
        "Databases": "Design and normalization of relational databases. Fundamentals of the SQL query language. Advanced SQL features. Working with database objects: functions, views, triggers. Query optimization.",
        
        "Mathematical Statistics": "Fundamentals of mathematical statistics: descriptive statistics, parameter estimation, confidence intervals, hypothesis testing, goodness of fit and homogeneity tests, linear regression. Monte Carlo method, resampling methods. Fundamentals of the Bayesian approach, Bayesian classification.",
        "3D Computer Vision": "Fundamentals of image processing, linear filters and convolutions. Key points: detectors, descriptors, matching, optical flow. Camera: device, internal and external parameters. Calculation of the point cloud and camera movement. PnP and Bundle Adjustment as optimization problems, Levenberg-Marquardt algorithm. Robust estimation: RANSAC, M-estimators. Modern SfM pipelines on the example of COLMAP.",
        "Convex Optimization": "Quadratic functions. Method of Lagrange multipliers and its generalizations. Basic varieties of gradient descent. Newton's method and interior point method."
    
    
    # This dictionary now contains all the subjects and their descriptions. You can use it for updating your HTML file or any other relevant application.

    # You can now use this updated dictionary for your HTML file or any other relevant applications.

}


# Function to create HTML structure for each subject
def create_subject_html(subject_name, description):
    return f"""
<div class="card mb-4">
<div class="card-body">
<a id="{subject_name.lower().replace(' ', '-')}"></a>
<h3 class="card-title">{subject_name}:</h3>
<p class="card-text">{description}</p>
</div>
</div>
"""


# Adding each subject to the HTML
for subject, description in subjects.items():
    html_template += create_subject_html(subject, description)

# Closing HTML tags
html_template += """
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"></script>
</body>
</html>
"""

# Saving the updated HTML content to a file
file_path = 'updated_curriculum.html'
with open(file_path, 'w') as file:
    file.write(html_template)


