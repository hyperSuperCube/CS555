ne = E2.shape[0]

# allocate sparse matrix arrays
m = 6  # FIXME for quadratics
AA = np.zeros((ne, m**2))
IA = np.zeros((ne, m**2), dtype=int)
JA = np.zeros((ne, m**2), dtype=int)
bb = np.zeros((ne, m))
ib = np.zeros((ne, m), dtype=int)
jb = np.zeros((ne, m), dtype=int)

# Assemble A and b
for ei in range(0, ne):
    # Step 1: set the vertices and indices
    K = E2[ei, :]
    x0, y0 = X[K[0]], Y[K[0]]
    x1, y1 = X[K[1]], Y[K[1]]
    x2, y2 = X[K[2]], Y[K[2]]

    # Step 2: compute the Jacobian, inv, and det
    J = np.array([[x1 - x0, x2 - x0],
                  [y1 - y0, y2 - y0]])
    invJ = np.linalg.inv(J.T)
    detJ = np.linalg.det(J)

    # Step 3a: set up quadrature nodes in the triangle
    ww = \
    np.array([0.44676317935602256, 0.44676317935602256, 0.44676317935602256,
              0.21990348731064327, 0.21990348731064327, 0.21990348731064327])
    xy = np.array([[-0.10810301816807008, -0.78379396366385990],
                   [-0.10810301816806966, -0.10810301816807061],
                   [-0.78379396366386020, -0.10810301816806944],
                   [-0.81684757298045740, -0.81684757298045920],
                   [0.63369514596091700, -0.81684757298045810],
                   [-0.81684757298045870, 0.63369514596091750]])
    xx, yy = (xy[:, 0]+1)/2, (xy[:, 1]+1)/2
    ww *= 0.5

    # Steb 3b: set element matrix and right-hand side to zero
    Aelem = np.zeros((m, m))
    belem = np.zeros((m,))
    
    # Step 3c: loop over each quadrature weight
    for w, x, y in zip(ww, xx, yy):
        # Step 3d: set quadratic basis at the quadrature points
        basis = \
        [(1-x-y)*(1-2*x-2*y), 
        x*(2*x-1), 
        y*(2*y-1), 
        4*x*(1-x-y), 
        4*x*y, 
        4*y*(1-x-y)]
        # FIXME
        

        dbasis = [[4*x-4*y-3, 4*x-1, 0, -8*x-4*y+4, 4*y, -4*y],
                  [4*x+4*y-3, 0, 4*y-1, -4*x, 4*x, -4*x-8*y+4]]# FIXME

        # Step 4: construct J^{-T} dphi
        dphi = invJ.dot(dbasis)

        # Step 5: add to element matrix
        xt, yt = J.dot(np.array([x, y])) + np.array([x0, y0])
        kappaelem = kappa(xt, yt)
        Aelem += (detJ / 2) * w * kappaelem * dphi.T @ dphi

        # Step 6: add to element rhs
        belem += (detJ / 2) * w * f(xt, yt) * basis

    # Step 7
    AA[ei, :] = Aelem.ravel()
    IA[ei, :] = np.repeat(K[np.arange(m)], m)
    JA[ei, :] = np.tile(K[np.arange(m)], m)
    bb[ei, :] = belem.ravel()
    ib[ei, :] = K[np.arange(m)]
    jb[ei, :] = 0

# convert matrices
A = sparse.coo_matrix((AA.ravel(), (IA.ravel(), JA.ravel())))
A.sum_duplicates()
b = \
sparse.coo_matrix((bb.ravel(), (ib.ravel(), jb.ravel()))).toarray().ravel()
