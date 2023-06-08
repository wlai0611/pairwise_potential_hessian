import numpy as np

def analytical_hessian(coordinates):
    '''
    Input: Matrix of natom by number of dimensions (2D or 3D)
    Output: A matrix of (natom x ndim) by (natom x ndim) where each cell represents second order
    partial derivative 
    '''
    num_atoms, num_dim = coordinates.shape
    TMP  = np.zeros((num_atoms, num_dim))#should be n_atom by n_dim
    H    = np.zeros((int(num_dim*num_atoms),int(num_dim*num_atoms)))
    for atom_i in range(num_atoms):
        for atom_j in range(atom_i):
            difference   = coordinates[atom_i,:] - coordinates[atom_j,:]
            bond_length  = np.sqrt(difference@difference)
            force_over_r = -24 * ((2/bond_length**14) - (1/bond_length**8))
            second_derivative_over_r_squared = 96 * (7 / pow (bond_length, 6) - 2) / pow (bond_length, 10)
            for dim_k in range(num_dim):
                delta_k = coordinates[atom_i,dim_k] - coordinates[atom_j,dim_k]
                #same atom, same coordinate
                TMP[atom_i,dim_k] += second_derivative_over_r_squared * delta_k**2 + force_over_r
                TMP[atom_j,dim_k] += second_derivative_over_r_squared * delta_k**2 + force_over_r
                for dim_l in range(num_dim):
                    delta_l = coordinates[atom_i,dim_l] - coordinates[atom_j,dim_l]
                    H_atom_i_index = int(num_dim*atom_i)
                    #same atom, different coordinate
                    H[H_atom_i_index+dim_k, H_atom_i_index+dim_l] += second_derivative_over_r_squared*delta_k*delta_l
                    H_atom_j_index = int(num_dim*atom_j)
                    H[H_atom_j_index+dim_k, H_atom_j_index+dim_l] += second_derivative_over_r_squared*delta_k*delta_l
                    if dim_k == dim_l:
                        H[H_atom_i_index+dim_k,H_atom_j_index+dim_l] = -second_derivative_over_r_squared*delta_k*delta_l - force_over_r
                        H[H_atom_j_index+dim_k,H_atom_i_index+dim_l] = -second_derivative_over_r_squared*delta_k*delta_l - force_over_r
                    else:
                        H[H_atom_i_index+dim_k,H_atom_j_index+dim_l] = -second_derivative_over_r_squared*delta_k*delta_l
                        H[H_atom_j_index+dim_k,H_atom_i_index+dim_l] = -second_derivative_over_r_squared*delta_k*delta_l

    for atom_i in range(num_atoms):
        for dim_k in range(num_dim):
            atom_dimension_index = int(atom_i*num_dim) + dim_k
            H[atom_dimension_index, atom_dimension_index] = TMP[atom_i, (atom_dimension_index-num_dim*atom_i)]

    return H


def numerical_hessian(coordinates, func, diff=0.01):
    '''
    ILLUSTRATION:  This is what the function returns for a two atom 2D system
    where X2 means X coordinate of atom 2.  V means the sum of the potential
    energy for all the atom pairs.
                         X1     |     Y1        |     X2   |   Y2   |
    ------------------------------------------------------------------
              |        d''V     |    d''V       |   d''V   |  d''V     
            X1|   ------------  |   -------     |  ------  | ------
              |       dX1X1     |    dX1Y1      |   dX1X2  |  dX1Y2
    ------------------------------------------------------------------- 
              |        d''V     |    d''V       |   d''V   |  d''V
            Y1|   ------------- |   -------     |   ----   |  ----
              |       dY1X1     |    dY1Y1      |   dY1X2  |  dY1Y2
    -------------------------------------------------------------------
              |      d''V       |   d''V        |   d''V   |  d''V
            X2|     ------      |  -------      |  ------  |  -----
              |      dX2X1      |   dX2Y1       |   dX2X2  |  dX2Y2
    ------------------------------------------------------------------
              |      d''V       |   d''V        |   d''V   |  d''V
            Y2|      -----      |   -----       |   -----  |  ----
              |      dY2X1      |   dY2Y1       |   dY2X2  |  dY2Y2

    I)  NOTATION
        d''V
        ----
        dY1X2   means the second partial derivative of V (sum of potentials) with respect to 
        the Y coordinate of atom 1 and the X coordinate of atom 2

        In a 2D, two atom example, V is a function of X1, Y1, X2, Y2: 
        V(X1,Y1,X2,Y2)

    II) CALCULATIONS
        1) OFFDIAGONALS
            To approximate the value on the OFFDIAGONALS such as the example I used above:
            
            d''V      V(X1,Y1+DIFF,X2+DIFF,Y2) + V(X1,Y1-DIFF,X2-DIFF,Y2) + 2*V(X1,Y1,X2,Y2) - V(X1,Y1+DIFF,X2,Y2) - V(X1,Y1,X2+DIFF,Y2)- V(X1,Y1-DIFF,X2,Y2) - V(X1,Y1,X2-DIFF,Y2)
            -----  = --------------------------------------------------------------------------------------------------------------------------------------------------------------
            dY1X2                                      2*DIFF^2
            
            where DIFF represents the perturbation magnitude (also called epsilon)

        1a) A NOTE ON VARIABLE NAMES
            In the computer code:
            I refer to Y1 as the "row". Y is the "row dimension". Atom 1 is the "row atom". 
            I refer to X2 as the "column". X is the "column dimension".  Atom 2 is the "column atom". 

            perturb_both_up[row,column] + perturb_both_down[row,column] + 2*reference - perturb_one_up[row] - perturb_one_up[column] - perturb_one_down[row] - perturb_one_down[row]
            ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                            2*DIFF^2

            where reference just means the original potential of x1,y1,x2,y2 without any perturbations
        
        2) DIAGONALS
            To approximate the value on the DIAGONALS such as another example:
            d''V      V(X1,Y1+DIFF,X2,Y2) - 2*V(X1,Y1,X2,Y2) + V(X1,Y1,X2+DIFF,Y2)
            -----  =  ------------------------------------------------------------
            dY1Y1                    DIFF^2

        2a) ANOTHER NOTE ON VARIABLE NAMES
            In the computer code:
            I refer to Y1 as the "row".  The above formula is represented:

            d''V    perturb_one_up[row] - 2*reference + perturb_one_down[column]
            ----- = ------------------------------------------------------------   
            dY1Y1                          DIFF^2
           
        
    '''
    natoms, ndims = coordinates.shape
    reference_potential = func(coordinates)
    H = np.zeros(shape=(natoms*ndims,natoms*ndims))

    perturb_one_up   = np.zeros(shape=(natoms*ndims))
    perturb_one_down = np.zeros(shape=(natoms*ndims))
    perturb_both_up  = np.zeros(shape=(natoms*ndims, natoms*ndims))
    perturb_both_down= np.zeros(shape=(natoms*ndims, natoms*ndims))

    column_perturb = np.zeros(shape=(natoms,ndims))
    row_perturb    = np.zeros(shape=(natoms,ndims))

    for column in range(natoms*ndims):
        column_atom      = column // ndims
        column_dimension = column % ndims
        
        column_perturb[column_atom,column_dimension] = diff
        perturb_one_up[column]   = func(coordinates + column_perturb)
        perturb_one_down[column] = func(coordinates - column_perturb)

        H[column,column] = perturb_one_up[column] - 2*reference_potential + perturb_one_down[column]
        H[column,column] = H[column,column]/(diff**2)

        for row in range(column):
            row_atom      = row//ndims
            row_dimension = row%ndims
            
            row_perturb[row_atom,row_dimension] = diff
            perturb_both_up[row,column]   = func(coordinates + column_perturb + row_perturb)
            perturb_both_down[row,column] = func(coordinates - column_perturb - row_perturb)
            row_perturb[row_atom,row_dimension] = 0

            numerator =  perturb_both_up[row,column] + perturb_both_down[row,column] + 2*reference_potential
            numerator -= perturb_one_up[column] + perturb_one_up[row] + perturb_one_down[column] + perturb_one_down[row]
            H[column,row] = H[row,column] = numerator/(2*diff**2)

        column_perturb[column_atom,column_dimension] = 0

    return H