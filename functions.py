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
    natoms, ndims = coordinates.shape
    reference_potential = func(coordinates)
    H = np.zeros(shape=(natoms*ndims,natoms*ndims))
    potentials = {'x+h,y': np.zeros(shape=(natoms,ndims)),
                  'x-h,y': np.zeros(shape=(natoms,ndims)),
                  'x+h,y+k':np.zeros(shape=(natoms*ndims, natoms*ndims)), 
                  'x-h,y-k':np.zeros(shape=(natoms*ndims, natoms*ndims)),
                 }
    column_perturb = np.zeros(shape=(natoms,ndims))
    row_perturb    = np.zeros(shape=(natoms,ndims))
    for column in range(natoms*ndims):
        column_atom = column//ndims
        column_dimension = column%ndims
        column_perturb[:,:] = 0
        column_perturb[column_atom,column_dimension] = diff
        potentials['x+h,y'][column_atom,column_dimension] = func(coordinates+column_perturb)
        potentials['x-h,y'][column_atom,column_dimension] = func(coordinates-column_perturb)
        H[column,column]=potentials['x+h,y'][column_atom,column_dimension]-2*reference_potential+potentials['x-h,y'][column_atom,column_dimension]
        H[column,column]=H[column,column]/(diff**2)
        for row in range(column):
            row_atom = row//ndims
            row_dimension = row%ndims
            row_perturb[:,:] = 0
            row_perturb[row_atom,row_dimension] = diff
            potentials['x+h,y+k'][row,column] = func(coordinates + column_perturb + row_perturb)
            potentials['x-h,y-k'][row,column] = func(coordinates - column_perturb - row_perturb)
            numerator=potentials['x+h,y+k'][row,column]-potentials['x+h,y'][column_atom,column_dimension]-potentials['x+h,y'][row_atom,row_dimension]
            numerator+=2*reference_potential-potentials['x-h,y'][column_atom,column_dimension]-potentials['x-h,y'][row_atom,row_dimension]
            numerator+=potentials['x-h,y-k'][row,column]
            H[column,row]=H[row,column]=numerator/(2*diff**2)
    return H