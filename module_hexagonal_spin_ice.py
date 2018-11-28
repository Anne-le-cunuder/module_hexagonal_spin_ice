#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:04:42 2018

@author: ifrerot
"""
from matplotlib.pylab import *
import itertools
from random import gauss
import numpy as np
import scipy.spatial as spa
from scipy.linalg import norm

    
class hexagonal_colloidal_spin_ice:
    """Class to define a colloidal spin ice model on a hexagonal lattice.
    Each edge contains a double-well trap which can cointain a single colloidal particle.
    Colloidal particules interact with each other via a 1/r^3 dipole-dipole interaction.
    A double-well is identified by a triplet (n1, n2, i) with i=0,1,2. 
    A particule in a double-well is located at position 
    R = n1 * u_1 + n2 * u_2 + e_i +/- (d / 2) * p_i
    where u_i generate the Bravais triangular lattice, 
    e_i points from the center of an hexagon towards the double-well i at the right of the hexagon
    and p_i is a unit vector along the symmetry axis of the double-well
    """

    
    def __init__(self,     
                 N1=6, #number of cells along the u1 direction
                 N2=6, #number of cells along the u2 direction
                 a=1, #distance between hexagon centers
                 d=0.3, #distance between the two wells of a double well
                 ordering='random', #intial positions of the colloids
                 verbose=True,
                 J = 1, #energy scale for interactions between colloids E = J / r^3
                 T = 1, #temperature
                 boundary_conditions='periodic', #periodic or free boundary conditions
                 ):
        assert N1 % 2 == 0 and N2 % 2 == 0, "needs even number of hexagons in both directions"
        if ordering == 'dipolar_spin_ice':
            assert N1 % 3 == 0 and N2 % 3 == 0, "needs a multiple of 3 for the number of hexagons in both directions for this ordering"
        assert a / d > sqrt(3), "double well too large ! Needs a / d > sqrt(3)"

        self.verbose = verbose
        self.boundary_conditions = boundary_conditions
        if boundary_conditions == 'free':
            print('WARNING : neighbours are calculated for PBC')
        self.ordering = ordering
        self.J = J
        self.T = T
        self.size = (N1, N2)
        
        u1 = a * array([1, 0]) #u1 and u2 : lattice vectors which join the centers of the hexagons
        u2 = a * array([0.5, -sqrt(3) / 2])
        self.lattice_vectors = (u1, u2)
        self.length_lattice_vector = a
 

        
        e0 = 0.5 * (u1 - u2) #e0, e1 and e2 : vectors from the center of an hexagon towards the center of a double-well
        e1 = 0.5 * u1
        e2 = 0.5 * u2
        self.double_well_center = (e0, e1, e2)
        
        p0 = d * array([-sqrt(3) / 2, 0.5]) #p0, p1 and p2 : vectors from one well to the other in a double-well
        p1 = d * array([0, 1])
        p2 = d * array([sqrt(3) / 2, 0.5])
        self.double_well_orientation = (p0, p1, p2)
        self.double_well_spacing = d
        
        v0 = (2 * u1 - u2) / 3 #v0, v1 : vectors from the center of an hexagon towards a vertex
        v1 = (u1 + u2) / 3
        self.vertex_center = (v0, v1)
        
        self.hexagon_centers = zeros((N1, N2, 2))
        for n1 in range(N1):
            for n2 in range(N2):
                v = n1 * u1 + n2 * u2
                self.hexagon_centers[n1, n2] = array([
                        v[0] % (a * N1),
                        v[1] % (a * N2 * sqrt(3) / 2)
                    ])
    
        self.double_well_centers = self.hexagon_centers[:, :, np.newaxis, :] + self.double_well_center
        self.vertex_centers = self.hexagon_centers[:, :, np.newaxis, :] + self.vertex_center
        
        self.initialize_state(ordering=ordering)       
        self.calculate_neighbours_dict()
        self.calculate_NN_list()
        self.energy = self.calculate_energy(
                self.calculate_colloid_positions(
                        self.spin_orientations
                        ).reshape(N1 * N2 * 3, 2)
                )
        
        self.distances = self.calculate_distances()
        
        self.nb_fig = 0 #nb de figures tracées
        if self.verbose:
            print("Initial ordering = {}; initial energy = {}".format(
                    ordering, self.energy)
            )
            print("Initial configuration :")
            figure(self.nb_fig)
            self.plot_colloid_config()
            title('Initial colloid configuration')
            self.nb_fig += 1

            figure(self.nb_fig)
            self.plot_spin_config()
            self.nb_fig += 1
            title('Initial spin configuration')

    
    def initialize_state(self, ordering):
        N1, N2 = self.size
        a = self.length_lattice_vector
        self.Lx = N1 * a
        self.Ly = N2 * a * sqrt(3) / 2

        if ordering == 'random':
            self.spin_orientations = 2 * randint(2, size=(N1, N2, 3)) - 1
        
        elif ordering == 'ferromagnetic':
            self.spin_orientations = ones((N1, N2, 3), dtype=int)
        
        elif ordering == 'Nicolas':
            self.spin_orientations = zeros((N1, N2, 3), dtype='int')
            self.spin_orientations[arange(N1)%2==0, :, 1] = 1
            self.spin_orientations[arange(N1)%2==1, :, 1] = 1
            self.spin_orientations[:, arange(N2)%2==0, 0] = -1
            self.spin_orientations[:, arange(N2)%2==0, 2] = -1
            self.spin_orientations[:, arange(N2)%2==1, 0] = 1
            self.spin_orientations[:, arange(N2)%2==1, 2] = 1
        
        elif ordering == 'dipolar_spin_ice':
            self.spin_orientations = ones((N1, N2, 3), dtype=int)
            for n1 in range(N1):
                for n2 in range(N2):
                    k = (n1 - n2) % 3
                    if k == 0:
                        self.spin_orientations[n1, n2, 1] = -1
                    elif k == 2:
                        self.spin_orientations[n1, n2, :] = [-1] * 3


        else : 
            raise Exception('I do not know this ordering')
            
    def calculate_NN_list(self):
        """forms a N_spins x 4 array A s.t. A[i] = [i0, i1, i2, i3] is the list
        of ravelled indices of NN spins of i."""
        N1, N2 = self.size
        L = np.array([self.Lx, self.Ly])
        Nb_spins = 3 * N1 * N2
        A = [[] for i in range(Nb_spins)]
        d1 = self.length_lattice_vector * np.sqrt(7 / 12)
        for i in range(Nb_spins):
            for j in range(i):
                spin_i = np.unravel_index(i, (N1, N2, 3))
                spin_j = np.unravel_index(j, (N1, N2, 3))                
                R_ij_1 = np.abs(self.double_well_centers[spin_i] -
                            self.double_well_centers[spin_j])
                if self.boundary_conditions == 'periodic':
                    R_ij = np.minimum(R_ij_1, L - R_ij_1)
                if norm(R_ij) < d1:
                    A[i].append(j)
                    A[j].append(i)
        self.NN_list = A
        return A
    
    def plot_NN_spins(self, spin):
        """plot the spins nearest neighbours of spin=(n1,n2,k)"""
        N1, N2 = self.size
        i = np.ravel_multi_index([[spin[k]] for k in range(3)], (N1, N2, 3))[0]
        pos = self.double_well_centers.reshape((3 * N1 * N2, 2))[self.NN_list[i]]
        plot(pos[:, 0], pos[:, 1], 'bo')

    
    def calculate_neighbours_dict(self):
        N1, N2 = self.size
        self.spin_to_vertex_dict = dict()
        for n1, n2, i in itertools.product(range(N1), range(N2), range(3)):
            spin = (n1, n2, i)
            orientation = self.spin_orientations[spin]      
            if i == 0:
                if n2 > 0:
                    self.spin_to_vertex_dict[spin] = [
                            (n1, n2, 0),#from vertex
                            (n1, n2-1, 1) #to vertex
                    ][::orientation]
                elif n2 == 0 and self.boundary_conditions == 'periodic':
                    self.spin_to_vertex_dict[spin] = [
                            (n1, n2, 0),#from vertex
                            ((n1 - N2 // 2) % N1, N2-1, 1) #to vertex
                    ][::orientation]

            if i == 1:
                self.spin_to_vertex_dict[spin] = [
                            (n1, n2, 1),#from vertex
                            (n1, n2, 0) #to vertex
                    ][::orientation]
                    
            if i == 2:
                if n2 < N2 - 1:
                    self.spin_to_vertex_dict[spin] = [
                            ((n1 - 1) % N1, n2 + 1, 0),#from vertex
                            (n1, n2, 1) #to vertex
                    ][::orientation]
                elif n2 == N2 - 1:
                    self.spin_to_vertex_dict[spin] = [
                            ((n1 - 1 + N2 // 2) % N1, 0, 0),#from vertex
                            (n1, n2, 1) #to vertex
                    ][::orientation]
                    
        self.vertex_to_spin_dict = dict()
        for n1, n2, i in itertools.product(range(N1), range(N2), range(2)):
            vertex = (n1, n2, i)
            self.vertex_to_spin_dict[vertex] = {
                    'in': [spin 
                           for spin in itertools.product(range(N1), range(N2), range(3))
                           if vertex == self.spin_to_vertex_dict[spin][1]
                   ],#ingoing spins
                    'out': [spin 
                           for spin in itertools.product(range(N1), range(N2), range(3))
                           if vertex == self.spin_to_vertex_dict[spin][0]
                   ]#outgoing spins
                }
        

    def calculate_colloid_positions(self, spin_orientations):
        return self.double_well_centers + 0.5 * spin_orientations[:,:,:, np.newaxis] * array(self.double_well_orientation)

    def plot_colloid_config(self, symbol='bo'):
        N1, N2 = self.size
        d = self.double_well_spacing
        p = self.double_well_orientation

        pos = self.calculate_colloid_positions(self.spin_orientations).reshape(N1 * N2 * 3, 2)

        for n1, n2, i in itertools.product(range(N1), range(N2), range(3)):
            c1 = self.double_well_centers[n1, n2, i] + p[i] / sqrt(3) / d / 2
            c2 = self.double_well_centers[n1, n2, i] - p[i] / sqrt(3) / d / 2
            plot([c1[0], c2[0]], [c1[1], c2[1]], 'k-')
        plot(pos[:, 0], pos[:, 1], symbol)

        a = self.length_lattice_vector
        xlim(- a, a * (N1 + 0.5))
        ylim(- a, a * (N2 * sqrt(3) / 2 + 0.5))

        
    def plot_spin_config(self):
        N1, N2 = self.size
        a = self.length_lattice_vector
        d = self.double_well_spacing
        p = self.double_well_orientation
        def drawArrow(A, B):
            arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
                      head_width=0.2, 
                      length_includes_head=True,
                      color='grey')
        for n1, n2, i in itertools.product(range(N1), range(N2), range(3)):
            c1 = self.double_well_centers[n1, n2, i] + self.spin_orientations[n1, n2, i] * p[i] / sqrt(3) / d / 2
            c2 = self.double_well_centers[n1, n2, i] - self.spin_orientations[n1, n2, i] * p[i] / sqrt(3) / d / 2
            drawArrow(c2, c1)
        
        for n1, n2, i in itertools.product(range(N1), range(N2), range(2)):
            v = (n1, n2, i)
            c = self.vertex_centers[v]
            spins = self.vertex_to_spin_dict[v]
            charge0 = - len(spins['out']) + len(spins['in'])
            charge0 *= (-1)**i
            charge = (3 + charge0) // 2
            color = ['r', 'C1', 'C0', 'b']
            plot([c[0]], [c[1]], 'o', color=color[charge])
                
        for n1, n2 in itertools.product(range(N1), range(N2)):
            chi = self.compute_chirality(n1, n2)
            x, y = self.hexagon_centers[n1, n2, :]
            text(x - a / 10, y - a / 10, str(chi))

        xlim(- a, a * (N1 + 0.5))
        ylim(- a, a * (N2 * sqrt(3) / 2 + 0.5))
    
    def compute_chirality(self, n1, n2):
        N1, N2 = self.size
        if 0 < n2 < N2 - 1:
            return (
                    sum(self.spin_orientations[n1, n2, :]) - 
                    self.spin_orientations[(n1-1)%N1, n2, 1] - 
                    self.spin_orientations[n1, n2-1, 2] - 
                    self.spin_orientations[(n1-1)%N1, (n2+1)%N2, 0]
                )
            
        else:
            return (
                    sum(self.spin_orientations[n1, n2, :]) - 
                    self.spin_orientations[(n1-1)%N1, n2, 1] - 
                    self.spin_orientations[(n1 - N2 // 2) % N1, (n2-1)%N2, 2] - 
                    self.spin_orientations[(n1-1)%N1, (n2+1)%N2, 0]
                )
                                


    def plot_spin_neighbours(self, spin):
        n1, n2, i = spin
        d = self.double_well_spacing
        p = self.double_well_orientation

        c1 = self.double_well_centers[n1, n2, i] + p[i] / sqrt(3) / d / 2
        c2 = self.double_well_centers[n1, n2, i] - p[i] / sqrt(3) / d / 2
        plot([c1[0], c2[0]], [c1[1], c2[1]], 'r-')
        from_vertex, to_vertex = self.spin_to_vertex_dict[spin]
        c1 = self.vertex_centers[from_vertex]
        c2 = self.vertex_centers[to_vertex]
        plot([c1[0]], [c1[1]], 'ks')
        plot([c2[0]], [c2[1]], 'rs')
        
    def plot_vertex_neighbours(self, vertex):
        n1, n2, i = vertex
        v = self.vertex_centers[vertex]
        plot([v[0]], [v[1]], 'ks')
        ingoing_spins = self.vertex_to_spin_dict[vertex]['in']
        outgoing_spins = self.vertex_to_spin_dict[vertex]['out']
        for spin in ingoing_spins:
            c = self.double_well_centers[spin]
            plot([c[0]], [c[1]], 'bs')
        for spin in outgoing_spins:
            c = self.double_well_centers[spin]
            plot([c[0]], [c[1]], 'rs')

    
    def calculate_interaction_energy(self, R1, R2):
        if self.boundary_conditions == 'free':
            return self.J / norm(R1 - R2)**3 
    
        elif self.boundary_conditions == 'periodic':
            N1, N2 = self.size
            Lx = N1 * self.length_lattice_vector
            Ly = N2 * self.length_lattice_vector * sqrt(3) / 2
            return self.J / (
                       ((Lx / pi) * sin(pi / Lx * (R1[0] - R2[0])))**2 + 
                       ((Ly / pi) * sin(pi / Ly * (R1[1] - R2[1])))**2 
                    )**(3 / 2)


    def calculate_interaction_energy_group(self, R):
        """Calculate the interaction energy between each pair of a group of particles"""
        if self.boundary_conditions == 'free':
            return self.J * sum(
                1 / spa.distance.pdist(R)**3
            )
        elif self.boundary_conditions == 'periodic':
            Lx = self.Lx
            Ly = self.Ly
            dx = spa.distance.pdist(R[:, 0][:, np.newaxis])
            dy = spa.distance.pdist(R[:, 1][:, np.newaxis])
            
            return  self.J * sum(
                    (
                        ((Lx / np.pi) * np.sin(np.pi / Lx * dx))**2 + 
                        ((Ly / np.pi) * np.sin(np.pi / Ly * dy))**2 
                    )**(-3 / 2)
            ) 


    def calculate_energy(self, pos):
        return self.calculate_interaction_energy_group(pos)
        
            
        
    def run_MC_step_SSF(self, nb_steps=1):#single spin flips
        N1, N2 = self.size
        N_particles = N1 * N2 * 3
        pos = self.calculate_colloid_positions(self.spin_orientations)
        E_MC = [self.energy]
        liste_spins = []
        for i in range(nb_steps):        
            spin = (randint(N1), randint(N2), randint(3))
            p = self.double_well_orientation[spin[-1]]
            s_before = self.spin_orientations[spin]
            R_before = pos[spin]
            R_after = R_before - s_before * p

            DeltaE = sum([
                    self.calculate_interaction_energy(pos[j], R_after) - 
                    self.calculate_interaction_energy(pos[j], R_before)
                    for j in itertools.product(range(N1), range(N2), range(3))
                    if j != spin
                ])

            if DeltaE < 0 :
                accepted = True
            else:
                accepted = (rand() < exp(- DeltaE / self.T))
            if accepted:
                self.energy += DeltaE
                move_status = 'accepted'
                self.spin_orientations[spin] = - s_before
                pos[spin] = R_after
                liste_spins.append(spin)
            else:
                move_status = 'rejected'
                
            if self.verbose:
                print("MC SSF move {} / {}, move {} (DeltaE={}), E/N={}".format(
                    i, 
                    nb_steps,
                    move_status, 
                    DeltaE,
                    self.energy / N_particles
                    )
            )
            E_MC.append(self.energy)
        self.calculate_neighbours_dict()
        return E_MC, len(liste_spins)

    def choose_loop(self):
        N1, N2 = self.size
        loop_spins = []
        loop_vertex = []
        
        new_vertex = (randint(N1), randint(N2), randint(2))
        while new_vertex not in loop_vertex:
            loop_vertex.append(new_vertex)
            outgoing_spins = self.vertex_to_spin_dict[new_vertex]['out']
            if len(outgoing_spins) == 0: #3-in vertex
                return ([], [])
            else:
                k = randint(len(outgoing_spins))
                new_spin = outgoing_spins[k]
                loop_spins.append(new_spin)
                new_vertex = self.spin_to_vertex_dict[new_spin][1]
        i = loop_vertex.index(new_vertex)
        return (loop_spins[i:], loop_vertex[i:])
                
            
    def run_MC_step_loop(self):
        N1, N2 = self.size
        spins, vertex = self.choose_loop()
        nb_spins_flipped = 0
        if len(spins) > 2:
            old_pos = self.calculate_colloid_positions(self.spin_orientations)
            new_orientations = copy(self.spin_orientations)
            for spin in spins:
                new_orientations[spin] = -new_orientations[spin]
            new_pos = self.calculate_colloid_positions(new_orientations)
            
            DeltaE = sum([
                    self.calculate_interaction_energy(new_pos[i], new_pos[j]) -
                    self.calculate_interaction_energy(old_pos[i], old_pos[j])
                    for i in itertools.product(range(N1), range(N2), range(3))
                    for j in spins
                    if i != j
                ]) - 0.5 * sum([
                    self.calculate_interaction_energy(new_pos[i], new_pos[j]) -
                    self.calculate_interaction_energy(old_pos[i], old_pos[j])
                    for i in spins
                    for j in spins
                    if i != j
                ])
        

            if DeltaE < 0 :
                accepted = True
            else:
                accepted = (rand() < exp(- DeltaE / self.T))
    
            if accepted:
                move_status = 'accepted'
#                figure(system.nb_fig)
#                system.plot_spin_config()
#                system.nb_fig += 1
#                for spin in spins:
#                    c = self.double_well_centers[spin]
#                    plot([c[0]], [c[1]], 'rs')
#                title(move_status + 'DeltaE=' + str(DeltaE))

                self.energy += DeltaE
                self.spin_orientations = new_orientations
                for i, s in enumerate(spins):
                    v = vertex[i]
                    self.spin_to_vertex_dict[s] = self.spin_to_vertex_dict[s][::-1]
                    self.vertex_to_spin_dict[v]['out'].remove(s)
                    self.vertex_to_spin_dict[v]['in'].append(s)
                    self.vertex_to_spin_dict[v]['in'].remove(spins[i-1])
                    self.vertex_to_spin_dict[v]['out'].append(spins[i-1])
                nb_spins_flipped = len(spins)
            else:
                move_status = 'rejected'
                
            if self.verbose:
                print("MC loop, move {} (DeltaE={}), E/N={}".format(
                        move_status, 
                        DeltaE,
                        self.energy / (N1 * N2 * 3)
                        )
                )
        return nb_spins_flipped
    
    def run_MC_cycle(self, N_cycles=1, N_SSF=10, N_loop=1):
        E_MC = []
        nb_spins_flipped_SSF = 0
        nb_spins_flipped_loop = 0
        for i in range(N_cycles):
            E, n = self.run_MC_step_SSF(nb_steps=N_SSF)
            E_MC += E
            nb_spins_flipped_SSF += n
            for k in range(N_loop):
                nb_spins_flipped_loop += self.run_MC_step_loop()
                E_MC.append(self.energy)
        return E_MC, nb_spins_flipped_SSF, nb_spins_flipped_loop

    
    def calculate_charge_order_param(self):
        N1, N2 = self.size
        return sum(
            (
                - len(self.vertex_to_spin_dict[s]['out']) 
                + len(self.vertex_to_spin_dict[s]['in'])
            ) * (-1)**s[-1] 
            for s in itertools.product(range(N1), range(N2), range(2))
        )
    
    def calculate_magnetization(self):
        N1, N2 = self.size
        return sum(
            self.double_well_orientation[i] * self.spin_orientations[(n1, n2, i)] 
            for n1, n2, i in itertools.product(range(N1), range(N2), range(3))
            )
    
    def calculate_distances(self):
        N1, N2 = self.size
        dist = {}
        for s0 in itertools.product(range(N1), range(N2), range(3)):
            R0 = self.double_well_centers[s0]
            for s1 in itertools.product(range(N1), range(N2), range(3)):
                R1 = self.double_well_centers[s1]
                d = tuple(around(R0 - R1, 8))
                if d in dist.keys():
                    dist[d] += 1
                else:
                    dist[d] = 1
        return dist

    

    def calculate_magnetic_correlations(self):
        N1, N2 = self.size
        correl = dict()
        for s0 in itertools.product(range(N1), range(N2), range(3)):
            for s1 in itertools.product(range(N1), range(N2), range(3)):
                c = self.spin_orientations[s0] * self.spin_orientations[s1] * dot(
                        self.double_well_orientation[s0[-1]],
                        self.double_well_orientation[s1[-1]]                   
                        )
                d = tuple(
                        around(
                                self.double_well_centers[s0] - 
                                self.double_well_centers[s1],
                                8)
                        )
                if d in correl.keys():
                    correl[d] += c
                else:
                    correl[d] = c
                    
        return correl

    def calculate_magnetic_correlations_bis(self):
        N1, N2 = self.size
        correl = dict()
        
        for s0 in itertools.product(range(N1), range(N2), range(3)):
            for s1 in itertools.product(range(N1), range(N2), range(3)):
                c = self.spin_orientations[s0] * self.spin_orientations[s1] * dot(
                        self.double_well_orientation[s0[-1]],
                        self.double_well_orientation[s1[-1]]                   
                        )
                d = tuple(
                        around(
                                self.double_well_centers[s0] - 
                                self.double_well_centers[s1],
                                8)
                        )
                if d in correl.keys():
                    correl[d] += c
                else:
                    correl[d] = c
                    
        return correl


        
    def belongs_to_BZ(self, k):
        pass
    
    def calculate_magnetic_structure_factor(self):
        correl = self.calculate_magnetic_correlations()
        X = array([i for i in correl.keys()])
        c = array([correls[tuple(x)] for x in X])
        N1, N2 = self.size
        a = self.length_lattice_vector
        Lx = N1 * a
        Ly = N2 * a * sqrt(3) / 2
        Nx = 200
        Ny = 100
        Kx = linspace(-4, 4, Nx, endpoint=False) * pi / a
        Ky = linspace(-4, 4, Ny, endpoint=False) * pi / a
        S = zeros((Nx, Ny), dtype='complex')
        for kx, ky in itertools.product(range(Nx), range(Ny)):
            S[kx, ky] = sum(
                    exp(1j * (
                            Kx[kx] * X[:, 0] + Ky[ky] * X[:, 1])
                    ) * c
            )
        return Kx, Ky, S

    def plot_BZ(self):
        """Plot the Brilloin zone"""
        a = self.length_lattice_vector
        G1 = 2 * pi / a / sqrt(3) * array([sqrt(3), 1])
        G2 = 4 * pi / a /sqrt(3) * array([0, -1])
        self.nb_fig += 1
        fig = figure(self.nb_fig)
        ax = fig.add_subplot(1, 1, 1)
        liste_G = [G1, G2, G1 + G2, -G1, -G2, -G1 - G2]
        for G in liste_G:
            ax.arrow(0, 0, *G, width=0.1)
        K1 = 2 * pi / a / 3 * array([1, sqrt(3)])
        K2 = 2 * pi / a / 3 * array([1, -sqrt(3)])
        BZ = [K1 + K2, K1, -K2, -K1 - K2, -K1, K2]
        x_BZ = [K[0] for K in BZ]
        y_BZ = [K[1] for K in BZ]
        ax.plot(x_BZ, y_BZ, 'o')
        self.fig_BZ = fig
        
    def plot_magnetic_structure_factor(self):
        Kx, Ky, S = self.calculate_magnetic_structure_factor()
        self.plot_BZ()
        ax = self.fig_BZ.axes[0]
        ax.pcolormesh(Kx, Ky, S.T.real)
        
    def calculate_ferro_SS_order_param(self):
        correls = self.calculate_magnetic_correlations()
        X = np.array([i[0] for i in correls.keys()])
        c = np.array([i for i in correls.values()])
        kx = 4 * pi / 3 / self.length_lattice_vector
        ferro_OP = np.sum(c)
        SS_OP = np.sum(np.exp(1j * kx * X) * c).real
        return (ferro_OP, SS_OP)
            
        

       

class hexagonal_colloidal_spin_ice_continuous(hexagonal_colloidal_spin_ice):
    """
    Class which allows particles to move in continuous space inside the double-well
    """
    def __init__(self,     
                 N1=6, #number of cells along the u1 direction
                 N2=6, #number of cells along the u2 direction
                 a=1, #distance between hexagon centers
                 d=0.3, #distance between the two wells of a double well
                 ordering='random', #intial positions of the colloids
                 verbose=True,
                 J = 1, #energy scale for interactions between colloids E = J / r^3
                 T = 1, #temperature
                 h = 0.5 * 0.3,#height of the barrier between two wells
                 boundary_conditions='periodic', #periodic or free boundary conditions
                 cutoff_neighbour=None, #cutoff length for interaction energy. nth neighbor, or None. 
                 ):
        assert N1 % 2 == 0 and N2 % 2 == 0, "needs even number of hexagons in both directions"
        if ordering == 'dipolar_spin_ice':
            assert N1 % 3 == 0 and N2 % 3 == 0, "needs a multiple of 3 for the number of hexagons in both directions for this ordering"
        assert a / d > sqrt(3), "double well too large ! Needs a / d > sqrt(3)"
        assert cutoff_neighbour in [None, 1, 2], "cutoff_neighbour must be None, 1 or 2"

        self.verbose = verbose
        self.boundary_conditions = boundary_conditions
        if boundary_conditions == 'free':
            print('WARNING : neighbours are calculated for PBC')
        self.ordering = ordering
        self.J = J
        self.T = T
        self.size = (N1, N2)

        self.h = h

        
        u1 = a * array([1, 0]) #u1 and u2 : lattice vectors which join the centers of the hexagons
        u2 = a * array([0.5, -sqrt(3) / 2])
        self.lattice_vectors = (u1, u2)
        self.length_lattice_vector = a
        
        e0 = 0.5 * (u1 - u2) #e0, e1 and e2 : vectors from the center of an hexagon towards the center of a double-well
        e1 = 0.5 * u1
        e2 = 0.5 * u2
        self.double_well_center = (e0, e1, e2)
        
        p0 = d * array([-sqrt(3) / 2, 0.5]) #p0, p1 and p2 : vectors from one well to the other in a double-well
        p1 = d * array([0, 1])
        p2 = d * array([sqrt(3) / 2, 0.5])
        self.double_well_orientation = (p0, p1, p2)
        self.double_well_spacing = d
        
        v0 = (2 * u1 - u2) / 3 #v0, v1 : vectors from the center of an hexagon towards a vertex
        v1 = (u1 + u2) / 3
        self.vertex_center = (v0, v1)
        
        self.hexagon_centers = zeros((N1, N2, 2))
        for n1 in range(N1):
            for n2 in range(N2):
                v = n1 * u1 + n2 * u2
                self.hexagon_centers[n1, n2] = array([
                        v[0] % (a * N1),
                        v[1] % (a * N2 * sqrt(3) / 2)
                    ])
    
        self.double_well_centers = self.hexagon_centers[:, :, np.newaxis, :] + self.double_well_center
        self.vertex_centers = self.hexagon_centers[:, :, np.newaxis, :] + self.vertex_center
        
        self.initialize_state(ordering=ordering)       
        self.colloid_positions = self.calculate_colloid_positions(self.spin_orientations)
        self.calculate_neighbours_dict()
        self.calculate_NN_list()
        self.distances = self.calculate_distances()

        if cutoff_neighbour == 1:
            self.cutoff_length = a
        elif cutoff_neighbour == 2:
            self.cutoff_length = 2 * a / np.sqrt(3)
        else:
            self.cutoff_length = None
            
        self.energy = self.calculate_energy()
        
        
        self.nb_fig = 0 #nb de figures tracées
        if self.verbose:
            print("Initial ordering = {}; initial energy = {}".format(
                    ordering, self.energy)
            )
            print("Initial configuration :")
            figure(self.nb_fig)
            self.plot_colloid_config()
            title('Initial colloid configuration')
            self.nb_fig += 1

            figure(self.nb_fig)
            self.plot_spin_config()
            self.nb_fig += 1
            title('Initial spin configuration')

        

        
    def plot_colloid_config(self, symbol='bo'):
        N1, N2 = self.size
        d = self.double_well_spacing
        p = self.double_well_orientation

        pos = self.colloid_positions.reshape(N1 * N2 * 3, 2)

        for n1, n2, i in itertools.product(range(N1), range(N2), range(3)):
            c1 = self.double_well_centers[n1, n2, i] + p[i] / sqrt(3) / d / 2
            c2 = self.double_well_centers[n1, n2, i] - p[i] / sqrt(3) / d / 2
            plot([c1[0], c2[0]], [c1[1], c2[1]], 'k-')
        plot(pos[:, 0], pos[:, 1], symbol)

        a = self.length_lattice_vector
        xlim(- a, a * (N1 + 0.5))
        ylim(- a, a * (N2 * sqrt(3) / 2 + 0.5))    

    def calculate_spin_orientations(self, colloid_positions):
        N1, N2 = self.size
        spin_orientations = zeros((N1, N2, 3), dtype='int')
        for i in range(3):
            spin_orientations[:, :, i] = sign(dot(
                    colloid_positions[:, :, i] - self.double_well_centers[:, :, i],
                    self.double_well_orientation[i]
                ))
                             
        return spin_orientations

    
    def potential(self, r):
        x, y = r
        return self.h * (
                (
                        2 * x / self.double_well_spacing
                    )**4 - 8 * (
                        x / self.double_well_spacing
                    )**2 + 1
                ) + 16 * self.h * (y / self.double_well_spacing)**2
        
    def calculate_potential(self, colloid, r):
        """
        Potential experienced by the colloid (n1, n2, i)
        when it is located in r. The potential is a double-well potential 
        around colloid (n1, n2, i)
        """
        n1, n2, i = colloid
        e_parallel = self.double_well_orientation[i] / self.double_well_spacing #unit vector along the double-well
        e_perp = np.array([e_parallel[1], -e_parallel[0]]) #unit vector perpendicular to the double-well
        U = np.array([e_parallel, e_perp])
        r_tilde = np.dot(U, r - self.double_well_centers[colloid]) #coordinates in the frame of the double-well
        return self.potential(r_tilde)
    
    def initialize_state(self, ordering):
        N1, N2 = self.size
        a = self.length_lattice_vector
        self.Lx = N1 * a
        self.Ly = N2 * a * np.sqrt(3) / 2
        if ordering == 'random':
            self.spin_orientations = 2 * randint(2, size=(N1, N2, 3)) - 1
        
        elif ordering == 'ferromagnetic':
            self.spin_orientations = ones((N1, N2, 3), dtype=int)
        
        elif ordering == 'Nicolas':
            self.spin_orientations = zeros((N1, N2, 3), dtype='int')
            self.spin_orientations[arange(N1)%2==0, :, 1] = 1
            self.spin_orientations[arange(N1)%2==1, :, 1] = 1
            self.spin_orientations[:, arange(N2)%2==0, 0] = -1
            self.spin_orientations[:, arange(N2)%2==0, 2] = -1
            self.spin_orientations[:, arange(N2)%2==1, 0] = 1
            self.spin_orientations[:, arange(N2)%2==1, 2] = 1
        
        elif ordering == 'dipolar_spin_ice':
            self.spin_orientations = ones((N1, N2, 3), dtype=int)
            for n1 in range(N1):
                for n2 in range(N2):
                    k = (n1 - n2) % 3
                    if k == 0:
                        self.spin_orientations[n1, n2, 1] = -1
                    elif k == 2:
                        self.spin_orientations[n1, n2, :] = [-1] * 3


        else : 
            raise Exception('I do not know this ordering')

    def run_MC_step_particle_move(self, nb_steps=1, step_move=0.01, compute_new_orientations=True):#move a colloid in continuous space
        N1, N2 = self.size
        N_particles = N1 * N2 * 3
        E_MC = [self.energy]
        liste_spins = []
        for i in range(nb_steps):
            colloid_ravel = randint(3 * N1 * N2)
            colloid = np.unravel_index(colloid_ravel,
                                       np.shape(self.colloid_positions[:,:,:,0])
                                       )

            R_before = self.colloid_positions[colloid]
            R_after = R_before + np.random.randn(2) * step_move            
            
            Db = np.delete(self.colloid_positions.reshape((3 * N1 * N2, 2)), 
                           colloid_ravel, 
                           axis=0
                           ) - R_before
            Da = Db + R_before - R_after
            
            if self.boundary_conditions == 'periodic':
                L = np.array([self.Lx, self.Ly])
                Db = norm(np.minimum(np.abs(Db), L - np.abs(Db)), axis=-1)
                Da = norm(np.minimum(np.abs(Da), L - np.abs(Da)), axis=-1)
            
            elif self.boundary_conditions == 'free':
                Db = norm(Db, axis=-1)
                Da = norm(Da, axis=-1)
                
            if self.cutoff_length:
                Db = Db[Db < self.cutoff_length]
                Da = Da[Da < self.cutoff_length]

            DeltaE = self.J * sum(Da**-3 - Db**-3
                ) + self.calculate_potential(colloid, R_after) - self.calculate_potential(colloid, R_before)
               
                
            if DeltaE < 0 :
                accepted = True
            else:
                if self.T == 0.:
                    accepted = False
                else:
                    accepted = (rand() < exp(- DeltaE / self.T))
            if accepted:
                self.energy += DeltaE
                move_status = 'accepted'
                self.colloid_positions[colloid] = R_after
                liste_spins.append(colloid)
            else:
                move_status = 'rejected'
                
            if self.verbose:
                print("MC particle move {} / {}, move {} (DeltaE={}, Ri={}, Rf={}), E/N={}".format(
                    i, 
                    nb_steps,
                    move_status, 
                    DeltaE,
                    R_before,
                    R_after,
                    self.energy / N_particles
                    )
            )
            E_MC.append(self.energy)
            
        if compute_new_orientations:
            self.spin_orientations = self.calculate_spin_orientations(self.colloid_positions)
            self.calculate_neighbours_dict()
        return E_MC, len(liste_spins) / nb_steps

    def run_MC_step_SSF(self, nb_steps=1):#single spin flips
        N1, N2 = self.size
        N_particles = N1 * N2 * 3
        E_MC = [self.energy]
        liste_spins = []
        for i in range(nb_steps):    
            colloid_ravel = randint(3 * N1 * N2)
            colloid = np.unravel_index(colloid_ravel,
                                       np.shape(self.colloid_positions[:,:,:,0])
                                       )

            R_before = self.colloid_positions[colloid]
            R_after = 2 * self.double_well_centers[colloid] - R_before

            Db = np.delete(self.colloid_positions.reshape((3 * N1 * N2, 2)), 
                           colloid_ravel, 
                           axis=0
                           ) - R_before
            Da = Db + R_before - R_after
            
            if self.boundary_conditions == 'periodic':
                L = np.array([self.Lx, self.Ly])
                Db = norm(np.minimum(np.abs(Db), L - np.abs(Db)), axis=-1)
                Da = norm(np.minimum(np.abs(Da), L - np.abs(Da)), axis=-1)
            
            elif self.boundary_conditions == 'free':
                Db = norm(Db, axis=-1)
                Da = norm(Da, axis=-1)
                
            if self.cutoff_length:
                Db = Db[Db < self.cutoff_length]
                Da = Da[Da < self.cutoff_length]

            DeltaE = self.J * sum(Da**-3 - Db**-3)
                        
            if DeltaE < 0 :
                accepted = True
            else:
                accepted = (rand() < exp(- DeltaE / self.T))
            if accepted:
                self.energy += DeltaE
                move_status = 'accepted'
                self.spin_orientations[colloid] *= -1
                self.colloid_positions[colloid] = R_after
                liste_spins.append(colloid)
            else:
                move_status = 'rejected'
                
            if self.verbose:
                print("MC SSF move {} / {}, move {} (DeltaE={}), E/N={}".format(
                    i, 
                    nb_steps,
                    move_status, 
                    DeltaE,
                    self.energy / N_particles
                    )
            )
            E_MC.append(self.energy)
        self.calculate_neighbours_dict()
        return E_MC, len(liste_spins)




    def run_MC_step_loop(self):
        N1, N2 = self.size
        spins, vertex = self.choose_loop()
        nb_spins_flipped = 0
        if len(spins) > 2:
            new_pos = np.copy(self.colloid_positions)
            for s in spins:
                new_pos[s] = 2 * self.double_well_centers[s] - self.colloid_positions[s]
            DeltaE = self.calculate_interaction_energy_group(
                    new_pos.reshape((N1 * N2 * 3, 2))
                ) - self.calculate_interaction_energy_group(
                    self.colloid_positions.reshape((N1 * N2 * 3, 2))
                )
            
            if DeltaE < 0 :
                accepted = True
            else:
                accepted = (rand() < exp(- DeltaE / self.T))
    
            if accepted:
                move_status = 'accepted'

                self.energy += DeltaE
                for i, s in enumerate(spins):
                    v = vertex[i]
                    self.spin_orientations[s] *= -1
                    self.colloid_positions[s] = new_pos[s]
                    self.spin_to_vertex_dict[s] = self.spin_to_vertex_dict[s][::-1]
                    self.vertex_to_spin_dict[v]['out'].remove(s)
                    self.vertex_to_spin_dict[v]['in'].append(s)
                    self.vertex_to_spin_dict[v]['in'].remove(spins[i-1])
                    self.vertex_to_spin_dict[v]['out'].append(spins[i-1])
                nb_spins_flipped = len(spins)
            else:
                move_status = 'rejected'
                
            if self.verbose:
                print("MC loop, move {} (DeltaE={}), E/N={}".format(
                        move_status, 
                        DeltaE,
                        self.energy / (N1 * N2 * 3)
                        )
                )
        return nb_spins_flipped
    
    def run_MC_cycle(self, N_cycles=1, N_SSF=10, N_loop=1, 
                     N_particle_move=100, step_particle=0.02
                     ):
        E_MC = []
        nb_spins_flipped_SSF = 0
        nb_spins_flipped_loop = 0
        for i in range(N_cycles):
            E, n = self.run_MC_step_SSF(nb_steps=N_SSF)
            E_MC += E
            nb_spins_flipped_SSF += n
            for k in range(N_loop):
                nb_spins_flipped_loop += self.run_MC_step_loop()
                E_MC.append(self.energy)
            E, pc = self.run_MC_step_particle_move(
                        nb_steps=N_particle_move, step_move=step_particle
                    )
            E_MC += E
        return E_MC, nb_spins_flipped_SSF, nb_spins_flipped_loop, pc


    def calculate_interaction_energy(self, R1, R2):
        if self.boundary_conditions == 'free':
            d = norm(R1 - R2)
    
        elif self.boundary_conditions == 'periodic':
            d = abs(R1 - R2)
            L = np.array([self.Lx, self.Ly])
            d = norm(np.minimum(d, L - d))
        
        if self.cutoff_length:
            if d < self.cutoff_length:     
                return self.J / d**3
            else:
                return 0

    
    def calculate_interaction_energy_group(self, R):
        """Calculate the interaction energy between each pair of a group of particles"""
        if self.boundary_conditions == 'free':
            d = spa.distance.pdist(R)
        elif self.boundary_conditions == 'periodic':
            dx = spa.distance.pdist(R[:, 0][:, np.newaxis])
            dy = spa.distance.pdist(R[:, 1][:, np.newaxis])
            d = np.sqrt(np.minimum(dx, self.Lx - dx)**2 + 
                        np.minimum(dy, self.Ly - dy)**2
                    )
        
        if self.cutoff_length:
            d = d[d < self.cutoff_length]
        
        return self.J * sum(
                d**-3
            )

    def calculate_energy(self):
        N1, N2 = self.size
        pos = self.colloid_positions.reshape((N1 * N2 * 3, 2))
        E_int = self.calculate_interaction_energy_group(pos)
            
        E_pot = sum(
                self.calculate_potential(
                        c, 
                        self.colloid_positions[c]
                        )
                for c in itertools.product(range(N1), range(N2), range(3))
                )
        return E_int + E_pot




        

    


