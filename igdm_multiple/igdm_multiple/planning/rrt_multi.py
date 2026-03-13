"""
Multi-source RRT planner extension.

Extends the base RRT with OIC/RSC/REC correction factors from:
Bai et al., "Autonomous radiation source searching using ADE-PSPF",
Robotics & Autonomous Systems, 2023.

Wraps the existing RRT class and adds multi-source branch information.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .rrt import RRT, Node
from ..estimation.multi_layer_pf import MultiLayerParticleFilter


class MultiSourceRRT(RRT):
    """
    RRT planner with multi-source correction factors.

    Inherits all tree-building from RRT. Overrides branch information
    calculation to add OIC, RSC, and REC corrections.
    """

    def __init__(self, occupancy_grid, N_tn, R_range, delta,
                 max_depth=4, discount_factor=0.8, positive_weight=0.5,
                 robot_radius=0.35,
                 oic_beta=0.5,
                 rsc_radius=1.0,
                 rec_radius=2.0):
        super().__init__(
            occupancy_grid=occupancy_grid,
            N_tn=N_tn,
            R_range=R_range,
            delta=delta,
            max_depth=max_depth,
            discount_factor=discount_factor,
            positive_weight=positive_weight,
            robot_radius=robot_radius
        )
        self.oic_beta = oic_beta
        self.rsc_radius = rsc_radius
        self.rec_radius = rec_radius

    def get_next_move_multi(self,
                            start_pos: Tuple[float, float],
                            multi_pf: MultiLayerParticleFilter,
                            measurement_positions: Optional[List[Tuple[float, float]]] = None
                            ) -> dict:
        """
        Get next move using multi-source corrected branch information.

        Uses the most uncertain (highest entropy) non-confirmed layer
        for mutual information calculation, then applies OIC/RSC/REC.
        """
        # Build tree (same as base)
        self.nodes = []
        self.sprawl(start_pos)
        paths = self.prune()

        if not paths:
            return {
                'next_position': start_pos,
                'best_path': [start_pos],
                'best_utility': -np.inf,
                'all_paths': [],
                'all_utilities': [],
                'tree_nodes': self.nodes.copy(),
                'num_branches': 0,
                'num_tree_nodes': len(self.nodes)
            }

        # Select planning layer (most uncertain non-confirmed)
        planning_layer = multi_pf.get_best_layer_for_planning()
        if planning_layer is None:
            # All confirmed, just use first layer
            planning_layer = multi_pf.layers[0]

        pf = planning_layer.pf
        confirmed = multi_pf.get_confirmed_sources()

        # Evaluate all branches
        all_bi = []
        best_path = None
        best_bi = -np.inf

        # Pre-compute measurement positions array for RSC
        if measurement_positions:
            meas_pos_array = np.array(measurement_positions)
        else:
            meas_pos_array = None

        # Pre-compute confirmed source positions for REC
        if confirmed:
            confirmed_pos_array = np.array([[s['x'], s['y']] for s in confirmed])
        else:
            confirmed_pos_array = None

        for path in paths:
            bi = self._calculate_branch_information_multi(
                path, pf, confirmed, meas_pos_array, confirmed_pos_array
            )
            all_bi.append(bi)
            if bi > best_bi:
                best_bi = bi
                best_path = path

        # Get next position
        if best_path is not None and len(best_path) > 1:
            next_position = best_path[1].position
        else:
            next_position = start_pos

        # Get estimated source from planning layer
        estimation, _ = pf.get_estimate()

        return {
            'next_position': next_position,
            'best_path': best_path if best_path else [],
            'best_utility': best_bi,
            'best_BI': best_bi,
            'best_entropy_gain': best_bi,
            'best_travel_cost': self.calculate_travel_cost(best_path, pf) if best_path else 0.0,
            'all_paths': paths,
            'all_utilities': all_bi,
            'all_branch_information': all_bi,
            'tree_nodes': self.nodes.copy(),
            'num_branches': len(paths),
            'estimated_source': (estimation['x'], estimation['y']),
            'start_position': start_pos,
            'sampling_radius': self.R_range,
            'max_depth': self.max_depth,
            'num_tree_nodes': len(self.nodes),
            'planning_layer': planning_layer.id
        }

    def _calculate_branch_information_multi(self,
                                             path: List[Node],
                                             pf,
                                             confirmed_sources: List[Dict],
                                             meas_pos_array: Optional[np.ndarray],
                                             confirmed_pos_array: Optional[np.ndarray]
                                             ) -> float:
        """
        Branch information with OIC, RSC, REC corrections.

        BI_multi = sum_i gamma^i * MI(v_i) * OIC(v_i) * RSC(v_i) * REC(v_i)
        """
        path = path[1:]  # Exclude root
        bi = 0.0

        for i, node in enumerate(path):
            pos = tuple(node.position)

            # Standard mutual information
            if node.entropy_gain != -np.inf:
                mi = node.entropy_gain
            else:
                current_entropy = pf.get_entropy()
                num_measurements = getattr(pf.sensor_model, 'num_levels', 2)

                expected_entropy = 0.0
                for measurement in range(num_measurements):
                    prob = pf.predict_measurement_probability(pos, measurement)
                    if prob > 1e-6:
                        hyp_entropy = pf.compute_hypothetical_entropy(measurement, pos)
                        expected_entropy += prob * hyp_entropy

                mi = current_entropy - expected_entropy
                node.entropy_gain = mi

            # OIC: Observation Intensity Correction
            oic = 1.0
            if confirmed_sources:
                # Higher correction where confirmed sources have strong signal
                # This encourages exploring areas NOT explained by confirmed sources
                total_confirmed_conc = 0.0
                for cs in confirmed_sources:
                    total_confirmed_conc += pf.dispersion_model.compute_concentration(
                        pos, (cs['x'], cs['y']), cs['Q']
                    )
                # Areas with low confirmed concentration are more interesting
                oic = 1.0 + self.oic_beta * max(0.0, 1.0 - total_confirmed_conc / 10.0)

            # RSC: Redundant Sampling Correction
            rsc = 1.0
            if meas_pos_array is not None and len(meas_pos_array) > 0:
                node_pos = np.array(pos[:2])
                dists = np.linalg.norm(meas_pos_array - node_pos, axis=1)
                min_dist = np.min(dists)
                rsc = 1.0 - np.exp(-min_dist ** 2 / (2.0 * self.rsc_radius ** 2))

            # REC: Repeat Exploring Correction
            rec = 1.0
            if confirmed_pos_array is not None and len(confirmed_pos_array) > 0:
                node_pos = np.array(pos[:2])
                dists = np.linalg.norm(confirmed_pos_array - node_pos, axis=1)
                min_dist = np.min(dists)
                rec = 1.0 - np.exp(-min_dist ** 2 / (2.0 * self.rec_radius ** 2))

            corrected_mi = mi * oic * rsc * rec
            bi += (self.discount_factor ** i) * corrected_mi

        return bi
