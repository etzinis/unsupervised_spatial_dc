"""!
@brief Get some random sampling for the position of two sources


@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import numpy as np


class RandomCirclePositioner(object):
    """
    ! Returns n_source_pairs positions based on a circle with
    specified radius Cartessian and Polar coordinates like follows:

    For each pair of the list we get a dictionary of:
    {
        'thetas': angles in rads [<(+x, s1), <(+x, s2)] in list,
        'd_theta': < (+x, s2) - < (+x, s1),
        'cartessian_positions': [(x_1, y_1), (x_2, y_2)],
        'distances': [[||si-mj||]] sources are rows and mics are
        columns
    }
    (theta_of_source_1, theta_of_source_1)



                        s2     OOO ooo
                                       OOo
               oOO
            oOO                               s1
          oOO
        oOO                                       OOo
       oOO                                         OOo
      oOO                                           OOo
     oOO                                             OOo
     oOO                                             OOo
     oOO          m1 <-- mic_distance --> m2     ================>>+x
     oOO                                             OOo
     oOO                                             OOo
      oOO                                           OOo
       oOO                                         OOo
        oOO                                       OOo
          oOO                                   OOo
            oO                                OOo
               oOO                         OOo
                   oOO                 OOo
                       ooo OOO OOO ooo
    """

    def __init__(self,
                 min_angle=0.,
                 max_angle=np.pi,
                 radius=1.0,
                 mic_distance_to_radius=0.001):

        self.min_angle = min_angle
        self.max_angle = max_angle
        self.radius = radius
        self.mic_distance = 1.0 * self.radius * mic_distance_to_radius
        self.mics = [(-self.mic_distance / 2, 0.),
                     (self.mic_distance / 2, 0.)]


    def get_cartessian_positions(self):
        pass


    def get_angles(self, n_source_pairs):
        d_thetas = [theta for theta in
                    np.random.uniform(low=self.min_angle,
                                      high=self.max_angle,
                                      size=n_source_pairs)]

        """ Also we have to place this angle arbitrarily over the 
        half circle in order to get the positions of the two sources"""
        thetas1 = [np.random.uniform(low=0.0,
                                     high=self.max_angle - d_theta)
                   for d_theta in d_thetas]

        thetas_1_d = zip(thetas1, d_thetas)
        thetas_1_2 = [(theta1, theta1+d_theta)
                      for (theta1, d_theta) in thetas_1_d]

        return thetas_1_2, d_thetas

    def get_sources_locations(self, n_source_pairs):
        posi
        thetas, d_thetas = self.get_angles(n_source_pairs)




if __name__ == "__main__":
    print("yolo ")
    random_positioner = RandomCirclePositioner()
    positions_info = random_positioner.get_sources_locations(4)
    print(positions_info)
