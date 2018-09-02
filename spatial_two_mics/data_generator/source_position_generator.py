"""!
@brief Get some random sampling for the position of two sources


@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import numpy as np
from scipy.spatial import distance as dst


class RandomCirclePositioner(object):
    """
    ! Returns n_source_pairs positions based on a circle with
    specified radius Cartessian and Polar coordinates like follows:

    For each pair of the list we get a dictionary of:
    {
        'thetas': angles in rads [<(+x, s1), <(+x, s2)] in list,
        'd_theta': < (+x, s2) - < (+x, s1),
        'xy_positons': [(x_1, y_1), (x_2, y_2)], Cartessian
        'distances': [[||si-mj||]] sources are rows and mics are
        columns
    }
    (theta_of_source_1, theta_of_source_1)



                        s2     OOO ooo
                                       OOo    (x1, y1)
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
                 radius=3.0,
                 mic_distance_percentage=0.01,
                 sound_speed=343,
                 max_mixture_ratio=0.8,
                 min_mixture_ratio=0.2,
                 fs=16000):
        """
        :param min_angle: minimum angle in rads for the 2 sources
        :param max_angle: maximum angle in rads for the 2 sources
        :param radius: Radius of the circle in **meters**
        :param mic_distance_percentage: Percentage of the radius
        corresponding to the distance between the two microphones
        :param sound_speed: Default 343 m/s in 20oC room temperature

        :param max_mixture_ratio: m2(t) = a1*s1(t+d1) + a2*s2(t+d2)
        :param min_mixture_ratio: a1 \in Uniform[.3, .7] and a2 = 1 - a1
        :param fs: sampling ratio in Hz
        """

        self.min_angle = min_angle
        self.max_angle = max_angle
        self.radius = radius
        self.mic_distance = self.radius * mic_distance_percentage
        self.m1 = (-self.mic_distance / 2, 0.)
        self.m2 = (self.mic_distance / 2, 0.)
        self.sound_speed = sound_speed
        self.min_mixture_ratio = min_mixture_ratio
        self.max_mixture_ratio = max_mixture_ratio
        self.fs = fs

    @staticmethod
    def get_cartessian_position(radius,
                                angle):
        return radius * np.cos(angle), radius * np.sin(angle)

    def get_time_delays_for_sources(self,
                                    distances):
        taus = {"tau1": (distances['s1m1']-distances['s1m2']),
                "tau2": (distances['s2m1'] - distances['s2m2'])}
        for tau in taus:
            taus[tau] *= (1. * self.fs) / self.sound_speed
            taus[tau] = int(taus[tau])

        return taus

    def compute_distances_for_sources_and_mics(self, s1, s2):
        """! si must be in format (xi, yi)"""
        points = {'s1': s1, 's2': s2, 'm1': self.m1, 'm2': self.m2}
        distances = {}

        for point_1, xy1 in points.items():
            for point_2, xy2 in points.items():
                distances[point_1+point_2] = dst.euclidean(xy1, xy2)

        return distances

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
        thetas, d_thetas = self.get_angles(n_source_pairs)
        xys = [(self.get_cartessian_position(self.radius, th_1),
                self.get_cartessian_position(self.radius, th_2))
                for (th_1, th_2) in thetas]

        distances = [self.compute_distances_for_sources_and_mics(s1, s2)
                     for (s1, s2) in xys]

        taus = [self.get_time_delays_for_sources(loc_dists)
                for loc_dists in distances]
        print(taus)
        # mix_amplitudes
        # time_delays = sel

        sources_locations = [
                             {'thetas': thetas[i],
                              'd_theta': d_thetas[i],
                              'xy_positons': xys[i],
                              'distances': distances[i]}
                             for i in np.arange(len(thetas))
        ]

        return sources_locations


if __name__ == "__main__":
    random_positioner = RandomCirclePositioner()
    positions_info = random_positioner.get_sources_locations(2)
    from pprint import pprint
    # pprint(positions_info)
