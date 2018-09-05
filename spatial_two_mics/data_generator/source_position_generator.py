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
        'distances': [[||si-mj||]] all precomputed distances
        'taus': time delays in sample format
        'amplitudes': a1 and a2 for: m2(t) = a1*s1(t+d1) + a2*s2(t+d2)
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
                 fs=16000):
        """
        :param min_angle: minimum angle in rads for the 2 sources
        :param max_angle: maximum angle in rads for the 2 sources
        :param radius: Radius of the circle in **meters**
        :param mic_distance_percentage: Percentage of the radius
        corresponding to the distance between the two microphones
        :param sound_speed: Default 343 m/s in 20oC room temperature
        :param fs: sampling ratio in Hz
        """

        self.min_angle = min_angle
        self.max_angle = max_angle
        self.radius = radius
        self.mic_distance = self.radius * mic_distance_percentage
        self.m1 = (-self.mic_distance / 2, 0.)
        self.m2 = (self.mic_distance / 2, 0.)
        self.sound_speed = sound_speed
        self.fs = fs

    @staticmethod
    def get_cartessian_position(radius,
                                angle):
        return radius * np.cos(angle), radius * np.sin(angle)

    def get_amplifier_values_for_sources(self,
                                         n_sources):
        """
        :return: A dictionary of all the amplitudes in order to infer
        the final mixture depending on the weighted summation of the
        source-signals
        """
        alphas = np.random.uniform(low=0.1,
                                   high=1.0,
                                   size=n_sources)
        total_amplitude = sum(alphas)

        return dict([("a"+str(i+1), a/total_amplitude)
                     for (i, a) in enumerate(alphas)])

    def get_time_delays_for_sources(self,
                                    distances,
                                    n_sources):
        taus_list = []
        for i in np.arange(n_sources):
            source = "s"+str(i+1)
            taus_list.append(("tau"+str(i+1),
                              distances[source+"m1"]
                              - distances[source+"m2"]))
        taus = dict(taus_list)
        for tau in taus:
            taus[tau] *= (1. * self.fs) / self.sound_speed
            taus[tau] = int(taus[tau])

        return taus

    def compute_distances_for_sources_and_mics(self,
                                               source_points):
        """! si \in source_points must be in format (xi, yi)
        \:return a dictionary of all given points"""
        points = {"m1": self.m1, "m2": self.m2}
        points.update(dict([("s"+str(i+1), xy)
                            for (i, xy) in enumerate(source_points)]))
        distances = {}

        for point_1, xy1 in points.items():
            for point_2, xy2 in points.items():
                distances[point_1+point_2] = dst.euclidean(xy1, xy2)

        return distances

    def get_angles(self, n_source_pairs):
        d_thetas_list = np.random.uniform(low=self.min_angle,
                                          high=self.max_angle,
                                          size=n_source_pairs-1)
        total_angle = sum(d_thetas_list)
        if total_angle > self.max_angle:
            d_thetas_list = [(theta * self.max_angle) / total_angle
                             for theta in d_thetas_list]

        thetas = [0.]
        acc = 0.
        for angle in d_thetas_list:
            thetas.append(acc+angle)
            acc += angle

        return thetas, d_thetas_list

    def get_sources_locations(self,
                              n_source_pairs):
        """!
        Generate the positions, angles and distances for
        n_source_pairs of the same mixture corersponding to 2 mics"""
        thetas, d_thetas = self.get_angles(n_source_pairs)
        xys = []
        for angle in thetas:
            xys.append(self.get_cartessian_position(self.radius,
                                                    angle))

        distances = self.compute_distances_for_sources_and_mics(xys)

        taus = self.get_time_delays_for_sources(distances,
                                                n_source_pairs)

        mix_amplitudes = self.get_amplifier_values_for_sources(
                              n_source_pairs)

        sources_locations = {'thetas': thetas,
                             'd_thetas': d_thetas,
                             'xy_positons': xys,
                             'distances': distances,
                             'taus': taus,
                             'amplitudes': mix_amplitudes}

        return sources_locations


if __name__ == "__main__":
    random_positioner = RandomCirclePositioner()
    positions_info = random_positioner.get_sources_locations(4)
    from pprint import pprint
    pprint(positions_info)
