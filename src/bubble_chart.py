import numpy as np
import pandas as pd

class BubbleChartPlotly:
    def __init__(self, labels, area, colors, bubble_spacing=10, plot_diameter=500):
        self.labels = labels
        self.colors = colors
        self.area = np.asarray(area)
        self.plot_diameter = plot_diameter
        self.plot_radius = plot_diameter / 2.5
        self.bubble_spacing = bubble_spacing
        total_area = np.sum(self.area)
        max_allowed_area = (np.pi * (self.plot_radius ** 2)) * 0.6
        scale_factor = max_allowed_area / total_area
        self.scaled_area = self.area * scale_factor
        self.radii = np.sqrt(self.scaled_area / np.pi)
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = self.radii
        self.bubbles[:, 3] = self.scaled_area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]
        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3])

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0], bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        return self.center_distance(bubble, bubbles) - bubble[2] - bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return np.argmin(distance, keepdims=True)

    def collapse(self, n_iterations=100):
        for _ in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                dir_vec = self.com - self.bubbles[i, :2]
                norm = np.linalg.norm(dir_vec)
                if norm == 0:
                    continue
                dir_vec = dir_vec / norm
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        norm = np.linalg.norm(dir_vec)
                        if norm == 0:
                            continue
                        dir_vec = dir_vec / norm
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        new_point1 = self.bubbles[i, :2] + orth * self.step_dist
                        new_point2 = self.bubbles[i, :2] - orth * self.step_dist
                        dist1 = self.center_distance(self.com, np.array([new_point1]))
                        dist2 = self.center_distance(self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()
            if moves / len(self.bubbles) < 0.05:
                self.step_dist /= 2

    def to_dataframe(self):
        return pd.DataFrame({
            'x': self.bubbles[:, 0],
            'y': self.bubbles[:, 1],
            'radius': self.bubbles[:, 2],
            'size': self.bubbles[:, 3],
            'label': self.labels,
            'color': self.colors,
            'count': self.area
        })
