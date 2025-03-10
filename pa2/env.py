import numpy as np
import random
import matplotlib.pyplot as plt

class RaceTrack:
    def __init__(self, width=40, height=40, track_width=8, seed=None, move_up=0.45):
        self.width = width
        self.height = height
        self.track_width = track_width  # Controls how wide the track is
        self.track = np.zeros((height, width), dtype=int)
        self.seed = seed  # Store the seed
        
        if seed is not None:
            random.seed(seed)  # Set the random seed for reproducibility

        self.move_up = move_up
        self.__generate_track()
        # start line is the bottommost row of the track
        self.start_line = [(x, self.height - 1) for x in range(self.width) if self.track[self.height - 1, x] == 1]
        # finish line is the top right most cells of the track
        self.finish_line = [(self.width - 1, y) for y in range(self.height) if self.track[y, self.width - 1] == 1]
        self.finished = False


    def __generate_track(self):
        """Creates a race track that always connects the start to the finish with some randomness."""
        path = []
        x, y = 0, self.height  # Start position
        finish_x, finish_y = self.width, 5   # Finish position

        while (x, y) != (finish_x, finish_y):
            path.append((x, y))

            # Calculate preferred directions toward the finish
            move_right = x < finish_x
            move_up = y > finish_y

            # Randomly decide movement while ensuring progress
            if move_right and move_up:
                if random.random() < self.move_up:
                    x += 1  # self.move_up chance to move right
                else:
                    y -= 1  # 1 - self.move_up chance to move up
            elif move_right:
                x += 1
            elif move_up:
                y -= 1

            # Occasionally add slight diagonal movements with a chance
            if random.random() < 0.1 and move_right and move_up:
                x += 1
                y -= 1

        path.append((finish_x, finish_y))  # Ensure the final point is added

        # Expand the path to create a proper track width
        self.apply_track_width(path)

    def apply_track_width(self, path):
        """Expands the track width to make it look natural."""
        for x, y in path:
            # Randomly vary the track width within a range
            current_track_width = self.track_width + random.randint(-2, 2)
            for dx in range(-current_track_width // 2, current_track_width // 2 + 1):
                for dy in range(-current_track_width // 3, current_track_width // 3 + 1):  # More oval shape
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        self.track[ny, nx] = 1

    def display_track(self):

        """Displays the track using matplotlib."""
        plt.imshow(self.track, cmap="Greys", origin="upper")
        plt.title(f"Race Track (Seed: {self.seed})")
        plt.show()

    def check_crash(self, position, velocity):
        """Checks if the car has crashed into the track."""
        x, y = position
        vx, vy = velocity

        steps = max(abs(vx), abs(vy))
        for step in range(1, steps + 1):
            new_x = x + step * (vx / steps)
            new_y = y + step * (vy / steps)
            if not self.__is_valid_position((round(new_x), round(new_y))):
                return True
            # check if the car has crossed the finish line
            if (round(new_x), round(new_y)) in self.finish_line:
                self.finished = True
                return False


        return False
    
    def __is_valid_position(self, position):
        """Checks if a position is within the track boundaries and is a 1"""
        x, y = position


        return 0 <= x < self.width and 0 <= y < self.height and self.track[y, x] == 1