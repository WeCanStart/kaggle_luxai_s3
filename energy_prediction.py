import numpy as np

from base import get_opposite


class EnergyPrediction:
    def __init__(self) -> None:
        self.energies_calc: np.ndarray | None = None
        self.last_prediction: np.ndarray | None = None

    def calc_value_for_point(
        self, first_node: np.ndarray, second_node: np.ndarray, point: np.ndarray
    ) -> float:
        d1 = np.linalg.norm(first_node - point)
        d2 = np.linalg.norm(second_node - point)
        return 4 * (np.sin(d1 * 1.2 + 1) + np.sin(d2 * 1.2 + 1))

    def calc_energies(self) -> np.ndarray:
        if self.energies_calc is None:
            energies = np.zeros(shape=(24, 24, 24, 24), dtype=np.float32)
            i_arr, j_arr = np.meshgrid(
                np.arange(24, dtype=np.float32),
                np.arange(24, dtype=np.float32),
                indexing="ij",
            )

            all_points = np.dstack((i_arr, j_arr))

            for i in range(24):
                for j in range(24):
                    first_node = np.array([i, j])
                    second_node = np.array(get_opposite(i, j))

                    calc_value = lambda point: self.calc_value_for_point(
                        first_node, second_node, point
                    )
                    values = np.array(
                        [calc_value(point) for point in all_points.reshape(-1, 2)],
                        dtype=np.float32,
                    )

                    values = values.reshape(24, 24)
                    mean = values.mean()

                    if mean < 1.5:
                        offset = 1.5 - mean
                        values += offset

                    values = np.round(values)

                    energies[i, j] = np.copy(values)
                    # energies = energies.at[i, j].set(values) sosatb

            self.energies_calc = energies
        else:
            energies = self.energies_calc

        return energies

    def find_possible_energies(
        self, points: list, values: list
    ) -> tuple[np.ndarray, bool]:
        energies = self.calc_energies()

        possible_energies = np.zeros(shape=(0, 24, 24))
        for i in range(0, 24):
            for j in range(0, 24 - i):
                expected_values = np.array(
                    [energies[i, j][point] for point in points], dtype=np.float32
                )
                if np.array_equal(values, expected_values):
                    possible_energies = np.concatenate(
                        [possible_energies, energies[i, j][np.newaxis, :, :]], axis=0
                    )
        if len(possible_energies) == 1:
            sure_for_energies = True
        else:
            sure_for_energies = False
        self.last_prediction = possible_energies.mean(axis=0)

        return self.last_prediction, sure_for_energies
