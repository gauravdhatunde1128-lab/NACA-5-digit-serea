import numpy as np
import matplotlib.pyplot as plt

def generate_naca5(naca, num_points=200):

    if len(naca) != 5:
        raise ValueError("Enter a valid NACA 5-digit code")

    t = int(naca[3:5]) / 100.0

    x = np.linspace(0, 1, num_points)

    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    camber_table = {
        210: (0.0580, 361.4),
        220: (0.1260, 51.64),
        230: (0.2025, 15.957),
        240: (0.2900, 6.643),
        250: (0.3910, 3.230),
    }

    series = int(naca[:3])

    if series not in camber_table:
        raise ValueError(
            "Supported series: 210xx, 220xx, 230xx, 240xx, 250xx"
        )

    m, k1 = camber_table[series]

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    for i in range(len(x)):

        if x[i] < m:

            yc[i] = (
                k1 / 6
                * (
                    x[i]**3
                    - 3 * m * x[i]**2
                    + m**2 * (3 - m) * x[i]
                )
            )

            dyc_dx[i] = (
                k1 / 6
                * (
                    3 * x[i]**2
                    - 6 * m * x[i]
                    + m**2 * (3 - m)
                )
            )

        else:

            yc[i] = (
                k1 * m**3 / 6
                * (1 - x[i])
            )

            dyc_dx[i] = -k1 * m**3 / 6

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl


def plot_airfoil(naca_code):

    xu, yu, xl, yl = generate_naca5(naca_code)

    plt.figure(figsize=(10, 4))

    plt.plot(xu, yu, linewidth=2, label="Upper Surface")
    plt.plot(xl, yl, linewidth=2, label="Lower Surface")

    plt.title(f"NACA {naca_code}")
    plt.xlabel("Chord Position")
    plt.ylabel("Thickness")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":

    airfoil = input(
        "Enter NACA 5-digit airfoil (e.g. 23012): "
    )

    plot_airfoil(airfoil)
