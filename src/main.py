"""
main.py – ejecuta todos los análisis PDP de golpe
"""
from pdp_bike import run as run_bike
from pdp_house import run as run_house


def main() -> None:
    print("=== Bike sharing PDPs ===")
    run_bike()

    print("\n=== House price PDPs ===")
    run_house()

    print("\nAnálisis completo. Figuras guardadas en la carpeta figures/.")


if __name__ == "__main__":
    main()