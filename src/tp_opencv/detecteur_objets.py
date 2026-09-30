import argparse

import cv2
import numpy as np


def detecter(img, aire_min=100, inverse=False):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flou = cv2.GaussianBlur(gris, (5, 5), 0)
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, binaire = cv2.threshold(flou, 0, 255, mode + cv2.THRESH_OTSU)
    binaire = cv2.morphologyEx(binaire, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binaire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > aire_min]


def annoter(img, contours):
    sortie = img.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(sortie, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(sortie, f"Objets : {len(contours)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return sortie


def main():
    parser = argparse.ArgumentParser(description="Détecte et compte les objets d'une image.")
    parser.add_argument("image", help="image en entrée")
    parser.add_argument("-o", "--sortie", default="resultat.png")
    parser.add_argument("--aire-min", type=float, default=100)
    parser.add_argument("--inverse", action="store_true", help="objets plus sombres que le fond")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Impossible de charger {args.image}")

    contours = detecter(img, args.aire_min, args.inverse)
    cv2.imwrite(args.sortie, annoter(img, contours))
    print(f"{len(contours)} objet(s) détecté(s) -> {args.sortie}")


if __name__ == "__main__":
    main()
