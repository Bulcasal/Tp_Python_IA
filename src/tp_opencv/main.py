import argparse
import os

import cv2
import matplotlib.pyplot as plt

LARGEUR_AFFICHAGE = 640


def montrer(titre, images, nom_fichier, no_gui):
    "Affiche une ou plusieurs images côte à côte avec matplotlib"
    fig, axes = plt.subplots(1, len(images), figsize=(3.5 * len(images), 3), squeeze=False)
    for ax, (sous_titre, im) in zip(axes[0], images):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(sous_titre)
        ax.axis("off")
    fig.suptitle(titre)
    fig.tight_layout()
    if no_gui:
        os.makedirs("sortie", exist_ok=True)
        fig.savefig(f"sortie/{nom_fichier}.png")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="TP OpenCV, étapes 1 à 4.")
    parser.add_argument("image", nargs="?", default="sunsettree.jpg")
    parser.add_argument("--no-gui", action="store_true", help="n'ouvre aucune fenêtre (sauvegarde dans sortie/)")
    args = parser.parse_args()

    if args.no_gui:
        plt.switch_backend("Agg")

    # Étape 1
    print("Étape 1")
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Impossible de charger {args.image}")

    hauteur, largeur, canaux = img.shape
    print(f"Hauteur={hauteur}, largeur={largeur}, canaux={canaux}")
    print(f"Type de données : {img.dtype}")

    if not args.no_gui:
        # Réduction pour l'affichage
        echelle = LARGEUR_AFFICHAGE / largeur
        apercu = cv2.resize(img, None, fx=echelle, fy=echelle, interpolation=cv2.INTER_AREA)
        cv2.imshow("Image originale (BGR) - appuyez sur une touche", apercu)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    montrer("Étape 1 : BGR vs RGB", [
        ("Sans conversion", img),
        ("Après conversion RGB", img_rgb),
    ], "etape1", args.no_gui)

    #  Étape 2
    print("\nÉtape 2")
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flou_gauss = cv2.GaussianBlur(gris, (5, 5), 0)
    flou_median = cv2.medianBlur(gris, 5)
    montrer("Étape 2 : filtrage", [
        ("Niveaux de gris", gris),
        ("Flou gaussien 5x5", flou_gauss),
        ("Filtre médian 5", flou_median),
    ], "etape2", args.no_gui)

    #  Étape 3
    print("\nÉtape 3")
    # Le logo est blanc sur fond bleu : objets plus clairs que le fond -> THRESH_BINARY
    _, binaire_manuel = cv2.threshold(flou_gauss, 200, 255, cv2.THRESH_BINARY)
    seuil_otsu, binaire_otsu = cv2.threshold(flou_gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Seuil manuel = 200, seuil d'Otsu calculé = {seuil_otsu:.0f}")
    print("Otsu : le seuil est calculé automatiquement")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Teinte H sous OpenCV : 0-179. Bleu ~ 100-130. Blanc = saturation faible + valeur forte.
    masque_bleu = cv2.inRange(hsv, (100, 100, 50), (130, 255, 255))
    masque_blanc = cv2.inRange(hsv, (0, 0, 200), (179, 60, 255))
    montrer("Étape 3 : seuillages", [
        ("Seuil manuel (200)", binaire_manuel),
        (f"Otsu ({seuil_otsu:.0f})", binaire_otsu),
        ("HSV : bleu (fond)", masque_bleu),
        ("HSV : blanc (logo)", masque_blanc),
    ], "etape3", args.no_gui)

    #Étape 4
    print("\nÉtape 4")
    contours, _ = cv2.findContours(binaire_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    retenus = [c for c in contours if cv2.contourArea(c) > 100]
    print(f"Contours bruts : {len(contours)}, retenus (aire > 100 px) : {len(retenus)}")

    resultat = img.copy()
    cv2.drawContours(resultat, retenus, -1, (0, 255, 0), 3)
    print(f"Nombre total d'objets détectés : {len(retenus)}")

    # Bonus : boîtes englobantes + identifiants
    resultat_bonus = img.copy()
    for i, c in enumerate(retenus, start=1):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(resultat_bonus, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(resultat_bonus, f"#{i}", (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    montrer("Étape 4 : contours", [
        ("Contours retenus", cv2.cvtColor(resultat, cv2.COLOR_BGR2RGB)),
        ("Bonus : boîtes + id", cv2.cvtColor(resultat_bonus, cv2.COLOR_BGR2RGB)),
    ], "etape4", args.no_gui)


if __name__ == "__main__":
    main()
