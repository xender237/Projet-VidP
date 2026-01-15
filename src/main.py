"""
Projet VidP : Pipeline hybride de stabilisation vidéo

Objectif :
Transformer une vidéo tremblante en une vidéo stable en estimant
le mouvement de la caméra et en appliquant une correction lissée.

Technologies :
- Python
- OpenCV
- NumPy
"""

import cv2
import numpy as np
import os


# ============================================================
# 1. LECTURE DE LA VIDÉO
# ============================================================

def read_video(video_path):
    """
    Ouvre la vidéo et récupère ses paramètres.

    :param video_path: chemin vers la vidéo d'entrée
    :return: cap (VideoCapture), fps, (width, height)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError("Erreur : impossible d'ouvrir la vidéo.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, fps, (width, height)


# ============================================================
# 2. DÉTECTION DE POINTS CARACTÉRISTIQUES
# ============================================================

def detect_features(gray_frame):
    """
    Détecte des points caractéristiques (Shi-Tomasi)
    sur une image en niveaux de gris.

    :param gray_frame: image grayscale
    :return: points caractéristiques
    """
    features = cv2.goodFeaturesToTrack(
        gray_frame,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=30,
        blockSize=3
    )
    return features


# ============================================================
# 3. ESTIMATION DU MOUVEMENT (OPTICAL FLOW)
# ============================================================

def estimate_motion(prev_gray, curr_gray, prev_features):
    """
    Estime le mouvement entre deux frames successives.

    :param prev_gray: frame précédente (grayscale)
    :param curr_gray: frame courante (grayscale)
    :param prev_features: points de la frame précédente
    :return: dx, dy, da (translation et rotation)
    """
    curr_features, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_features,
        None
    )

    # Filtrer les points valides
    valid_prev = prev_features[status == 1]
    valid_curr = curr_features[status == 1]

    # Estimation de la transformation affine
    matrix, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr)

    dx = matrix[0, 2]
    dy = matrix[1, 2]
    da = np.arctan2(matrix[1, 0], matrix[0, 0])

    return dx, dy, da


# ============================================================
# 4. LISSAGE DE LA TRAJECTOIRE
# ============================================================

def smooth_trajectory(trajectory, radius=10):
    """
    Applique un lissage par moyenne glissante
    à la trajectoire de la caméra.

    :param trajectory: liste de (x, y, angle)
    :param radius: taille de la fenêtre de lissage
    :return: trajectoire lissée
    """
    smoothed = []

    for i in range(len(trajectory)):
        start = max(0, i - radius)
        end = min(len(trajectory), i + radius)
        avg = np.mean(trajectory[start:end], axis=0)
        smoothed.append(avg)

    return smoothed


# ============================================================
# 5. STABILISATION D'UNE FRAME
# ============================================================

def stabilize_frame(frame, dx, dy, da):
    """
    Applique une transformation inverse pour stabiliser la frame.

    :param frame: image originale
    :param dx: translation horizontale
    :param dy: translation verticale
    :param da: rotation
    :return: frame stabilisée
    """
    h, w = frame.shape[:2]

    cos_a = np.cos(da)
    sin_a = np.sin(da)

    transform = np.array([
        [cos_a, -sin_a, dx],
        [sin_a,  cos_a, dy]
    ])

    stabilized = cv2.warpAffine(frame, transform, (w, h))
    return stabilized


# ============================================================
# 6. FONCTION PRINCIPALE (PIPELINE COMPLET)
# ============================================================

def main():
    """
    Pipeline complet de stabilisation vidéo :
    - Lecture vidéo
    - Estimation du mouvement
    - Lissage
    - Reconstruction vidéo
    """

    input_path = "data/input_video.mp4"
    output_path = "output/stabilized_video.mp4"

    cap, fps, size = read_video(input_path)

    # Étape 1 : calcul de la trajectoire
    trajectory = []

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_features = detect_features(prev_gray)

    trajectory.append((0, 0, 0))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dx, dy, da = estimate_motion(prev_gray, curr_gray, prev_features)

        x, y, a = trajectory[-1]
        trajectory.append((x + dx, y + dy, a + da))

        prev_gray = curr_gray
        prev_features = detect_features(prev_gray)

    # Étape 2 : lissage
    smoothed_trajectory = smooth_trajectory(trajectory)

    # Étape 3 : reconstruction vidéo
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size
    )

    for i in range(len(smoothed_trajectory)):
        ret, frame = cap.read()
        if not ret:
            break

        dx, dy, da = smoothed_trajectory[i]
        stabilized = stabilize_frame(frame, -dx, -dy, -da)
        writer.write(stabilized)

    cap.release()
    writer.release()
    print("Vidéo stabilisée générée avec succès.")


# ============================================================
# 7. POINT D'ENTRÉE DU PROGRAMME
# ============================================================

if __name__ == "__main__":
    main()
