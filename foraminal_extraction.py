import numpy as np
from scipy.ndimage import center_of_mass
from skimage.measure import regionprops
import nibabel as nib

def get_shifted_point_along_disk(disk_mask):
    """
    Calcule un point décalé depuis le centre du disque dans sa direction postérieure,
    avec une distance égale au rayon du disque selon son axe.
    
    Args:
        disk_mask: Masque binaire 3D du disque
    
    Returns:
        point: numpy array des coordonnées (x,y,z) du point décalé
        disk_radius: rayon calculé du disque selon son axe
        direction_vector: vecteur normalisé indiquant la direction du disque
    """
    # Obtenir les propriétés du disque
    props = regionprops(disk_mask.astype(int))[0]
    
    # Obtenir le centre du disque
    centroid = np.array(props.centroid)
    
    # Calculer l'orientation du disque (angle dans le plan sagittal)
    orientation = props.orientation  # en radians
    
    # Créer le vecteur direction (dans le plan sagittal)
    # Note: orientation = 0 correspond à l'axe y (AP)
    direction_vector = np.array([
        np.sin(orientation),  # composante x
        np.cos(orientation),  # composante y
        0                    # composante z
    ])
    
    # Pour aller vers postérieur (en LPI), on doit aller dans le sens positif
    # Si ce n'est pas le cas, on inverse le vecteur
    if direction_vector[1] < 0:  # si la composante y est négative
        direction_vector = -direction_vector
    
    # Calculer le rayon du disque dans la direction de son axe
    # En utilisant les points du masque projetés sur l'axe du disque
    mask_points = np.array(np.where(disk_mask)).T
    
    # Centrer les points
    centered_points = mask_points - centroid
    
    # Projeter les points sur l'axe du disque
    projections = np.abs(centered_points @ direction_vector)
    
    # Le rayon est la distance maximale du centre
    disk_radius = np.max(projections)
    
    # Calculer le point décalé
    shifted_point = centroid + direction_vector * disk_radius
    
    return shifted_point, disk_radius, direction_vector

def extract_oriented_patch(image_3d, disk_mask, patch_size=(64,64,64), shift_distance=20):
    """
    [Code précédent inchangé]
    """
    # ... [reste du code comme avant]

def visualize_orientation(image_3d, disk_mask, patch, shifted_point=None, slice_idx=None):
    """
    Visualise l'orientation du disque, le patch extrait et le point décalé
    """
    import matplotlib.pyplot as plt
    
    if slice_idx is None:
        slice_idx = image_3d.shape[2] // 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Afficher l'image originale avec le masque du disque
    ax1.imshow(image_3d[:, :, slice_idx], cmap='gray')
    ax1.imshow(disk_mask[:, :, slice_idx], alpha=0.3, cmap='red')
    
    # Afficher le point décalé si fourni
    if shifted_point is not None:
        ax1.plot(shifted_point[1], shifted_point[0], 'g*', markersize=10, label='Point décalé')
        ax1.legend()
    
    ax1.set_title('Image originale avec masque du disque')
    
    # Afficher le patch extrait
    ax2.imshow(patch[:, :, patch.shape[2]//2], cmap='gray')
    ax2.set_title('Patch extrait orienté')
    
    plt.show()