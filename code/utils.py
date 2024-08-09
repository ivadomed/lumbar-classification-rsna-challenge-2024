import pydicom as dicom
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

def get_bounding_box1(points):
    n = len(points)
    bbox = np.zeros((n, 4, 2))
    
    for i in range(n):
        if i==0:
            dist = np.linalg.norm(points[i] - points[i+1]) 
            u = points[i+1] - points[i]
            v = np.array([-points[i+1][1] + points[i][1], points[i+1][0] - points[i][0]])
            v /= np.linalg.norm(v)
            
            bbox[i, 0] = points[i] + u/2 + 2*v*dist
            bbox[i, 1] = points[i] + u/2 - 2*v*dist
            bbox[i, 2] = points[i] - u/2 - 2*v*dist
            bbox[i, 3] = points[i] - u/2 + 2*v*dist
        
        elif i<n-1:
            
            dist = np.linalg.norm(points[i+1] - points[i-1]) 
            u = points[i+1] - points[i-1]
            v = np.array([-points[i+1][1] + points[i-1][1], points[i+1][0] - points[i-1][0]])
            v /= np.linalg.norm(v)
            
            bbox[i, 0] = points[i] + u/4 + v*dist
            bbox[i, 1] = points[i] + u/4 - v*dist
            bbox[i, 2] = points[i] - u/4 - v*dist
            bbox[i, 3] = points[i] - u/4 + v*dist

        if i==n-1:
            dist = np.linalg.norm(points[i] - points[i-1]) 
            u = points[i] - points[i-1]
            v = np.array([-points[i][1] + points[i-1][1], points[i][0] - points[i-1][0]])
            v /= np.linalg.norm(v)
            
            bbox[i, 0] = points[i] + u/2 + 2*v*dist
            bbox[i, 1] = points[i] + u/2 - 2*v*dist
            bbox[i, 2] = points[i] - u/2 - 2*v*dist
            bbox[i, 3] = points[i] - u/2 + 2*v*dist
            
    return bbox 

def get_bounding_box2(img, points):
    h, w = img.shape
    n = len(points)
    bbox = np.zeros((n, 4, 2))
    
    v = w/4 * np.array([1, 0])
    for i in range(n):
        if i==0:
            dist = np.abs(points[i][1] - points[i+1][1]) 
            u = dist / 2 * np.array([0, 1])
            
            bbox[i, 0] = points[i] + u + v
            bbox[i, 1] = points[i] + u - v
            bbox[i, 2] = points[i] - u - v
            bbox[i, 3] = points[i] - u + v
        
        elif i<n-1:
            
            dist1 = np.linalg.norm(points[i+1] - points[i-1]) 
            dist2 = np.abs(points[i][1] - points[i-1][1]) 
            u = min(dist1, dist2) / 2 * np.array([0, 1])
            
            bbox[i, 0] = points[i] + u + v
            bbox[i, 1] = points[i] + u - v
            bbox[i, 2] = points[i] - u - v
            bbox[i, 3] = points[i] - u + v
        

        if i==n-1:
            dist = np.abs(points[i-1][1] - points[i][1]) 
            u = dist / 2 * np.array([0, 1])
            
            bbox[i, 0] = points[i] + u + v
            bbox[i, 1] = points[i] + u - v
            bbox[i, 2] = points[i] - u - v
            bbox[i, 3] = points[i] - u + v
        
            
    return bbox

if __name__=="__main__":
    description = pd.read_csv("./data/train_series_descriptions.csv")
    train = pd.read_csv("./data/train.csv")
    train = train.dropna() 
    coordinates = pd.read_csv("./data/train_label_coordinates.csv")
    df_count = coordinates.groupby(["study_id"]).size().reset_index(name="count")
    # clean
    df_count = df_count[df_count["count"]==25]
    study_ids = df_count.values[:,0]
    seq2cond = {
            "sag-T1": [
                "left_neural_foraminal_narrowing",
                "right_neural_foraminal_narrowing",
            ],
            "sag-T2": ["spinal_canal_stenosis"],
            "ax-T2": ["left_subarticular_stenosis", "right_subarticular_stenosis"],
        }
    
    study_id = 3832874334 # good example
    X, Y = [], []
    for (study_id, 
        series_id, 
        instance_number, 
        condition, 
        level, 
        x, 
        y) in coordinates[(coordinates["study_id"]==study_id)&(coordinates["condition"]=="Spinal Canal Stenosis")].values:
        X.append(x)
        Y.append(y)
        
    img = dicom.read_file(f"./data/train_images/{study_id}/{series_id}/{instance_number}.dcm")

    
    tmps = sorted(list(zip(X, Y)), key=lambda x : x[1])
    points = []
    for p in tmps:
        points.append(np.array(p))
        
    bbox1 = get_bounding_box1(points)
    bbox2 = get_bounding_box2(img.pixel_array, points)

    fig, ax = plt.subplots(ncols=2, figsize=(15, 8))
    for P in bbox1:
        P = list(P)
        P.append(P[0])
        P=np.array(P)
        ax[0].plot(P[:,0], P[:,1], lw=2)
            
    for P in bbox2:
        P = list(P)
        P.append(P[0])
        P=np.array(P)
        ax[1].plot(P[:,0], P[:,1], lw=2)
        
    ax[0].imshow(img.pixel_array, cmap="gray")
    ax[1].imshow(img.pixel_array, cmap="gray")
    ax[0].axis("off")
    ax[1].axis("off")
    ax[0].scatter(X, Y, c='indianred',lw=2, alpha=1)
    ax[1].scatter(X, Y, c='indianred',lw=2, alpha=1)

    ax[0].set_title("Boxes with sides following spine curve")
    ax[1].set_title("Boxes with sides parallel to image axes")

    print(f"subject {study_id} Sagittal T2")

    plt.savefig("bbox.png")
    plt.show()