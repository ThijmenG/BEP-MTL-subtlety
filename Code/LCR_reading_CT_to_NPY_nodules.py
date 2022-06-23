# -*- coding: utf-8 -*-
"""
Creates a nodule meta file for LIDC-IDRI dataset, containing all the annotation info and saves all individual nodules
as an 3d image (.npy file).
"""

## imports
import pylidc as pl
import numpy as np
from pylidc.utils import volume_viewer
from pylidc.utils import consensus
from statistics import mean as mean
import pandas as pd
import os

## Variables
path_to_save=r'D:\Docs\BEP\LONG DATA\new_data'
path_to_LIDC_data= 'D:\Docs\BEP\LONG DATA\manifest-1600709154662\LIDC-IDRI'

def patient_nodule_builder(pid):
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    patient_id=pid[10:16]
    nods_set = scan.cluster_annotations()
       
    for i,nod in enumerate(nods_set):
        #creating basic nodule info
        nodule_name= f'{patient_id}_NOD_{i+1}'
        nr_annotators=len(nod)
        
        # create volume
        vol = nod[0].scan.to_volume(verbose=0)
        
        # calculate the mean masks
        cmask,cbbox,masks = consensus(nod, clevel=0.5)             
        img3d = vol[cbbox]
        mask3d= cmask
        shape = img3d.shape
        
        #save image and mask
        isExist = os.path.exists(os.path.join(path_to_save, 'Images2', f'Patient_{patient_id}'))
        if not isExist:
            os.makedirs(os.path.join(path_to_save, 'Images2', f'Patient_{patient_id}'))
            
        isExist = os.path.exists(os.path.join(path_to_save, 'Masks2', f'Mask_{patient_id}'))
        if not isExist:
            os.makedirs(os.path.join(path_to_save, 'Masks2', f'Mask_{patient_id}'))

        np.save(os.path.join(path_to_save, 'Images2', f'Patient_{patient_id}', f'img_{nodule_name}'), img3d)
        np.save(os.path.join(path_to_save, 'Masks2', f'Mask_{patient_id}', f'mask_{nodule_name}'), cmask)
        
        #saving all the augmentations
        malignacy=[]
        subtlety=[]
        internalstructure=[]
        calcification=[]
        sphericity=[]
        margin=[]
        lobulation=[]
        spiculation=[]
        texture=[]
        
        for j in range(len(nod)):
            malignacy.append(nod[j].malignancy)
            subtlety.append(nod[j].subtlety)
            internalstructure.append(nod[j].internalStructure)
            calcification.append(nod[j].calcification)
            sphericity.append(nod[j].sphericity)
            margin.append(nod[j].margin)
            lobulation.append(nod[j].lobulation)
            spiculation.append(nod[j].spiculation)
            texture.append(nod[j].texture)
        
        
        name_augmentation.append((nodule_name, nr_annotators, shape, mean(malignacy), mean(subtlety),
                                  mean(internalstructure),
                                  mean(calcification),
                                  mean(sphericity),
                                  mean(margin),
                                  mean(lobulation),
                                  mean(spiculation),
                                  mean(texture)))
                                  
        print(f'created {nodule_name}')
        #volume_viewer(img3d, cmask, ls='-', lw=2, c='r')
    return

PIDLIST=next(os.walk(path_to_LIDC_data))

name_augmentation = []

for i in range(len(PIDLIST)):
    patient_nodule_builder(PIDLIST[i])
    
#create a nodule excel info file
df = pd.DataFrame(name_augmentation, columns =['Nodule_name', 'nr_annotators', 'shape', 'malignacy', 'subtlety', 'internalstructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture'])
df.to_excel(os.path.join(path_to_save, 'Nodule_meta.xlsx'), index = False, header=True)

print('Done')
























