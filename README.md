# DSNCA
This project utilizes 3D reconstruction for micro-expression recognition.
## Requirements
my computer platform
Ubuntu 20.04  
CuDNN 11.4
recommend env
* Python 3.7 (numpy, skimage, scipy, opencv)  
* PyTorch >= 1.6 (pytorch3d)   

## Usage
Preprocess the SMIC dataset to detect the apex frames.
smic_processing.py: Preprocess the SMIC dataset: detect the apex frames.

To reproduce the results in the log file, run the following scripts:
```bash
python train_me_loso.py
python get_result_log.py
```
The filenames of the code specifically represent the following:
* "baseline" in the filename indicates that the baseline code is being run.
* "casme" in the filename means that the code is trained on the CASME dataset. You can also run this code by simply changing the dataset name to SMIC if desired.
* "samm"  in the filename indicates that the code is trained on the SAMM dataset.
* "dual" in filename means dual stream
* "3d"   in the filename means that 3D reconstruction is being used..
* "composite"in the filename means using composite dataset which consists of the SMIC, SAMM, and CASME datasets
## 3D face reconstruction can eliminate obstacles such as glasses, thereby improving recognition accuracy
<img width="113" alt="image" src="https://github.com/user-attachments/assets/6d898362-33ef-4d89-a781-098d3938d981" />
the original frontal face photo

<img width="112" alt="image" src="https://github.com/user-attachments/assets/f5f4c467-36fb-4553-a0e2-b3c132397603" />
the frontal view of the 3D face reconstruction.
there is no glass and restored facial details.

## There is evidence that algorithms can focus more on high-information-value areas.
<img width="232" alt="image" src="https://github.com/user-attachments/assets/41ea8a4f-ab45-44b3-9b7f-072de583fc3c" />
## The log file result
after you run the code,the result_log.csv will record the result,you will find it in a result filefolder
> `get_result_log.csv`.

## Referenced other projects
3D face restruction
https://github.com/yfeng95/DECA/tree/master

CapsuleNet
https://github.com/davidnvq/me_recognition/tree/master
## NOTE
Please email me for inquiries，mapp@mails.ccnu.edu.cn
