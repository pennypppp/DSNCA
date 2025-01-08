# DSNCA_MER
## NOTE
please mail to me for inquiries
## Requirements
my computer platform
Ubuntu 16.04 Python 3.6 Keras 2.0.6 Opencv 3.1.0 pandas 0.19.2 CuDNN 5110
recommend env
* Python 3.7 (numpy, skimage, scipy, opencv)  
* PyTorch >= 1.6 (pytorch3d)  
* face-alignment (Optional for detecting face)
* 
## The log file result
after you run the code,the result_log.csv will record the result,you will find it in a result filefolder
> `result_log.csv`.

## Referenced other projects
3D face restruction
https://github.com/yfeng95/DECA/tree/master

CapsuleNet
https://github.com/davidnvq/me_recognition/tree/master
## Usage
Preprocess the SMIC dataset: detect the apex frames.
smic_processing.py: Preprocess the SMIC dataset: detect the apex frames.

Run the following script to reproduce the result in the log file.
run the code as followed
```bash
python train_me_loso.py
python get_result_log.py
```


the code filename specifically represented as follow
* "baseline" in filename means run baseline code
* "casme" in filename means training on the CASME dataset.you can also run this code just change the dataset name of SMIC
* "samm" in filename means training on the SAMM dataset
* "dual" in filename means dual stream
* "3d" in filename means using 3D reconstruction
* "composite"in filename means using composite dataset which consists of the SMIC, SAMM, and CASME datasets
## 3D face reconstruction can eliminate obstacles such as glasses, thereby improving recognition accuracy
<img width="113" alt="image" src="https://github.com/user-attachments/assets/6d898362-33ef-4d89-a781-098d3938d981" />
the original frontal face photo

<img width="112" alt="image" src="https://github.com/user-attachments/assets/f5f4c467-36fb-4553-a0e2-b3c132397603" />
the frontal view of the 3D face reconstruction.
there is no glass and restored facial details.


## There is evidence that algorithms can focus more on high-information-value areas.
<img width="232" alt="image" src="https://github.com/user-attachments/assets/41ea8a4f-ab45-44b3-9b7f-072de583fc3c" />
