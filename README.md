# Detection-Done-Using-Object-Detection-API-TENSORFLOW
## Following each steps of my ipynb you can create your own Custom object detection
### Google colab for gpu support

step -1 Create a new tensorflow directory
```bash
mkdir tensorflow
```

step -2 going to created directory 
```bash
cd tensorflow
```

step -3 clone tensorflow model
```bash
!git clone https://github.com/tensorflow/models.git
```

step -4 Go to research folder
```bash
cd models/research/
```

step -5 used to compile Protocol Buffers (.proto) files
```bash
!protoc object_detection/protos/*.proto --python_out=.
```

step -6 now clone another git to research
```bash
!git clone https://github.com/cocodataset/cocoapi.git
```

step -7 now go to PythonAPI
```bash
cd cocoapi/PythonAPI
```

step -8 run this command
```bash
!make
!cp -r pycocotools /tensorflow/models/research
```

step -9 Now go back to research path
```bash
cd /content/tensorflow/models/research
```

step -10 Before copying setup.py comment out keras from setup.py then copy
```bash
!cp object_detection/packages/tf2/setup.py .

step -11 Other dependencies
```bash
!python -m pip install .
```

step -12 check pip list
```bash
!pip list
```

step -13 Run for tensorflow gpu
```bash
pip install tensorflow[and-cuda]==2.14.0
```

step -14 Check for gpu
```bash
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())
```

step -15 check nvidia
```bash
!nvidia-smi
```

step -16 Number of GPU available
```bash
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

step -17 Back to research and test the model
```bash
!python object_detection/builders/model_builder_tf2_test.py
```

step -18 Follow tensorflow object detection api
```bash
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
follow above link and you will get all images ready with in 
```

step -19 Use Exported model from model training zoo and then make changes to pipeline and then start training with your dataset
```bash
python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config
```

step -20 check for tensorboard 
step -21 export your model 
stpe -22 Test using image with your trained model 

My Detection 
![1000116868](https://github.com/user-attachments/assets/b1c8dd78-b58c-4bce-974b-23edc7d3aa0e)
![1000116869](https://github.com/user-attachments/assets/188a55b7-cb50-4181-b199-7bed8499cbf0)
