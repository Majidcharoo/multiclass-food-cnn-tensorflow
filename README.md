##Multi-Class Food Classification CNN Model using Tensor-Flow & Keras (6-Food Categories)
##Author : Majid Charoo / Dated : 19-07-2023

## Prerequisites:
- Python 3.1  & above
- Tensorflow
- Keras

## Getting started
To get the server running locally:

- Clone this repo
- A: Instructions to Explore Source-Code: 
	-To view & Explore the code ( Data Processing , Parameter-Setup , Model Creation & Meterics buidling )
	 	-A1. Open the Source file : "food_source.py"
		-A2. Set path of the image Directories ( food Images ) &  Run/Execute the code

-B: Instructions for simply testing an image: 
	-To test a image
	-B1. OPen the python file : "test_image.py"
	-B2. Set the Path for the Image to be tested &  Run/Execute the code
-C: Instructions for trained-Model Embedding / Utilization of embedded into other code: 
	C1. Keep " foodcm.pkl " in the CWD/Directory to use the already trained model
	C2. Load Model from Pickle-File & Test any given image by providing image path in following code
	C3.  Copy Follwing Code ( 9-Lines of Code):

		pickled_model = pickle.load(open( 'CWD/Directory_path','rb'))
		img_path = r'C:\Users\HP 840\Desktop\a-test.JPG'
  	  	img = image.load_img(img_path, target_size=(100, 100))
    		image_array = image.img_to_array(img)
    		image_array = image.img_to_array(img)
    		x_train = np.expand_dims(image_array, axis=0)
    		pred = pickled_model.predict(x_train)
    		pred = np.argmax(pred, axis=-1)
    		print(class_names[pred[0]]) 


-----------------------------------------------------------------END-------------------------------------------------------------------------








