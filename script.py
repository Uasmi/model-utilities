import os
import time
import re
import bisect
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import scipy.ndimage
import scipy.misc
import pickle
import tfutil
import matplotlib
import PIL
matplotlib.use('TkAgg')
import cv2
import misc
#import legacy
import config
import random
from scipy.fftpack import fft 

#-------------------#
import pyaudio
import struct
import wave
#-------------------#

class LegacyUnpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
    	#print (module)
    	if module == 'network' and name == 'Network':
            return tfutil.Network
    	return super().find_class(module, name)

def load_pkl(filename):
    with open(filename, 'rb') as file:
    	print (file)
    	return LegacyUnpickler(file, encoding='latin1').load()

def load_network_pkl():
    return load_pkl(str("network-snapshot-001590.pkl"))

def convert_to_pil_image(image, drange=[0,255]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB' if image.ndim == 3 else 'L'
    #print (image, format)
    return PIL.Image.fromarray(image, format)

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def generate_fake_images(G, D, Gs, input_array):
    run_id=12 
    snapshot=None
    grid_size=[1,1]
    num_pngs=1
    image_shrink=1 
    png_prefix=None 
    random_seed=1000 
    minibatch_size=8

    result_subdir = "test"
    png_idx = 1
    
    #print('Generating png %d / %d...' % (png_idx, num_pngs))
   
    #print("tensor", input_array.shape)
    latents = input_array
    #print("latents", latents.shape)
    labels = np.zeros([latents.shape[0], 0], np.float32)
    image = Gs.run(latents, labels, minibatch_size=minibatch_size, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        #misc.save_image_grid(images, os.path.join(result_subdir, '%s%06d.png' % (png_prefix, png_idx)), [0,255], grid_size)
        #open(os.path.join(result_subdir, '_done.txt'), 'wt').close()
        #png_idx += 1
    #print (image)
    return image

#tfutil.init_tf(config.tf_config)
with tf.Session(graph=tf.Graph()) as sess:
    network_pkl = str("network-snapshot-001590.pkl")
    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = load_network_pkl() 
    input_array = np.random.randn(1, *G.input_shape[1:]).astype(np.float32)
    print (input_array.shape)
    i = 0
    print(*G.input_shape[1:])

    """
    checkpoint = tf.train.get_checkpoint_state("my-model/")
    #print(checkpoint)
    input_checkpoint = checkpoint.model_checkpoint_path
    saver = tf.train.import_meta_graph('my-model/my-model.meta', clear_devices=True)
    saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, # The session is used to retrieve the weights
    tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
    ["Gs/images_out"] # The output node names are used to select the usefull nodes
    ) 

        # Finally we serialize and dump the output graph to the filesystem
    #with tf.gfile.GFile("frozen_model.pb", "wb") as f:
    #    f.write(output_graph_def.SerializeToString())
    #print("%d ops in the final graph." % len(output_graph_def.node))
    """

    Chunk = 1024 * 4
    Format = pyaudio.paInt16
    Channels = 2
    Rate = 44100
    print (G)
    generate_fake_images(G, D, Gs, input_array)
    
    #for op in tf.get_default_graph().get_operations():
    #    print(str(op.name))

    #saver = tf.train.Saver()
    #save_path = saver.save(sess, './my-model')

    p = pyaudio.PyAudio()
    p.get_default_input_device_info()

    """
    wf = wave.open('song3.wav','rb')

    stream = p.open(
        format = p.get_format_from_width(wf.getsampwidth()),
        channels = wf.getnchannels(),
        rate = wf.getframerate(),
        output = True,
        input = True,
        )

    audioData = wf.readframes(Chunk)

    while audioData != '':
        
        stream.write(audioData)
        audioData = wf.readframes(Chunk)
        """
    stream = p.open(
        format=Format,
        channels=Channels,
        rate=Rate,
        input=True,
        output=True,
        frames_per_buffer=Chunk
    )


    while True:
        start_time = time.time()
        audioData = stream.read(Chunk)  
        dataInt = struct.unpack(str(4 * Chunk) + 'B', audioData)
        fftData = fft(dataInt)
        fftArray = np.abs(fftData[0: Chunk]) * 2 / (256 * Chunk)
        fftArray = fftArray[5: 10]
        #print (fftArray.shape)
        #print (fftArray.max())
        data = generate_fake_images(G, D, Gs, input_array)
        data = data[0, :, :, :]
        data = convert_to_pil_image(data)
        cvimage = cv2.cvtColor(np.array(data), cv2.COLOR_RGB2BGR)
        #cvimage = cv2.resize(cvimage,(1024, 1024))
        #cv2.imshow("image", cvimage)
        #cv2.waitKey(10)
        input_array += random.uniform(-0.5,0.5)
        print("--- %s seconds ---" % (time.time() - start_time))

        if fftArray.max() > 0.8:
         #   print(dataInt.mean())
            input_array = np.random.randn(1, *G.input_shape[1:]).astype(np.float32)
            print(*G.input_shape[1:])
            
            