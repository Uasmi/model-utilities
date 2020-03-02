import tensorflow as tf
import argparse 
import numpy as np
import cv2
import PIL
import wave
import pyaudio
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph
def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data
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
    print (image, format)
    return PIL.Image.fromarray(image, format)


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="frozen_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))  
    # We access the input and output nodes 
    x = graph.get_tensor_by_name('prefix/Gs/latents_in:0')
    x2 = graph.get_tensor_by_name('prefix/Gs/labels_in:0')
    y = graph.get_tensor_by_name('prefix/Gs/images_out:0')
    
    p = pyaudio.PyAudio()
    p.get_default_input_device_info()
    wf = wave.open('inst16b.wav','rb')

    stream = p.open(
        format = p.get_format_from_width(wf.getsampwidth()),
        channels = wf.getnchannels(),
        rate = wf.getframerate(),
        output = True,
        input = False,
        )

    audioData = wf.readframes(Chunk)

    while audioData != '':
        
        stream.write(audioData)
        audioData = wf.readframes(Chunk)
	

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 
        latents = np.random.randn(1, 512).astype(np.float32)
        labels = np.zeros([latents.shape[0], 0], np.float32)
        y_out = sess.run(y, feed_dict = { x: latents, x2: labels})
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
data = y_out[0, :, :, :]
data = data * 127.5
data = data + 127.5
print(data)
#data = np.rint(data).clip(0, 255).astype(np.uint8)
#print (data)
#im = PIL.Image.fromarray(data, 'RGB')
data = convert_to_pil_image(data)

data.save('test.jpg', 'JPEG')
print(y_out.shape) # [[ False ]] Yay, it works!