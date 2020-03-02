import tensorflow as tf
import pickle
import os
import sys
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

def load_network_pkl(input_file):
    return load_pkl(str(input_file))



if __name__ == "__main__":
    #globals()[sys.argv[1]](sys.argv[2])
    input_file = sys.argv[1]
    out_directory = sys.argv[2]
    with tf.Session(graph=tf.Graph()) as sess:
        network_pkl = str(input_file)
        print('Loading network from "%s"...' % network_pkl)
        G, D, Gs = load_network_pkl(network_pkl) 
     
        saver = tf.train.Saver()
        #saver = tf.get_default_graph()
        save_path = saver.save(sess, 'my-model/%s' % input_file)


        checkpoint = tf.train.get_checkpoint_state("my-model/")
        input_checkpoint = checkpoint.model_checkpoint_path
        saver = tf.train.import_meta_graph('my-model/%s.meta' %input_file, clear_devices=True)
        saver.restore(sess, input_checkpoint)

            # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, # The session is used to retrieve the weights
        tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
        ["Gs/images_out"] # The output node names are used to select the usefull nodes
        ) 

            # Finally we serialize and dump the output graph to the filesystem
        try:
            os.mkdir(out_directory)
        except:
            print ('directory exists')
        with tf.gfile.GFile("%s/frozen_model.pb" % out_directory, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        
