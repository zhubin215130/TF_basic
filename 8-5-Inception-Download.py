import tensorflow as tf
import os
import tarfile
import requests

inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# location to save model
inception_pretrain_model_dir='./inception/inception_model'
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)

#get name and file path
filename=inception_pretrain_model_url.split('/')[-1]
filepath=os.path.join(inception_pretrain_model_dir,filename)

#download model
if not os.path.exists(filepath):
    print('download:',filename)
    r=requests.get(inception_pretrain_model_url,stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print('finish: ',filename)

#unzip file
tarfile.open(filepath,'r:gz').extractall(inception_pretrain_model_dir)

log_dir='./inception/inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


inception_pretrain_model_file=os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')
with tf.Session() as sess:
    #create graph to save model
    with tf.gfile.FastGFile(inception_pretrain_model_file,'rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,name='')
    #save graph structure
    writer=tf.summary.FileWriter(log_dir,sess.graph)
    writer.close()




