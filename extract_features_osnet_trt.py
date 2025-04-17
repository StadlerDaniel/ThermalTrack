import tensorrt as trt
import numpy as np
import os
import cv2
from tqdm import tqdm
import pickle
import time
import torch
import sys

import pycuda.driver as cuda
import pycuda.autoinit


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    
    def __repr__(self):
        return self.__str__()

class TrtModel:    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()
        self.input_size_bs1 = int(self.inputs[0].host.shape[0] / self.max_batch_size)
        self.output_size_bs1 = int(self.outputs[0].host.shape[0] / self.max_batch_size)
    
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume((1,) + self.engine.get_binding_shape(binding)[1:]) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream       
    
    def __call__(self,x:np.ndarray,batch_size=2):
        x = x.astype(self.dtype)
        
        np.copyto(self.inputs[0].host[:batch_size * self.input_size_bs1],x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.set_binding_shape(0, (batch_size,) + self.engine.get_binding_shape(0)[1:])
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        
        self.stream.synchronize()
        return self.outputs[0].host[:batch_size * self.output_size_bs1].reshape(batch_size,-1)


dets_dir = sys.argv[1]
sequences_dir = sys.argv[2]
trt_engine_path = sys.argv[3]

out_name = '_w_trt_osnet_features'
runtime_out_file = os.path.join(os.path.dirname(dets_dir), os.path.basename(dets_dir) + out_name + '_runtime.txt')
out_dir = os.path.join(os.path.dirname(dets_dir), os.path.basename(dets_dir) + out_name)

min_score = 0.2

max_batch_size = 32
feature_dim = 512
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]

model = TrtModel(trt_engine_path, max_batch_size=max_batch_size)
input_shape = model.engine.get_binding_shape(0)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

sequences = os.listdir(sequences_dir)

preproc_times = []
runtimes = []

for j, seq in enumerate(sequences):
    filename = seq + '.pkl'
    dets_path = os.path.join(dets_dir, filename)
    with open(dets_path, 'rb') as f:
        dets = pickle.load(f)
    
    seq_dir = os.path.join(sequences_dir, seq, 'thermal')
    images = sorted(os.listdir(seq_dir))
    
    assert len(images) == len(dets)
    
    dets_w_feats_seq = []
    
    for i in tqdm(range(len(images))):
        dets[i] = dets[i][dets[i][:, 4] >= min_score]
        
        xyxys = dets[i][:, :4]
        image = images[i]
        img = cv2.imread(os.path.join(seq_dir, image))
        
        frame_features = []
        crops = []
        start_time = time.perf_counter()
        for xyxy in xyxys:
            x1, y1, x2, y2 = xyxy.astype(int)
            crop = img[y1:y2, x1:x2, :]
            crop = cv2.resize(crop, (input_shape[-1], input_shape[-2]))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = crop.astype(np.float32)
            crop -= mean
            crop /= std
            crop /= 255.0
            crop = np.transpose(crop, axes=[2, 0, 1])
            crops.append(crop)
        crops = np.stack(crops)
        end_time = time.perf_counter()
        preproc_times.append(end_time - start_time)
        
        # warmup
        if j == 0 and i == 0:
            print('Performing warmup ...')
            for j in range(100):
                model(crops, crops.shape[0])
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        frame_features = model(crops, crops.shape[0])
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        runtimes.append(end_time - start_time)
        
        frame_features = frame_features / np.linalg.norm(frame_features, axis=1, keepdims=True)
        dets_w_feats = np.concatenate([dets[i], frame_features], axis=1)
        dets_w_feats_seq.append(dets_w_feats)
    
    out_path = os.path.join(out_dir, seq + '.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(dets_w_feats_seq, f)

mean_preproc_time = np.mean(preproc_times) * 1000
median_preproc_time = np.median(preproc_times) * 1000

mean_runtime = np.mean(runtimes) * 1000
median_runtime = np.median(runtimes) * 1000
        
with open(runtime_out_file, 'w') as f:
    f.write(f'Median preproc time image in ms:\t{median_preproc_time:.1f}\n')
    f.write(f'Median runtime per image in ms:\t{median_runtime:.1f}\n')
    f.write(f'Summed median time per image in ms:\t{median_runtime+median_preproc_time:.1f}\n')
