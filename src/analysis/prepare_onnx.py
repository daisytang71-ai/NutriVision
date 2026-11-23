from pathlib import Path
import time
import torch
import numpy as np
import onnxruntime as ort
from run_color import load_models
from analysis_paths import ONNX_DIR
torch.set_num_threads(2)
p=ONNX_DIR; p.mkdir(parents=True,exist_ok=True)
x=torch.randn(16,3,224,224)
options=ort.SessionOptions(); options.intra_op_num_threads=4; options.inter_op_num_threads=1
for name,(m,mu,sd) in load_models('cpu').items():
    f=p/f'{name}.onnx'
    with torch.inference_mode():
        torch.onnx.export(m,x,f,input_names=['images'],output_names=['predictions'],dynamic_axes={'images':{0:'batch'},'predictions':{0:'batch'}},opset_version=17,dynamo=False)
        expected=m(x).numpy()
    session=ort.InferenceSession(str(f),sess_options=options,providers=['CPUExecutionProvider'])
    got=session.run(None,{'images':x.numpy()})[0]
    assert np.allclose(got,expected,rtol=1e-3,atol=1e-4),(name,np.max(np.abs(got-expected)))
    t=time.time(); session.run(None,{'images':x.numpy()}); session.run(None,{'images':x.numpy()})
    print(name,'ORT seconds / 16 images',(time.time()-t)/2,'max normalized prediction diff',np.max(np.abs(got-expected)),flush=True)
